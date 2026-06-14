from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Sequence, Tuple, Union

from openai import OpenAI

from .config import config
from .utils import logger


class LLMCallError(RuntimeError):
    """Raised when the LLM call fails or the response cannot be parsed."""


_clients: Dict[Tuple[str | None, str], OpenAI] = {}


@dataclass(frozen=True)
class LLMProviderCandidate:
    model: str
    api_key: str
    base_url: str | None

_PROFILE_ALIASES = {
    "default": "thinking",
    "think": "thinking",
    "thinking": "thinking",
    "generation": "thinking",
    "evolution": "thinking",
    "plan": "thinking",
    "research_plan": "thinking",
    "critic": "critic",
    "critique": "critic",
    "review": "critic",
    "reflection": "critic",
    "rank": "critic",
    "ranking": "critic",
    "meta_review": "critic",
}
_PROFILE_LLM_SUFFIXES = (
    "api_key",
    "llm_api_key",
    "openai_api_key",
    "openrouter_api_key",
    "model",
    "llm_model",
    "base_url",
    "llm_base_url",
    "openrouter_base_url",
    "providers",
)
_API_KEY_KEYS = ("api_key", "llm_api_key", "openai_api_key", "openrouter_api_key")
_MODEL_KEYS = ("model", "llm_model")
_BASE_URL_KEYS = ("base_url", "llm_base_url", "openrouter_base_url")


def _normalize_profile(profile: str | None) -> str:
    normalized = str(profile or "thinking").strip().lower().replace("-", "_")
    return _PROFILE_ALIASES.get(normalized, normalized or "thinking")


def _profile_prefixes(profile: str | None) -> List[str]:
    normalized = _normalize_profile(profile)
    prefixes = [normalized]
    if normalized == "thinking":
        prefixes.extend(["think", "generation"])
    elif normalized == "critic":
        prefixes.extend(["review", "reflection", "rank", "ranking"])
    return prefixes


def _profile_config(profile: str | None) -> Any:
    for prefix in _profile_prefixes(profile):
        block = config.get(f"{prefix}_llm")
        if isinstance(block, (dict, list)):
            return block
    return {}


def _first_profile_config_value(profile: str | None, keys: Sequence[str], default: Any = None) -> Any:
    block = _profile_config(profile)
    for key in keys:
        if key in block:
            return block[key]
    for prefix in _profile_prefixes(profile):
        for key in keys:
            top_level_key = f"{prefix}_{key}"
            if top_level_key in config:
                return config[top_level_key]
    return default


def _nonempty_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _config_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _mapping_first_value(mapping: Any, keys: Sequence[str]) -> str | None:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping:
            value = _nonempty_string(mapping.get(key))
            if value:
                return value
    return None


def _providers_from_block(block: Any) -> List[Dict[str, Any]]:
    if isinstance(block, list):
        return [item for item in block if isinstance(item, dict)]
    if not isinstance(block, dict):
        return []

    providers = block.get("providers")
    if isinstance(providers, list):
        return [item for item in providers if isinstance(item, dict)]

    if any(key in block for key in (*_API_KEY_KEYS, *_MODEL_KEYS, *_BASE_URL_KEYS)):
        return [block]
    return []


def _scoped_provider_from_top_level(profile: str | None) -> Dict[str, Any] | None:
    provider: Dict[str, Any] = {}
    for prefix in _profile_prefixes(profile):
        scoped: Dict[str, Any] = {}
        for key in _API_KEY_KEYS:
            value = _nonempty_string(config.get(f"{prefix}_{key}"))
            if value:
                scoped["api_key"] = value
                break
        for key in _MODEL_KEYS:
            value = _nonempty_string(config.get(f"{prefix}_{key}"))
            if value:
                scoped["model"] = value
                break
        for key in _BASE_URL_KEYS:
            value = _nonempty_string(config.get(f"{prefix}_{key}"))
            if value:
                scoped["base_url"] = value
                break
        if scoped:
            provider.update(scoped)
            break
    return provider or None


def _global_provider_defaults() -> Dict[str, Any]:
    provider: Dict[str, Any] = {}
    api_key = _nonempty_string(config.get("openai_api_key"))
    if api_key:
        provider["api_key"] = api_key
    base_url = _nonempty_string(config.get("openrouter_base_url"))
    if base_url:
        provider["base_url"] = base_url
    model = _nonempty_string(config.get("llm_model"))
    if model:
        provider["model"] = model
    return provider


def _profile_shared_api_key(profile: str | None) -> str | None:
    block = _profile_config(profile)
    value = _mapping_first_value(block, _API_KEY_KEYS)
    if value:
        return value

    for prefix in _profile_prefixes(profile):
        for key in _API_KEY_KEYS:
            value = _nonempty_string(config.get(f"{prefix}_{key}"))
            if value:
                return value
    return None


def _global_shared_api_key() -> str | None:
    return (
        _nonempty_string(os.getenv("LLM_API_KEY"))
        or _nonempty_string(os.getenv("OPENROUTER_API_KEY"))
        or _nonempty_string(config.get("openai_api_key"))
    )


def _provider_entries(profile: str | None) -> List[Dict[str, Any]]:
    entries = _providers_from_block(_profile_config(profile))
    if entries:
        return entries

    scoped_provider = _scoped_provider_from_top_level(profile)
    if scoped_provider:
        return [scoped_provider]

    global_provider = _global_provider_defaults()
    if global_provider:
        return [global_provider]
    return []


def _role_env_api_key(profile: str | None) -> str | None:
    for prefix in _profile_prefixes(profile):
        env_prefix = prefix.upper()
        for env_name in (
            f"{env_prefix}_LLM_API_KEY",
            f"{env_prefix}_OPENAI_API_KEY",
            f"{env_prefix}_OPENROUTER_API_KEY",
        ):
            value = _nonempty_string(os.getenv(env_name))
            if value:
                return value
    return None


def get_configured_api_key(profile: str | None = None) -> str | None:
    """Return the configured API key for an LLM profile without logging it."""

    profile_key = _role_env_api_key(profile) or _profile_shared_api_key(profile)
    if profile_key:
        return profile_key
    for provider in _provider_entries(profile):
        value = _mapping_first_value(provider, _API_KEY_KEYS)
        if value:
            return value
    return _global_shared_api_key()


def _get_client(base_url: str | None, api_key: str) -> OpenAI:
    api_key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    cache_key = (base_url, api_key_fingerprint)
    client = _clients.get(cache_key)
    if client is None:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            timeout=600,  # 10分钟超时
        )
        _clients[cache_key] = client
    return client


def _candidate_providers(requested_model: str | None, profile: str | None = None) -> List[LLMProviderCandidate]:
    normalized_profile = _normalize_profile(profile)
    shared_api_key = _role_env_api_key(profile) or _profile_shared_api_key(profile) or _global_shared_api_key()
    configured_entries = _provider_entries(profile)
    raw_entries = configured_entries or [{}]
    requested_primary_model = _nonempty_string(requested_model)
    ordered: List[LLMProviderCandidate] = []
    seen: set[Tuple[str, str, str | None]] = set()
    has_any_model = False
    has_any_api_key = False

    for index, entry in enumerate(raw_entries):
        configured_model = _mapping_first_value(entry, _MODEL_KEYS)
        model = requested_primary_model if index == 0 and requested_primary_model else configured_model
        if model:
            has_any_model = True

        api_key = _mapping_first_value(entry, _API_KEY_KEYS) or shared_api_key
        if api_key:
            has_any_api_key = True

        if not model or not api_key:
            continue

        base_url = _mapping_first_value(entry, _BASE_URL_KEYS)
        dedupe_key = (model, api_key, base_url)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        ordered.append(LLMProviderCandidate(model=model, api_key=api_key, base_url=base_url))

    if not has_any_model:
        raise LLMCallError(f"LLM model is not configured for the '{normalized_profile}' profile.")
    if not has_any_api_key:
        raise LLMCallError(f"LLM API key is not configured for the '{normalized_profile}' profile.")
    if not ordered:
        raise LLMCallError(f"No usable LLM providers are configured for the '{normalized_profile}' profile.")
    return ordered


def _normalize_messages(messages: Union[str, Sequence[Dict[str, str]]]) -> List[Dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    normalized = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        normalized.append({"role": role, "content": content})
    return normalized


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _json_response_sections(raw_text: str) -> List[str]:
    cleaned = _strip_code_fence(raw_text)
    sections = []
    think_end = cleaned.lower().rfind("</think>")
    if think_end >= 0:
        final_answer = _strip_code_fence(cleaned[think_end + len("</think>") :])
        if final_answer:
            sections.append(final_answer)
    if cleaned and cleaned not in sections:
        sections.append(cleaned)
    return sections


def extract_json_payload(
    raw_text: str,
    expected_type: type | tuple[type, ...] | None = None,
) -> Any:
    decoder = json.JSONDecoder()
    first_payload = None
    found_payload = False

    for cleaned in _json_response_sections(raw_text):
        try:
            payload = json.loads(cleaned)
            if expected_type is None or isinstance(payload, expected_type):
                return payload
            if not found_payload:
                first_payload = payload
                found_payload = True
        except json.JSONDecodeError:
            pass

        for index, character in enumerate(cleaned):
            if character not in "[{":
                continue
            try:
                payload, _ = decoder.raw_decode(cleaned[index:])
            except json.JSONDecodeError:
                continue
            if expected_type is None or isinstance(payload, expected_type):
                return payload
            if not found_payload:
                first_payload = payload
                found_payload = True

    if expected_type is not None and found_payload:
        expected_name = (
            " or ".join(candidate.__name__ for candidate in expected_type)
            if isinstance(expected_type, tuple)
            else expected_type.__name__
        )
        raise LLMCallError(
            f"No valid JSON {expected_name} found in model response; "
            f"the first JSON payload was {type(first_payload).__name__}."
        )

    raise LLMCallError("No valid JSON payload found in model response.")


def call_llm(
    messages: Union[str, Sequence[Dict[str, str]]],
    model: str | None = None,
    temperature: float = 0.7,
    profile: str | None = None,
) -> str:
    normalized_profile = _normalize_profile(profile)
    initial_delay = float(config.get("initial_retry_delay", 1))
    retry_until_success = _config_bool(config.get("llm_retry_until_success"), default=True)
    max_retries = None if retry_until_success else int(config.get("max_retries", 3))
    max_retry_delay = float(config.get("max_retry_delay_seconds", 60))
    last_error = "Unknown LLM failure"
    normalized_messages = _normalize_messages(messages)
    candidate_providers = _candidate_providers(model, normalized_profile)

    attempt = 0
    while max_retries is None or attempt < max_retries:
        attempt += 1
        attempt_label = str(attempt) if max_retries is None else f"{attempt}/{max_retries}"
        for provider in candidate_providers:
            try:
                client = _get_client(
                    base_url=provider.base_url,
                    api_key=provider.api_key,
                )
                logger.info(
                    "LLM call attempt %s using profile=%s model=%s base_url=%s temperature=%.2f",
                    attempt_label,
                    normalized_profile,
                    provider.model,
                    provider.base_url or "default",
                    temperature,
                )
                completion = client.chat.completions.create(
                    model=provider.model,
                    messages=normalized_messages,
                    temperature=temperature,
                    max_tokens=100000,
                )
                if completion.choices and completion.choices[0].message:
                    return completion.choices[0].message.content or ""
                last_error = "Response did not contain any choices."
            except Exception as exc:  # noqa: BLE001
                last_error = (
                    f"profile={normalized_profile} model={provider.model} "
                    f"base_url={provider.base_url or 'default'} error={exc}"
                )
                logger.warning(
                    "LLM call failed on attempt %s with profile=%s model=%s base_url=%s: %s",
                    attempt_label,
                    normalized_profile,
                    provider.model,
                    provider.base_url or "default",
                    exc,
                )

        if max_retries is None or attempt < max_retries:
            delay = min(max_retry_delay, initial_delay * (2 ** (attempt - 1)))
            logger.info("Retrying LLM call in %.1fs after failure: %s", delay, last_error)
            time.sleep(delay)

    raise LLMCallError(last_error)


def call_json(
    messages: Union[str, Sequence[Dict[str, str]]],
    model: str | None = None,
    temperature: float = 0.7,
    profile: str | None = None,
    expected_type: type | tuple[type, ...] | None = None,
) -> Any:
    response = call_llm(messages=messages, model=model, temperature=temperature, profile=profile)
    return extract_json_payload(response, expected_type=expected_type)
