from __future__ import annotations

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


_clients: Dict[Tuple[str | None, float, str], OpenAI] = {}

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
    "safety_review": "critic",
    "prior_art": "critic",
}
_MISSING = object()
_PROFILE_LLM_SUFFIXES = (
    "api_key",
    "llm_api_key",
    "openai_api_key",
    "openrouter_api_key",
    "model",
    "llm_model",
    "model_fallbacks",
    "llm_model_fallbacks",
    "base_url",
    "llm_base_url",
    "openrouter_base_url",
    "base_url_fallbacks",
    "llm_base_url_fallbacks",
    "timeout_seconds",
    "llm_timeout_seconds",
)


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


def _profile_config(profile: str | None) -> Dict[str, Any]:
    for prefix in _profile_prefixes(profile):
        block = config.get(f"{prefix}_llm")
        if isinstance(block, dict):
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


def _has_profile_specific_llm_config(profile: str | None) -> bool:
    for prefix in _profile_prefixes(profile):
        if isinstance(config.get(f"{prefix}_llm"), dict):
            return True
        for suffix in _PROFILE_LLM_SUFFIXES:
            if f"{prefix}_{suffix}" in config:
                return True
    return False


def _nonempty_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _string_list(value: Any) -> List[str]:
    if not value:
        return []
    if not isinstance(value, list):
        value = [value]
    return [item for item in (_nonempty_string(item) for item in value) if item]


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

    profile_key = _role_env_api_key(profile) or _nonempty_string(
        _first_profile_config_value(
            profile,
            ("api_key", "llm_api_key", "openai_api_key", "openrouter_api_key"),
        )
    )
    if profile_key:
        return profile_key
    return (
        _nonempty_string(os.getenv("LLM_API_KEY"))
        or _nonempty_string(os.getenv("OPENROUTER_API_KEY"))
        or _nonempty_string(config.get("openai_api_key"))
    )


def _get_client(base_url: str | None, timeout_seconds: float, profile: str | None = None) -> OpenAI:
    normalized_profile = _normalize_profile(profile)
    api_key = get_configured_api_key(normalized_profile)
    if not api_key:
        raise LLMCallError(f"LLM API key is not configured for the '{normalized_profile}' profile.")

    api_key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    cache_key = (base_url, timeout_seconds, api_key_fingerprint)
    client = _clients.get(cache_key)
    if client is None:
        # Keep retry behavior in this module so stalled provider calls do not
        # block the whole reproduction run indefinitely.
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )
        _clients[cache_key] = client
    return client


def _candidate_models(requested_model: str | None, profile: str | None = None) -> List[str]:
    configured_primary = _first_profile_config_value(profile, ("model", "llm_model"))
    primary = _nonempty_string(requested_model) or _nonempty_string(configured_primary) or _nonempty_string(config.get("llm_model"))
    configured_fallbacks = _first_profile_config_value(
        profile,
        ("model_fallbacks", "llm_model_fallbacks"),
        default=_MISSING,
    )
    if configured_fallbacks is _MISSING:
        fallback_source = [] if _has_profile_specific_llm_config(profile) else config.get("llm_model_fallbacks", [])
    else:
        fallback_source = configured_fallbacks
    fallbacks = _string_list(fallback_source)
    ordered: List[str] = []
    for model in [primary] + fallbacks:
        if model and model not in ordered:
            ordered.append(model)
    if not ordered:
        raise LLMCallError(f"LLM model is not configured for the '{_normalize_profile(profile)}' profile.")
    return ordered


def _candidate_base_urls(profile: str | None = None) -> List[str | None]:
    primary = _first_profile_config_value(
        profile,
        ("base_url", "llm_base_url", "openrouter_base_url"),
        default=_MISSING,
    )
    if primary is _MISSING:
        primary = config.get("openrouter_base_url")
    configured_fallbacks = _first_profile_config_value(
        profile,
        ("base_url_fallbacks", "llm_base_url_fallbacks"),
        default=_MISSING,
    )
    if configured_fallbacks is _MISSING:
        fallbacks = [] if _has_profile_specific_llm_config(profile) else config.get("llm_base_url_fallbacks", []) or []
    else:
        fallbacks = configured_fallbacks or []
    if not isinstance(fallbacks, list):
        fallbacks = [fallbacks]
    ordered: List[str | None] = []
    for value in [primary] + list(fallbacks):
        normalized = str(value).strip() if value is not None else None
        if normalized == "":
            normalized = None
        if normalized not in ordered:
            ordered.append(normalized)
    return ordered or [None]


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


def extract_json_payload(raw_text: str) -> Any:
    cleaned = _strip_code_fence(raw_text)
    decoder = json.JSONDecoder()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for index, character in enumerate(cleaned):
        if character not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
            return payload
        except json.JSONDecodeError:
            continue

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
    timeout_seconds = float(
        _first_profile_config_value(normalized_profile, ("timeout_seconds", "llm_timeout_seconds"))
        or config.get("llm_timeout_seconds", 60)
    )
    last_error = "Unknown LLM failure"
    normalized_messages = _normalize_messages(messages)
    candidate_models = _candidate_models(model, normalized_profile)
    candidate_base_urls = _candidate_base_urls(normalized_profile)

    if not get_configured_api_key(normalized_profile):
        raise LLMCallError(f"LLM API key is not configured for the '{normalized_profile}' profile.")

    attempt = 0
    while max_retries is None or attempt < max_retries:
        attempt += 1
        attempt_label = str(attempt) if max_retries is None else f"{attempt}/{max_retries}"
        for llm_model in candidate_models:
            for base_url in candidate_base_urls:
                try:
                    client = _get_client(base_url=base_url, timeout_seconds=timeout_seconds, profile=normalized_profile)
                    logger.info(
                        "LLM call attempt %s using profile=%s model=%s base_url=%s temperature=%.2f",
                        attempt_label,
                        normalized_profile,
                        llm_model,
                        base_url or "default",
                        temperature,
                    )
                    completion = client.chat.completions.create(
                        model=llm_model,
                        messages=normalized_messages,
                        temperature=temperature,
                    )
                    if completion.choices and completion.choices[0].message:
                        return completion.choices[0].message.content or ""
                    last_error = "Response did not contain any choices."
                except Exception as exc:  # noqa: BLE001
                    last_error = f"profile={normalized_profile} model={llm_model} base_url={base_url or 'default'} error={exc}"
                    logger.warning(
                        "LLM call failed on attempt %s with profile=%s model=%s base_url=%s: %s",
                        attempt_label,
                        normalized_profile,
                        llm_model,
                        base_url or "default",
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
) -> Any:
    response = call_llm(messages=messages, model=model, temperature=temperature, profile=profile)
    return extract_json_payload(response)
