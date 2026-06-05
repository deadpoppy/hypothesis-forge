from __future__ import annotations

import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Sequence, TypeVar

from .config import config


logging.basicConfig(
    level=config.get("logging_level", logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("aicoscientist")

T = TypeVar("T")
R = TypeVar("R")


def is_huggingface_space() -> bool:
    hf_env_vars = ["SPACE_ID", "SPACE_AUTHOR_NAME", "SPACES_BUILDKIT_VERSION", "HF_HOME"]
    return any(os.getenv(variable) for variable in hf_env_vars)


def get_deployment_environment() -> str:
    if is_huggingface_space():
        return "Hugging Face Spaces"
    if os.getenv("LOCAL_DEV") or not os.getenv("PORT"):
        return "Local Development"
    return "Unknown"


def filter_free_models(all_models: List[str]) -> List[str]:
    return [model for model in all_models if ":free" in model]


def get_max_concurrency(limit: int | None = None) -> int:
    configured = limit if limit is not None else config.get("max_concurrency", 8)
    try:
        value = int(configured)
    except (TypeError, ValueError):
        value = 8
    return max(1, min(32, value))


def run_concurrently(items: Sequence[T], worker: Callable[[T], R], max_workers: int | None = None) -> List[R]:
    if not items:
        return []

    worker_count = min(len(items), get_max_concurrency(max_workers))
    if worker_count <= 1:
        return [worker(item) for item in items]

    results: List[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {executor.submit(worker, item): index for index, item in enumerate(items)}
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

    return [result for result in results if result is not None]


def generate_unique_id(prefix: str = "H") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def coerce_string_list(items: Any) -> List[str]:
    """Normalize LLM list-ish output without splitting bare strings into characters."""
    seen = set()
    result = []

    if items is None:
        return result
    if isinstance(items, str):
        raw_items = [items]
    elif isinstance(items, dict):
        raw_items = [items]
    else:
        try:
            raw_items = list(items)
        except TypeError:
            raw_items = [items]

    def append_one(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, (list, tuple, set)):
            for nested in item:
                append_one(nested)
            return
        if isinstance(item, dict):
            normalized = json.dumps(item, ensure_ascii=False, sort_keys=True)
        else:
            normalized = str(item).strip()
        if not normalized:
            return
        dedupe_key = normalized.casefold()
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        result.append(normalized)

    for item in raw_items:
        append_one(item)
    return result


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    return coerce_string_list(items)


def rating_bucket(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 4.0:
        return "HIGH"
    if score >= 2.5:
        return "MEDIUM"
    return "LOW"


def generate_visjs_data(adjacency_graph: Dict[str, List[Dict[str, float]]], threshold: float = 0.2) -> Dict[str, list]:
    nodes = []
    edges = []
    seen_edges = set()
    for node_id, connections in adjacency_graph.items():
        nodes.append({"id": node_id, "label": node_id})
        for connection in connections:
            similarity = connection.get("similarity", 0.0)
            if similarity >= threshold:
                other_id = connection.get("other_id")
                edge_key = tuple(sorted([str(node_id), str(other_id)]))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edges.append(
                    {
                        "from": node_id,
                        "to": other_id,
                        "label": f"{similarity:.2f}",
                    }
                )
    return {"nodes": nodes, "edges": edges}


_sentence_transformer_model = None
_similarity_warning_emitted = False


class _TransformerEmbeddingModel:
    def __init__(self, model_name: str, local_files_only: bool) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.eval()

    def encode(self, text: str):
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with self.torch.no_grad():
            output = self.model(**encoded)
        token_embeddings = output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
        return pooled.squeeze(0)


def _resolve_embedding_model_name(model_name: str) -> str:
    if "/" in model_name:
        return model_name
    return f"sentence-transformers/{model_name}"


def get_sentence_transformer_model():
    global _sentence_transformer_model
    if _sentence_transformer_model is False:
        raise RuntimeError("Sentence transformer model is unavailable.")
    if _sentence_transformer_model is None:
        if not bool(config.get("sentence_transformer_enabled", True)):
            _sentence_transformer_model = False
            raise RuntimeError("Sentence transformer similarity is disabled by configuration.")

        model_name = _resolve_embedding_model_name(config.get("sentence_transformer_model", "all-MiniLM-L6-v2"))
        local_files_only = bool(config.get("sentence_transformer_local_files_only", True))
        logger.info(
            "Loading embedding model: %s (local_files_only=%s)",
            model_name,
            local_files_only,
        )
        try:
            _sentence_transformer_model = _TransformerEmbeddingModel(model_name, local_files_only)
        except Exception as exc:  # noqa: BLE001
            _sentence_transformer_model = False
            raise RuntimeError(f"Embedding model load failed: {exc}") from exc
    return _sentence_transformer_model


def lexical_similarity_score(text_a: str, text_b: str) -> float:
    tokens_a = {token for token in text_a.lower().split() if len(token) > 2}
    tokens_b = {token for token in text_b.lower().split() if len(token) > 2}
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    return overlap / len(tokens_a | tokens_b)


@lru_cache(maxsize=4096)
def _cached_embedding(text: str):
    model = get_sentence_transformer_model()
    return model.encode(text)


def similarity_score(text_a: str, text_b: str) -> float:
    global _similarity_warning_emitted
    if not text_a or not text_b:
        return 0.0

    try:
        embedding_a = _cached_embedding(text_a)
        embedding_b = _cached_embedding(text_b)
        embedding_a = embedding_a.reshape(-1)
        embedding_b = embedding_b.reshape(-1)
        denominator = float((embedding_a.norm() * embedding_b.norm()).item())
        if denominator <= 0.0:
            return 0.0
        similarity = float(((embedding_a * embedding_b).sum() / denominator).item())
        return max(0.0, min(1.0, similarity))
    except Exception as exc:  # noqa: BLE001
        if not _similarity_warning_emitted:
            logger.warning("Similarity computation failed; using lexical fallback: %s", exc)
            _similarity_warning_emitted = True
        return lexical_similarity_score(text_a, text_b)
