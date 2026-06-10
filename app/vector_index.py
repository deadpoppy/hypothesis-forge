from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence
import re

import numpy as np

from .config import config
from .literature import _paper_key, dedupe_notes
from .utils import get_sentence_transformer_model, logger


RECALL_TEXT_VERSION = "title_abstract_full_v2"

_ABSTRACT_LABEL_RE = re.compile(
    r"(?i)\b(?:problem framing|central insight|theoretical story|mechanism|novelty rationale|"
    r"why not simple combination|why not a simple combination|core hypothesis|hypothesis|proposal|description|summary)\s*:\s*"
)


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def normalize_abstract_like_text(text: Any) -> str:
    raw_text = str(text or "").strip()
    if not raw_text:
        return ""
    normalized = " ".join(line.strip() for line in raw_text.splitlines() if line.strip())
    normalized = _ABSTRACT_LABEL_RE.sub("", normalized)
    return " ".join(normalized.split())


def title_abstract_recall_text(title: Any, abstract: Any) -> str:
    normalized_title = _one_line(title)
    normalized_abstract = normalize_abstract_like_text(abstract)
    parts = []
    if normalized_title:
        parts.append(f"Title: {normalized_title}")
    if normalized_abstract:
        parts.append(f"Abstract: {normalized_abstract}")
    return "\n".join(parts)


def paper_recall_text(note: Dict[str, Any]) -> str:
    abstract = note.get("abstract") or note.get("summary") or ""
    return title_abstract_recall_text(
        note.get("title"),
        abstract,
    )


class PaperVectorIndex:
    """Persistent cosine-similarity index for local prior-art recall."""

    def __init__(
        self,
        index_path: str | Path | None = None,
        model_name: str | None = None,
        backend: str | None = None,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        local_files_only: bool | None = None,
        batch_size: int | None = None,
        search_chunk_size: int | None = None,
    ) -> None:
        self.index_path = Path(index_path or config.get("prior_art_vector_index_path", ".cache/prior_art_vector_index.npz"))
        self.model_name = str(model_name or config.get("sentence_transformer_model", "BAAI/bge-small-en-v1.5"))
        self.local_files_only = (
            bool(local_files_only)
            if local_files_only is not None
            else bool(config.get("sentence_transformer_local_files_only", True))
        )
        self.backend = str(backend or config.get("prior_art_vector_index_backend", "auto")).casefold()
        self.query_prefix = str(query_prefix if query_prefix is not None else config.get("prior_art_query_prefix", ""))
        self.document_prefix = str(
            document_prefix if document_prefix is not None else config.get("prior_art_document_prefix", "")
        )
        self.batch_size = max(1, int(batch_size or config.get("prior_art_vector_batch_size", 64)))
        self.search_chunk_size = max(1, int(search_chunk_size or config.get("prior_art_vector_search_chunk_size", 20000)))

        self._fingerprint: str | None = None
        self._embeddings: np.ndarray | None = None
        self._notes: List[Dict[str, Any]] = []
        self._faiss_index: Any | None = None
        self._faiss_available: bool | None = None
        self._query_embedding_cache: Dict[str, np.ndarray] = {}

    def search(self, query_text: str, notes: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not query_text or top_k <= 0:
            return []
        corpus = dedupe_notes(notes)
        if not corpus:
            return []

        fingerprint = self._fingerprint_for(corpus)
        embeddings, indexed_notes = self._load_or_build(corpus, fingerprint)
        if embeddings.size == 0:
            return []

        query_embedding = self._query_embedding(self.query_prefix + query_text)
        scores = self._search_scores(query_embedding, embeddings)
        limit = min(top_k, len(indexed_notes))
        if limit <= 0:
            return []

        top_indices = np.argsort(-scores, kind="mergesort")[:limit]
        backend = "faiss" if self._faiss_index is not None else "numpy"
        return [
            {
                "note": indexed_notes[int(index)],
                "semantic_score": self._normalize_score(float(scores[int(index)])),
                "vector_rank": rank,
                "vector_index_backend": backend,
            }
            for rank, index in enumerate(top_indices, start=1)
        ]

    def _load_or_build(
        self,
        notes: Sequence[Dict[str, Any]],
        fingerprint: str,
    ) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        if self._fingerprint == fingerprint and self._embeddings is not None:
            return self._embeddings, self._notes

        loaded = self._load_from_disk(fingerprint)
        if loaded is not None:
            embeddings, loaded_notes = loaded
            self._set_active_index(fingerprint, embeddings, loaded_notes)
            return embeddings, loaded_notes

        embeddings = self._build_embeddings(notes)
        stored_notes = [dict(note) for note in notes]
        self._save_to_disk(fingerprint, embeddings, stored_notes)
        self._set_active_index(fingerprint, embeddings, stored_notes)
        return embeddings, stored_notes

    def _build_embeddings(self, notes: Sequence[Dict[str, Any]]) -> np.ndarray:
        texts = [self.document_prefix + paper_recall_text(note) for note in notes]
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        logger.info("Building prior-art vector index for %d papers with %s.", len(texts), self.model_name)
        return self._encode(texts)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        model = get_sentence_transformer_model(self.model_name, local_files_only=self.local_files_only)
        try:
            embeddings = model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except TypeError:
            embeddings = model.encode(list(texts), convert_to_numpy=True)

        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return self._normalize_embeddings(array)

    def _query_embedding(self, text: str) -> np.ndarray:
        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self._query_embedding_cache.get(cache_key)
        if cached is not None:
            return cached
        embedding = self._encode([text])[0]
        self._query_embedding_cache[cache_key] = embedding
        return embedding

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms <= 0.0] = 1.0
        return (embeddings / norms).astype(np.float32, copy=False)

    def _search_scores(self, query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        if self._use_faiss(embeddings):
            scores, indices = self._faiss_index.search(query_embedding.reshape(1, -1).astype(np.float32), len(embeddings))
            ordered_scores = np.full(len(embeddings), -1.0, dtype=np.float32)
            for score, index in zip(scores[0], indices[0]):
                if index >= 0:
                    ordered_scores[int(index)] = float(score)
            return ordered_scores

        scores = np.empty(len(embeddings), dtype=np.float32)
        for start in range(0, len(embeddings), self.search_chunk_size):
            end = min(start + self.search_chunk_size, len(embeddings))
            scores[start:end] = embeddings[start:end] @ query_embedding
        return scores

    def _use_faiss(self, embeddings: np.ndarray) -> bool:
        if self.backend == "numpy":
            self._faiss_index = None
            return False
        if self.backend not in {"auto", "faiss"}:
            self._faiss_index = None
            return False
        if self._faiss_available is False:
            self._faiss_index = None
            return False
        try:
            import faiss  # type: ignore
        except Exception:  # noqa: BLE001
            self._faiss_available = False
            if self.backend == "faiss":
                logger.warning("FAISS backend requested but unavailable; falling back to numpy vector search.")
            self._faiss_index = None
            return False

        self._faiss_available = True
        if self._faiss_index is None or self._faiss_index.ntotal != len(embeddings):
            index = faiss.IndexFlatIP(int(embeddings.shape[1]))
            index.add(embeddings.astype(np.float32, copy=False))
            self._faiss_index = index
        return True

    def _load_from_disk(self, fingerprint: str) -> tuple[np.ndarray, List[Dict[str, Any]]] | None:
        if not self.index_path.exists():
            return None
        try:
            with np.load(self.index_path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
                if metadata.get("fingerprint") != fingerprint:
                    return None
                embeddings = np.asarray(data["embeddings"], dtype=np.float32)
                notes = metadata.get("notes", [])
                if not isinstance(notes, list) or len(notes) != len(embeddings):
                    return None
                return embeddings, [note for note in notes if isinstance(note, dict)]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load prior-art vector index; rebuilding: %s", exc)
            return None

    def _save_to_disk(self, fingerprint: str, embeddings: np.ndarray, notes: Sequence[Dict[str, Any]]) -> None:
        metadata = {
            "version": 1,
            "fingerprint": fingerprint,
            "model_name": self.model_name,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
            "paper_count": len(notes),
            "notes": list(notes),
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.index_path.with_suffix(f"{self.index_path.suffix}.tmp")
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                embeddings=embeddings.astype(np.float32, copy=False),
                metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
            )
        tmp_path.replace(self.index_path)

    def _set_active_index(self, fingerprint: str, embeddings: np.ndarray, notes: Sequence[Dict[str, Any]]) -> None:
        self._fingerprint = fingerprint
        self._embeddings = embeddings
        self._notes = list(notes)
        self._faiss_index = None

    def _fingerprint_for(self, notes: Sequence[Dict[str, Any]]) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.model_name.encode("utf-8"))
        hasher.update(self.query_prefix.encode("utf-8"))
        hasher.update(self.document_prefix.encode("utf-8"))
        hasher.update(RECALL_TEXT_VERSION.encode("utf-8"))
        for note in notes:
            key = _paper_key(note)
            text_hash = hashlib.sha256(paper_recall_text(note).encode("utf-8")).hexdigest()
            hasher.update(f"{key}\0{text_hash}\n".encode("utf-8"))
        return hasher.hexdigest()

    def _normalize_score(self, score: float) -> float:
        return max(0.0, min(1.0, score))


_prior_art_vector_index: PaperVectorIndex | None = None


def get_prior_art_vector_index() -> PaperVectorIndex:
    global _prior_art_vector_index
    if _prior_art_vector_index is None:
        _prior_art_vector_index = PaperVectorIndex()
    return _prior_art_vector_index
