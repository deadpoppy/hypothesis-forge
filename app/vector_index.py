from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence
import re
import threading

import numpy as np

from .config import config
from .literature import _paper_key, dedupe_notes
from .utils import get_sentence_transformer_model, logger


RECALL_TEXT_VERSION = "title_abstract_full_v2"
INDEX_FORMAT_VERSION = 2

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
        self._lock = threading.RLock()

    def search(self, query_text: str, notes: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not query_text or top_k <= 0:
            return []
        corpus = dedupe_notes(notes)
        if not corpus:
            return []

        with self._lock:
            fingerprint = self._fingerprint_for(corpus)
            embeddings, indexed_notes = self._load_or_build(corpus, fingerprint)
            if embeddings.size == 0:
                return []

            query_embedding = self._query_embedding(self.query_prefix + query_text)
            limit = min(top_k, len(indexed_notes))
            if limit <= 0:
                return []

            top_matches = self._search_top_matches(query_embedding, embeddings, limit)
            backend = "faiss" if self._faiss_index is not None else "numpy"
            return [
                {
                    "note": indexed_notes[int(index)],
                    "semantic_score": self._normalize_score(float(score)),
                    "vector_rank": rank,
                    "vector_index_backend": backend,
                }
                for rank, (index, score) in enumerate(top_matches, start=1)
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

        reusable = self._load_reusable_from_disk()
        if reusable is None:
            embeddings = self._build_embeddings(notes)
        else:
            previous_embeddings, previous_notes = reusable
            embeddings = self._build_incremental_embeddings(
                notes,
                previous_embeddings,
                previous_notes,
            )
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

    def _build_incremental_embeddings(
        self,
        notes: Sequence[Dict[str, Any]],
        previous_embeddings: np.ndarray,
        previous_notes: Sequence[Dict[str, Any]],
    ) -> np.ndarray:
        previous_by_key = {}
        for index, note in enumerate(previous_notes):
            if index >= len(previous_embeddings):
                break
            previous_by_key[_paper_key(note)] = (
                self._recall_text_hash(note),
                previous_embeddings[index],
            )

        vectors: List[np.ndarray | None] = []
        pending_positions = []
        pending_texts = []
        reused_count = 0
        current_keys = set()
        for position, note in enumerate(notes):
            key = _paper_key(note)
            current_keys.add(key)
            previous = previous_by_key.get(key)
            if previous is not None and previous[0] == self._recall_text_hash(note):
                vectors.append(np.asarray(previous[1], dtype=np.float32))
                reused_count += 1
                continue
            vectors.append(None)
            pending_positions.append(position)
            pending_texts.append(self.document_prefix + paper_recall_text(note))

        if pending_texts:
            new_embeddings = self._encode(pending_texts)
            for position, embedding in zip(pending_positions, new_embeddings):
                vectors[position] = embedding

        removed_count = len(set(previous_by_key) - current_keys)
        logger.info(
            "Updated prior-art vector index: total=%d reused=%d encoded=%d removed=%d.",
            len(notes),
            reused_count,
            len(pending_texts),
            removed_count,
        )
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        if any(vector is None for vector in vectors):
            raise RuntimeError("Incremental prior-art index update did not encode every paper.")
        return np.vstack(vectors).astype(np.float32, copy=False)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        model = get_sentence_transformer_model(self.model_name, local_files_only=self.local_files_only)
        text_list = list(texts)
        if not text_list:
            return np.empty((0, 0), dtype=np.float32)

        encoded_batches = []
        for start in range(0, len(text_list), self.batch_size):
            batch = text_list[start : start + self.batch_size]
            encoded_batches.append(self._encode_batch(model, batch))

        array = np.vstack(encoded_batches).astype(np.float32, copy=False)
        return self._normalize_embeddings(array)

    def _encode_batch(self, model: Any, texts: Sequence[str]) -> np.ndarray:
        try:
            embeddings = model.encode(
                list(texts),
                batch_size=min(self.batch_size, len(texts)),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except TypeError:
            embeddings = model.encode(list(texts), convert_to_numpy=True)

        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array

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

    def _search_top_matches(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        top_k: int,
    ) -> List[tuple[int, float]]:
        if top_k <= 0:
            return []
        if self._use_faiss(embeddings):
            scores, indices = self._faiss_index.search(query_embedding.reshape(1, -1).astype(np.float32), top_k)
            matches = []
            for score, index in zip(scores[0], indices[0]):
                if index >= 0:
                    matches.append((int(index), float(score)))
            return sorted(matches, key=lambda item: (-item[1], item[0]))[:top_k]

        best_indices = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=np.float32)
        for start in range(0, len(embeddings), self.search_chunk_size):
            end = min(start + self.search_chunk_size, len(embeddings))
            chunk_scores = embeddings[start:end] @ query_embedding
            chunk_limit = min(top_k, len(chunk_scores))
            if chunk_limit <= 0:
                continue
            if chunk_limit < len(chunk_scores):
                chunk_top = np.argpartition(-chunk_scores, chunk_limit - 1)[:chunk_limit]
            else:
                chunk_top = np.arange(len(chunk_scores))
            candidate_indices = np.concatenate([best_indices, chunk_top.astype(np.int64) + start])
            candidate_scores = np.concatenate([best_scores, chunk_scores[chunk_top].astype(np.float32, copy=False)])
            candidate_limit = min(top_k, len(candidate_scores))
            if candidate_limit < len(candidate_scores):
                keep = np.argpartition(-candidate_scores, candidate_limit - 1)[:candidate_limit]
            else:
                keep = np.arange(len(candidate_scores))
            best_indices = candidate_indices[keep]
            best_scores = candidate_scores[keep]

        matches = [(int(index), float(score)) for index, score in zip(best_indices, best_scores)]
        return sorted(matches, key=lambda item: (-item[1], item[0]))[:top_k]

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
        loaded = self._load_index_data()
        if loaded is None:
            return None
        metadata, embeddings, notes = loaded
        if metadata.get("fingerprint") != fingerprint:
            return None
        return embeddings, notes

    def _load_reusable_from_disk(self) -> tuple[np.ndarray, List[Dict[str, Any]]] | None:
        loaded = self._load_index_data()
        if loaded is None:
            return None
        metadata, embeddings, notes = loaded
        if metadata.get("version") != INDEX_FORMAT_VERSION:
            return None
        if metadata.get("recall_text_version") != RECALL_TEXT_VERSION:
            return None
        if metadata.get("model_name") != self.model_name:
            return None
        if metadata.get("query_prefix") != self.query_prefix:
            return None
        if metadata.get("document_prefix") != self.document_prefix:
            return None
        return embeddings, notes

    def _load_index_data(
        self,
    ) -> tuple[Dict[str, Any], np.ndarray, List[Dict[str, Any]]] | None:
        if not self.index_path.exists():
            return None
        try:
            with np.load(self.index_path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
                embeddings = np.asarray(data["embeddings"], dtype=np.float32)
                notes = metadata.get("notes", [])
                if not isinstance(notes, list) or len(notes) != len(embeddings):
                    return None
                normalized_notes = [note for note in notes if isinstance(note, dict)]
                if len(normalized_notes) != len(embeddings):
                    return None
                return metadata, embeddings, normalized_notes
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load prior-art vector index; rebuilding: %s", exc)
            return None

    def _save_to_disk(self, fingerprint: str, embeddings: np.ndarray, notes: Sequence[Dict[str, Any]]) -> None:
        metadata = {
            "version": INDEX_FORMAT_VERSION,
            "fingerprint": fingerprint,
            "recall_text_version": RECALL_TEXT_VERSION,
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
            text_hash = self._recall_text_hash(note)
            hasher.update(f"{key}\0{text_hash}\n".encode("utf-8"))
        return hasher.hexdigest()

    def _recall_text_hash(self, note: Dict[str, Any]) -> str:
        return hashlib.sha256(paper_recall_text(note).encode("utf-8")).hexdigest()

    def _normalize_score(self, score: float) -> float:
        return max(0.0, min(1.0, score))


_prior_art_vector_index: PaperVectorIndex | None = None
_prior_art_vector_index_lock = threading.Lock()


def get_prior_art_vector_index() -> PaperVectorIndex:
    global _prior_art_vector_index
    if _prior_art_vector_index is None:
        with _prior_art_vector_index_lock:
            if _prior_art_vector_index is None:
                _prior_art_vector_index = PaperVectorIndex()
    return _prior_art_vector_index
