from __future__ import annotations

import json
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from dateutil import parser as date_parser

from .config import config
from .tools.arxiv_search import ArxivSearchTool
from .tools.semantic_scholar import SemanticScholarSearchTool
from .utils import dedupe_preserve_order, get_max_concurrency, logger, run_concurrently


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().casefold())


def _normalize_title(title: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", title.casefold())
    return re.sub(r"\s+", " ", normalized).strip()


def _paper_key(note: Dict[str, Any]) -> str:
    doi = str(note.get("doi") or "").strip().casefold()
    if doi:
        return f"doi:{doi}"
    arxiv_id = str(note.get("arxiv_id") or "").strip().casefold()
    if arxiv_id:
        normalized_arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        return f"arxiv:{normalized_arxiv_id}"
    semantic_scholar_id = str(note.get("semantic_scholar_id") or "").strip()
    if semantic_scholar_id:
        return f"s2:{semantic_scholar_id}"
    return f"title:{_normalize_title(str(note.get('title') or ''))}"


class LiteratureCache:
    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._data: Dict[str, Any] | None = None

    def get(self, source: str, query: str) -> List[Dict[str, Any]]:
        key = self._entry_key(source, query)
        with self._lock:
            data = self._load()
            entry = data.get("entries", {}).get(key, {})
            papers = entry.get("papers", [])
            return papers if isinstance(papers, list) else []

    def set(self, source: str, query: str, papers: List[Dict[str, Any]]) -> None:
        key = self._entry_key(source, query)
        with self._lock:
            data = self._load()
            data.setdefault("entries", {})[key] = {
                "source": source,
                "query": query,
                "normalized_query": _normalize_query(query),
                "saved_at": time.time(),
                "papers": papers,
            }
            self._save(data)

    def search_cached_notes(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        query_tokens = set(_normalize_query(query).split())
        if not query_tokens:
            return []

        matches = []
        with self._lock:
            entries = self._load().get("entries", {})
            for entry in entries.values():
                entry_query = str(entry.get("normalized_query") or "")
                entry_tokens = set(entry_query.split())
                if entry_query == _normalize_query(query) or len(query_tokens & entry_tokens) >= min(3, len(query_tokens)):
                    for paper in entry.get("papers", []):
                        if isinstance(paper, dict):
                            matches.append(paper)
        return dedupe_notes(matches)[:max_results]

    def all_notes(self) -> List[Dict[str, Any]]:
        matches = []
        with self._lock:
            entries = self._load().get("entries", {})
            for entry in entries.values():
                for paper in entry.get("papers", []):
                    if isinstance(paper, dict):
                        matches.append(paper)
        return dedupe_notes(matches)

    def _entry_key(self, source: str, query: str) -> str:
        return f"{source}::{_normalize_query(query)}"

    def _load(self) -> Dict[str, Any]:
        if self._data is not None:
            return self._data
        if not self.path.exists():
            self._data = {"version": 1, "entries": {}}
            return self._data
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._data = {"version": 1, "entries": {}}
        if not isinstance(self._data, dict):
            self._data = {"version": 1, "entries": {}}
        self._data.setdefault("entries", {})
        return self._data

    def _save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)


def dedupe_notes(notes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for raw_note in notes:
        if not isinstance(raw_note, dict):
            continue
        note = dict(raw_note)
        key = _paper_key(note)
        if key in {"title:", "doi:", "arxiv:", "s2:"}:
            continue
        note_sources = note.get("sources", [])
        if not isinstance(note_sources, list):
            note_sources = []
        if key not in merged:
            note["sources"] = sorted(set(note_sources + [note.get("source", "unknown")]))
            merged[key] = note
            continue

        existing = merged[key]
        existing_sources = existing.get("sources", [])
        if not isinstance(existing_sources, list):
            existing_sources = []
        existing["sources"] = sorted(set(existing_sources + note_sources + [note.get("source", "unknown")]))
        for field in (
            "abstract",
            "summary",
            "citation",
            "arxiv_url",
            "pdf_url",
            "doi",
            "arxiv_id",
            "semantic_scholar_id",
            "url",
            "venue",
            "published",
            "updated",
            "year",
            "citation_count",
        ):
            if not existing.get(field) and note.get(field):
                existing[field] = note[field]
        existing_authors = existing.get("authors", [])
        note_authors = note.get("authors", [])
        if isinstance(existing_authors, list) and isinstance(note_authors, list):
            existing["authors"] = list(dict.fromkeys(existing_authors + note_authors))

    return list(merged.values())


def _citation_count(note: Dict[str, Any]) -> int | None:
    value = note.get("citation_count")
    if value is None:
        value = note.get("citationCount")
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _published_timestamp(note: Dict[str, Any]) -> float | None:
    raw_value = note.get("published") or note.get("updated")
    if not raw_value and note.get("year"):
        raw_value = f"{note['year']}-01-01"
    if not raw_value:
        return None
    try:
        return date_parser.parse(str(raw_value)).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _selection_budgets(max_results: int) -> Tuple[int, int, int]:
    if max_results >= 5:
        cited = 2
        recent = 2
    elif max_results >= 3:
        cited = 1
        recent = 1
    elif max_results == 2:
        cited = 1
        recent = 1
    else:
        cited = max_results
        recent = 0
    exploratory = max(0, max_results - cited - recent)
    return cited, recent, exploratory


def _annotate_selection(note: Dict[str, Any], reason: str) -> Dict[str, Any]:
    annotated = dict(note)
    annotated["selection_reason"] = reason
    return annotated


def _select_literature_notes(
    notes: Sequence[Dict[str, Any]],
    max_results: int,
    sample_seed: str | None = None,
) -> List[Dict[str, Any]]:
    deduped = dedupe_notes(notes)
    if max_results <= 0:
        return []
    if len(deduped) <= max_results:
        return deduped

    cited_budget, recent_budget, exploratory_budget = _selection_budgets(max_results)
    high_relevance_pool = deduped[: max(max_results * 4, 12)]
    selected: List[Dict[str, Any]] = []
    selected_keys = set()

    def add(note: Dict[str, Any], reason: str) -> bool:
        key = _paper_key(note)
        if key in selected_keys:
            return False
        selected.append(_annotate_selection(note, reason))
        selected_keys.add(key)
        return True

    citation_ranked = sorted(
        enumerate(high_relevance_pool),
        key=lambda item: (
            _citation_count(item[1]) is not None,
            _citation_count(item[1]) or -1,
            _published_timestamp(item[1]) or -1.0,
            -item[0],
        ),
        reverse=True,
    )
    for _index, note in citation_ranked:
        if len([item for item in selected if item.get("selection_reason") == "high_relevance_high_citation"]) >= cited_budget:
            break
        add(note, "high_relevance_high_citation")

    recent_ranked = sorted(
        enumerate(high_relevance_pool),
        key=lambda item: (
            _published_timestamp(item[1]) is not None,
            _published_timestamp(item[1]) or -1.0,
            _citation_count(item[1]) or -1,
            -item[0],
        ),
        reverse=True,
    )
    for _index, note in recent_ranked:
        if len([item for item in selected if item.get("selection_reason") == "high_relevance_recent"]) >= recent_budget:
            break
        add(note, "high_relevance_recent")

    remaining_pool = [note for note in high_relevance_pool if _paper_key(note) not in selected_keys]
    if len(remaining_pool) < exploratory_budget:
        remaining_pool.extend(note for note in deduped if _paper_key(note) not in selected_keys and note not in remaining_pool)
    seed = sample_seed or "|".join(str(note.get("title") or note.get("arxiv_id") or "") for note in deduped[:8])
    rng = random.Random(f"{seed}::exploratory_literature")
    rng.shuffle(remaining_pool)
    for note in remaining_pool:
        if len([item for item in selected if item.get("selection_reason") == "exploratory_random"]) >= exploratory_budget:
            break
        add(note, "exploratory_random")

    for note in deduped:
        if len(selected) >= max_results:
            break
        add(note, "relevance_fill")

    return selected[:max_results]


class LiteratureSearchService:
    def __init__(self) -> None:
        self.cache = LiteratureCache(config.get("literature_cache_path", ".cache/literature_search_cache.json"))
        self.arxiv_tool = ArxivSearchTool()
        self.semantic_scholar_tool = SemanticScholarSearchTool()
        self.sources = set(config.get("literature_sources", ["semantic_scholar", "arxiv"]))
        self._semantic_scholar_skip_logged = False

    def _semantic_scholar_enabled(self) -> bool:
        if "semantic_scholar" not in self.sources:
            return False
        if not getattr(self.semantic_scholar_tool, "api_key", None) and not self._semantic_scholar_skip_logged:
            logger.info("Semantic Scholar source enabled without API key; using unauthenticated rate limits.")
            self._semantic_scholar_skip_logged = True
        return True

    def search(self, queries: Sequence[str], max_results: int, sample_seed: str | None = None) -> List[Dict[str, Any]]:
        if max_results <= 0:
            return []

        candidate_limit = max(max_results * 4, max_results)
        clean_queries = dedupe_preserve_order([str(item).strip() for item in queries if str(item).strip()])
        if not clean_queries:
            return []

        all_notes: List[Dict[str, Any]] = []
        cache_counts: Dict[str, int] = {}
        for query in clean_queries:
            cached_notes = self.cache.search_cached_notes(query, candidate_limit)
            cache_counts[query] = len(cached_notes)
            all_notes.extend(cached_notes)

        live_notes, source_counts, query_source_counts = self._search_live_source_batch(
            clean_queries,
            max_results=candidate_limit,
            use_exact_cache=False,
        )
        all_notes.extend(live_notes)

        if not live_notes and not all_notes:
            logger.warning("No literature results found for %d batched queries from cache, Semantic Scholar, or arXiv.", len(clean_queries))
        else:
            logger.info(
                "Batched literature search returned queries=%d cache=%d live=%s",
                len(clean_queries),
                sum(cache_counts.values()),
                source_counts,
            )
            logger.debug("Batched literature query/source counts: %s", query_source_counts)

        return _select_literature_notes(all_notes, max_results=max_results, sample_seed=sample_seed)

    def prefetch_corpus(
        self,
        queries: Sequence[str],
        results_per_query: int,
        max_total: int | None = None,
    ) -> Dict[str, Any]:
        clean_queries = dedupe_preserve_order([str(item).strip() for item in queries if str(item).strip()])
        if results_per_query <= 0 or not clean_queries:
            return {
                "queries": clean_queries,
                "query_count": len(clean_queries),
                "results_per_query": max(0, results_per_query),
                "fetched_count": 0,
                "local_corpus_size": len(self.local_corpus(max_total=max_total)),
                "source_counts": {},
                "query_source_counts": {},
            }

        cached_notes: List[Dict[str, Any]] = []
        cache_counts: Dict[str, int] = {}
        for query in clean_queries:
            matches = self.cache.search_cached_notes(query, results_per_query)
            cache_counts[query] = len(matches)
            cached_notes.extend(matches)

        live_notes, source_counts, query_source_counts = self._search_live_source_batch(
            clean_queries,
            max_results=results_per_query,
            use_exact_cache=True,
        )
        fetched = dedupe_notes(cached_notes + live_notes)
        local_corpus_size = len(self.local_corpus(max_total=max_total))

        return {
            "queries": clean_queries,
            "query_count": len(clean_queries),
            "results_per_query": results_per_query,
            "fetched_count": len(fetched),
            "local_corpus_size": local_corpus_size,
            "source_counts": source_counts,
            "query_source_counts": query_source_counts,
            "cache_counts": cache_counts,
            "sample_titles": [str(note.get("title") or "") for note in fetched[:8] if note.get("title")],
        }

    def local_corpus(self, max_total: int | None = None) -> List[Dict[str, Any]]:
        if not bool(config.get("prior_art_include_cache_corpus", True)):
            return []
        notes = self.cache.all_notes()
        if max_total is not None:
            return notes[:max(0, max_total)]
        return notes

    def _search_live_source_batch(
        self,
        queries: Sequence[str],
        max_results: int,
        use_exact_cache: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, Dict[str, int]]]:
        tasks = []
        for query in [str(item).strip() for item in queries if str(item).strip()]:
            if self._semantic_scholar_enabled():
                tasks.append(("semantic_scholar", query))
            if "arxiv" in self.sources:
                tasks.append(("arxiv", query))

        def run_source(task: Tuple[str, str]) -> Tuple[str, str, List[Dict[str, Any]]]:
            source, task_query = task
            return task_query, source, self._search_one_source(
                source,
                task_query,
                max_results=max_results,
                use_exact_cache=use_exact_cache,
            )

        results = run_concurrently(tasks, run_source, max_workers=min(get_max_concurrency(), len(tasks) or 1))
        notes: List[Dict[str, Any]] = []
        source_counts: Dict[str, int] = {}
        query_source_counts: Dict[str, Dict[str, int]] = {}
        for task_query, source, source_notes in results:
            source_counts[source] = source_counts.get(source, 0) + len(source_notes)
            query_source_counts.setdefault(task_query, {})[source] = len(source_notes)
            notes.extend(source_notes)
        return dedupe_notes(notes), source_counts, query_source_counts

    def _search_one_source(
        self,
        source: str,
        query: str,
        max_results: int,
        use_exact_cache: bool,
    ) -> List[Dict[str, Any]]:
        cached = self.cache.get(source, query)
        if use_exact_cache and cached:
            return cached

        if source == "semantic_scholar":
            if hasattr(self.semantic_scholar_tool, "search_papers_bulk"):
                papers = self.semantic_scholar_tool.search_papers_bulk(query, max_results=max_results)
            else:
                papers = self.semantic_scholar_tool.search_papers(query, max_results=max_results)
            notes = [self._semantic_scholar_note(paper) for paper in papers]
            if notes:
                self.cache.set(source, query, notes)
                return notes
            return cached

        papers = self.arxiv_tool.search_papers(query=query, max_results=max_results, sort_by="relevance")
        notes = [self._arxiv_note(paper) for paper in papers]
        if notes:
            self.cache.set(source, query, notes)
            return notes
        return cached

    def _semantic_scholar_note(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        title = str(paper.get("title") or "").strip()
        authors = paper.get("authors") or []
        citation_parts = [title]
        if paper.get("year"):
            citation_parts.append(str(paper["year"]))
        if paper.get("venue"):
            citation_parts.append(str(paper["venue"]))

        return {
            "source": "semantic_scholar",
            "semantic_scholar_id": paper.get("semantic_scholar_id"),
            "title": title,
            "summary": str(paper.get("abstract") or "").strip()[:700],
            "abstract": paper.get("abstract") or "",
            "authors": authors,
            "citation": " - ".join(part for part in citation_parts if part),
            "published": paper.get("published"),
            "year": paper.get("year"),
            "venue": paper.get("venue"),
            "doi": paper.get("doi"),
            "arxiv_id": paper.get("arxiv_id"),
            "arxiv_url": paper.get("arxiv_url") or "",
            "url": paper.get("url") or "",
            "pdf_url": paper.get("pdf_url") or "",
            "citation_count": paper.get("citation_count"),
            "sources": ["semantic_scholar"],
        }

    def _arxiv_note(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        title = str(paper.get("title") or "").strip()
        authors = paper.get("authors") or []
        citation = title
        if paper.get("arxiv_id"):
            citation = f"{citation} ({paper['arxiv_id']})"
        if authors:
            citation = f"{citation} - {', '.join(authors[:3])}"

        return {
            "source": "arxiv",
            "semantic_scholar_id": "",
            "title": title,
            "summary": str(paper.get("abstract") or "").strip()[:700],
            "abstract": paper.get("abstract") or "",
            "authors": authors,
            "citation": citation,
            "published": paper.get("published"),
            "updated": paper.get("updated"),
            "year": str(paper.get("published") or "")[:4],
            "venue": paper.get("journal_ref") or "",
            "doi": paper.get("doi"),
            "arxiv_id": paper.get("arxiv_id"),
            "arxiv_url": paper.get("arxiv_url") or "",
            "url": paper.get("arxiv_url") or "",
            "pdf_url": paper.get("pdf_url") or "",
            "sources": ["arxiv"],
        }


_literature_service: LiteratureSearchService | None = None
_service_lock = threading.Lock()


def get_literature_service() -> LiteratureSearchService:
    global _literature_service
    if _literature_service is None:
        with _service_lock:
            if _literature_service is None:
                _literature_service = LiteratureSearchService()
    return _literature_service
