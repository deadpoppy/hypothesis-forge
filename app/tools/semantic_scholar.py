from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from app.config import config
from app.utils import logger


class SemanticScholarSearchTool:
    """Small wrapper around the Semantic Scholar Graph API paper search endpoint."""

    SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    BULK_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    DEFAULT_FIELDS = ",".join(
        [
            "paperId",
            "title",
            "abstract",
            "authors",
            "year",
            "venue",
            "publicationDate",
            "citationCount",
            "url",
            "externalIds",
            "isOpenAccess",
            "openAccessPdf",
            "citationCount",
        ]
    )

    def __init__(self, timeout_seconds: Optional[float] = None):
        self.timeout_seconds = float(timeout_seconds or config.get("semantic_scholar_timeout_seconds", 8))
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or config.get("semantic_scholar_api_key")
        self.min_delay_seconds = float(config.get("semantic_scholar_min_delay_seconds", 2.0))
        self.max_retries = int(config.get("semantic_scholar_max_retries", 2))
        self.retry_after_seconds = float(config.get("semantic_scholar_retry_after_seconds", 8.0))
        self.cooldown_seconds = float(config.get("semantic_scholar_cooldown_seconds", 600.0))
        self._lock = threading.Lock()
        self._last_call_at = 0.0
        self._disabled_until = 0.0

    def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        if max_results <= 0:
            return []
        if time.time() < self._disabled_until:
            return []

        params = urllib.parse.urlencode(
            {
                "query": self._semantic_scholar_query(query),
                "limit": max_results,
                "fields": self.DEFAULT_FIELDS,
            }
        )
        request = urllib.request.Request(f"{self.SEARCH_URL}?{params}", headers=self._headers())

        payload = self._request_json(request, query)
        if not payload:
            return []

        papers = payload.get("data", []) if isinstance(payload, dict) else []
        return [self._format_paper(paper) for paper in papers if isinstance(paper, dict)]

    def search_papers_bulk(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        if max_results <= 0:
            return []
        if time.time() < self._disabled_until:
            return []

        papers: List[Dict[str, Any]] = []
        token = None
        while len(papers) < max_results:
            params: Dict[str, Any] = {
                "query": self._semantic_scholar_query(query),
                "fields": self.DEFAULT_FIELDS,
            }
            if token:
                params["token"] = token
            request = urllib.request.Request(
                f"{self.BULK_SEARCH_URL}?{urllib.parse.urlencode(params)}",
                headers=self._headers(),
            )
            payload = self._request_json(request, query)
            if not payload:
                break
            batch = payload.get("data", []) if isinstance(payload, dict) else []
            papers.extend(self._format_paper(paper) for paper in batch if isinstance(paper, dict))
            token = payload.get("token") if isinstance(payload, dict) else None
            if not token or not batch:
                break

        return papers[:max_results]

    def _request_json(self, request: urllib.request.Request, query: str) -> Dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            self._respect_rate_limit()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    self._disabled_until = 0.0
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < self.max_retries:
                    retry_after = self._retry_after(exc)
                    logger.warning(
                        "Semantic Scholar rate-limited query '%s'; retrying in %.1fs (%d/%d)",
                        query,
                        retry_after,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(retry_after)
                    continue
                if exc.code == 429:
                    self._disabled_until = time.time() + self.cooldown_seconds
                    logger.warning(
                        "Semantic Scholar search disabled for %.0fs after repeated rate limits on query '%s'",
                        self.cooldown_seconds,
                        query,
                    )
                logger.warning("Semantic Scholar search failed for query '%s' with HTTP %s", query, exc.code)
                return {}
            except urllib.error.URLError as exc:
                logger.warning("Semantic Scholar search failed for query '%s': %s", query, exc)
                return {}
            except TimeoutError:
                logger.warning("Semantic Scholar search timed out for query '%s'", query)
                return {}
            except json.JSONDecodeError:
                logger.warning("Semantic Scholar returned invalid JSON for query '%s'", query)
                return {}
        return {}

    def _respect_rate_limit(self) -> None:
        with self._lock:
            elapsed = time.time() - self._last_call_at
            if elapsed < self.min_delay_seconds:
                time.sleep(self.min_delay_seconds - elapsed)
            self._last_call_at = time.time()

    def _retry_after(self, exc: urllib.error.HTTPError) -> float:
        raw_value = exc.headers.get("Retry-After") if exc.headers else None
        try:
            return max(float(raw_value), self.retry_after_seconds) if raw_value else self.retry_after_seconds
        except (TypeError, ValueError):
            return self.retry_after_seconds

    def _headers(self) -> Dict[str, str]:
        headers = {"User-Agent": "hypothesis-forge/0.1"}
        if self.api_key:
            headers["x-api-key"] = str(self.api_key)
        return headers

    def _semantic_scholar_query(self, query: str) -> str:
        # Semantic Scholar treats space-separated terms as a narrow query; use
        # its OR separator when upstream planning emits academic OR queries.
        return " | ".join(part.strip() for part in str(query).split(" OR ") if part.strip()) or str(query)

    def _format_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        external_ids = paper.get("externalIds") or {}
        authors = []
        for author in paper.get("authors") or []:
            if isinstance(author, dict) and author.get("name"):
                authors.append(str(author["name"]))

        open_access_pdf = paper.get("openAccessPdf") or {}
        pdf_url = open_access_pdf.get("url") if isinstance(open_access_pdf, dict) else None

        return {
            "semantic_scholar_id": paper.get("paperId"),
            "title": paper.get("title") or "",
            "abstract": paper.get("abstract") or "",
            "authors": authors,
            "year": paper.get("year"),
            "venue": paper.get("venue"),
            "published": paper.get("publicationDate"),
            "citation_count": paper.get("citationCount"),
            "doi": external_ids.get("DOI"),
            "arxiv_id": external_ids.get("ArXiv") or external_ids.get("ARXIV"),
            "external_ids": external_ids,
            "url": paper.get("url"),
            "arxiv_url": f"https://arxiv.org/abs/{external_ids.get('ArXiv')}" if external_ids.get("ArXiv") else "",
            "pdf_url": pdf_url or "",
            "citation_count": paper.get("citationCount"),
            "source": "semantic_scholar",
        }
