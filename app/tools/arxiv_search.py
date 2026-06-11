import arxiv
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dateutil import parser
from pathlib import Path
import re
import time

from app.config import config

logger = logging.getLogger(__name__)


class ArxivSearchTool:
    """Tool for searching and retrieving papers from arXiv"""

    _FIELD_QUERY_RE = re.compile(
        r"(?i)(?:^|[\s(])(?:all|ti|au|abs|co|jr|cat|rn|id|submittedDate):"
    )
    _TOKEN_RE = re.compile(r"[^\W_]+(?:\.[^\W_]+)*", flags=re.UNICODE)
    _LOW_SIGNAL_TOKENS = {
        "a",
        "an",
        "and",
        "for",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "using",
        "via",
        "with",
    }
    
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self.client = arxiv.Client(page_size=max_results, delay_seconds=0.5, num_retries=0)
        self.failure_cooldown_seconds = float(config.get("arxiv_failure_cooldown_seconds", 900.0))
        self.max_consecutive_failures = int(config.get("arxiv_max_consecutive_failures", 3))
        self.rate_limit_max_retries = int(config.get("arxiv_rate_limit_max_retries", 1))
        self.rate_limit_retry_after_seconds = float(config.get("arxiv_rate_limit_retry_after_seconds", 20.0))
        self.state_path = Path(config.get("arxiv_state_path", ".cache/arxiv_search_state.json"))
        self._consecutive_failures = 0
        self._disabled_until = 0.0
        self._load_state()

    def _load_state(self) -> None:
        try:
            if not self.state_path.exists():
                return
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._consecutive_failures = int(payload.get("consecutive_failures", 0) or 0)
            self._disabled_until = float(payload.get("disabled_until", 0.0) or 0.0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self._consecutive_failures = 0
            self._disabled_until = 0.0

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(
                    {
                        "consecutive_failures": self._consecutive_failures,
                        "disabled_until": self._disabled_until,
                        "saved_at": time.time(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            logger.debug("Failed to persist arXiv cooldown state at %s", self.state_path, exc_info=True)

    def _is_rate_limited_error(self, error: Exception) -> bool:
        message = str(error)
        return "HTTP 429" in message or "status_code=429" in message or "too many requests" in message.casefold()
        
    def search_papers(self, query: str, max_results: Optional[int] = None, 
                     categories: Optional[List[str]] = None,
                     sort_by: str = "relevance") -> List[Dict]:
        """
        Search arXiv for papers matching query
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            categories: List of arXiv categories to filter by (e.g., ['cs.AI', 'cs.LG'])
            sort_by: Sort criteria ('relevance', 'lastUpdatedDate', 'submittedDate')
        
        Returns:
            List of paper dictionaries with metadata
        """
        if max_results is None:
            max_results = self.max_results
        if time.time() < self._disabled_until:
            logger.info(
                "ArXiv live search temporarily disabled until %.0f for query '%s'; returning no live results.",
                self._disabled_until,
                query,
            )
            return []
            
        # Translate source-neutral keyword bundles into arXiv field syntax.
        search_query = self._arxiv_query(query)
        if categories:
            category_filter = " OR ".join([f"cat:{cat}" for cat in categories])
            search_query = f"({search_query}) AND ({category_filter})"
            
        # Set sort criteria
        sort_criterion = arxiv.SortCriterion.Relevance
        if sort_by == "lastUpdatedDate":
            sort_criterion = arxiv.SortCriterion.LastUpdatedDate
        elif sort_by == "submittedDate":
            sort_criterion = arxiv.SortCriterion.SubmittedDate

        client = self.client
        client_page_size = getattr(client, "page_size", None)
        if isinstance(client_page_size, int) and max_results > client_page_size:
            client = arxiv.Client(page_size=min(max_results, 100), delay_seconds=0.5, num_retries=0)
            
        # Log search parameters
        logger.info(f"ArXiv search initiated - Query: '{query}', Max Results: {max_results}, "
                   f"Categories: {categories}, Sort: {sort_by}")
        if search_query != query:
            logger.debug(f"Expanded search query: '{search_query}'")
            
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending
        )

        for attempt in range(self.rate_limit_max_retries + 1):
            try:
                start_time = time.time()

                papers = []
                for paper in client.results(search):
                    papers.append(self._format_paper(paper))

                search_time = (time.time() - start_time) * 1000  # Convert to ms

                # Enhanced logging with performance metrics
                logger.info(f"ArXiv search completed - Found {len(papers)} papers for query: '{query}' "
                           f"in {search_time:.2f}ms")

                # Log paper details at debug level
                if papers and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"ArXiv papers found:")
                    for i, paper in enumerate(papers[:3], 1):  # Log first 3 papers
                        logger.debug(f"  {i}. {paper['title']} ({paper['arxiv_id']}) - "
                                   f"Published: {paper['published']}")
                    if len(papers) > 3:
                        logger.debug(f"  ... and {len(papers) - 3} more papers")

                # Log categories distribution
                if papers:
                    categories_count = {}
                    for paper in papers:
                        for cat in paper.get('categories', []):
                            categories_count[cat] = categories_count.get(cat, 0) + 1
                    top_categories = sorted(categories_count.items(), key=lambda x: x[1], reverse=True)[:5]
                    logger.info(f"ArXiv search result categories: {dict(top_categories)}")

                self._consecutive_failures = 0
                self._disabled_until = 0.0
                self._save_state()

                return papers

            except Exception as e:
                rate_limited = self._is_rate_limited_error(e)
                if rate_limited and attempt < self.rate_limit_max_retries:
                    logger.warning(
                        "ArXiv rate-limited query '%s'; retrying in %.1fs (%d/%d)",
                        query,
                        self.rate_limit_retry_after_seconds,
                        attempt + 1,
                        self.rate_limit_max_retries,
                    )
                    time.sleep(self.rate_limit_retry_after_seconds)
                    continue

                self._consecutive_failures += 1
                if rate_limited or self._consecutive_failures >= self.max_consecutive_failures:
                    self._disabled_until = time.time() + self.failure_cooldown_seconds
                self._save_state()
                logger.warning(
                    "ArXiv live search skipped for query '%s' after %d consecutive failures%s.",
                    query,
                    self._consecutive_failures,
                    " due to rate limiting" if rate_limited else "",
                )
                if self._disabled_until:
                    logger.warning("ArXiv live search disabled for %.0fs.", self.failure_cooldown_seconds)
                if rate_limited:
                    logger.warning("ArXiv search skipped for query '%s' after rate limit retry failed: %s", query, e)
                else:
                    logger.error(f"ArXiv search failed for query '{query}': {e}", exc_info=True)
                return []

        return []
    
    def search_by_author(self, author_name: str, max_results: Optional[int] = None) -> List[Dict]:
        """Search for papers by a specific author"""
        query = f"au:{author_name}"
        return self.search_papers(query, max_results)
    
    def search_recent_papers(self, query: str, days_back: int = 7, 
                           max_results: Optional[int] = None) -> List[Dict]:
        """Search for recent papers within specified time frame"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for arXiv search
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        
        # Add date filter to query
        date_query = f"({self._arxiv_query(query)}) AND submittedDate:[{start_str} TO {end_str}]"
        return self.search_papers(date_query, max_results, sort_by="submittedDate")
    
    def search_by_category(self, category: str, max_results: Optional[int] = None,
                          days_back: Optional[int] = None) -> List[Dict]:
        """Search papers in a specific arXiv category"""
        query = f"cat:{category}"
        
        if days_back:
            return self.search_recent_papers(query, days_back, max_results)
        else:
            return self.search_papers(query, max_results)
    
    def get_paper_details(self, arxiv_id: str) -> Optional[Dict]:
        """Get detailed information for a specific paper by arXiv ID"""
        logger.info(f"Fetching arXiv paper details for ID: {arxiv_id}")
        try:
            import time
            start_time = time.time()
            
            search = arxiv.Search(id_list=[arxiv_id])
            papers = list(self.client.results(search))
            
            fetch_time = (time.time() - start_time) * 1000
            
            if papers:
                paper = self._format_paper(papers[0])
                logger.info(f"Successfully retrieved paper '{paper['title']}' ({arxiv_id}) in {fetch_time:.2f}ms")
                return paper
            else:
                logger.warning(f"No paper found with arXiv ID: {arxiv_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving paper {arxiv_id}: {e}", exc_info=True)
            return None
    
    def _format_paper(self, paper: arxiv.Result) -> Dict:
        """Format arXiv paper result into a standardized dictionary"""
        # Extract arXiv ID from entry_id URL
        arxiv_id = paper.get_short_id()
        
        # Clean and format abstract
        abstract = self._clean_text(paper.summary)
        
        # Format authors
        authors = [str(author) for author in paper.authors]
        
        # Extract DOI if available
        doi = None
        if paper.doi:
            doi = paper.doi
            
        # Format categories
        categories = paper.categories if paper.categories else []
        
        return {
            'arxiv_id': arxiv_id,
            'entry_id': paper.entry_id,
            'title': self._clean_text(paper.title),
            'abstract': abstract,
            'authors': authors,
            'primary_category': paper.primary_category,
            'categories': categories,
            'published': paper.published.isoformat() if paper.published else None,
            'updated': paper.updated.isoformat() if paper.updated else None,
            'doi': doi,
            'pdf_url': paper.pdf_url,
            'arxiv_url': f"https://arxiv.org/abs/{arxiv_id}",
            'comment': paper.comment,
            'journal_ref': paper.journal_ref,
            'source': 'arxiv'
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and newlines"""
        if not text:
            return ""
        # Replace multiple whitespace with single space
        cleaned = re.sub(r'\s+', ' ', text)
        return cleaned.strip()

    def _arxiv_query(self, query: str) -> str:
        raw_query = " ".join(str(query or "").split())
        if not raw_query or self._FIELD_QUERY_RE.search(raw_query):
            return raw_query

        try:
            max_terms = int(config.get("arxiv_max_terms_per_clause", 8))
        except (TypeError, ValueError):
            max_terms = 8
        max_terms = max(2, min(12, max_terms))

        clauses = re.split(r"\s+\bOR\b\s+", raw_query, flags=re.IGNORECASE)
        translated = []
        for clause in clauses:
            value = clause.strip().strip("() ").strip('"')
            tokens = self._TOKEN_RE.findall(value.replace("-", " ").replace("/", " "))
            filtered = [token for token in tokens if token.casefold() not in self._LOW_SIGNAL_TOKENS]
            if filtered:
                tokens = filtered
            tokens = list(dict.fromkeys(tokens))[:max_terms]
            if not tokens:
                continue
            translated.append("(" + " AND ".join(f"all:{token}" for token in tokens) + ")")

        return " OR ".join(translated) or raw_query
    
    def analyze_research_trends(self, query: str, days_back: int = 30) -> Dict:
        """Analyze research trends for a given topic"""
        logger.info(f"Starting arXiv trends analysis for '{query}' over last {days_back} days")
        
        papers = self.search_recent_papers(query, days_back, max_results=50)
        
        if not papers:
            logger.warning(f"No papers found for trends analysis of '{query}' in last {days_back} days")
            return {
                'total_papers': 0,
                'categories': {},
                'top_authors': {},
                'papers': []
            }
        
        # Analyze categories
        category_counts = {}
        author_counts = {}
        
        for paper in papers:
            # Count categories
            for category in paper.get('categories', []):
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Count authors
            for author in paper.get('authors', []):
                author_counts[author] = author_counts.get(author, 0) + 1
        
        # Sort by frequency
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Log trends analysis results
        logger.info(f"ArXiv trends analysis completed for '{query}': {len(papers)} papers, "
                   f"top categories: {dict(top_categories[:3])}")
        if top_authors:
            logger.info(f"Most active authors: {dict(top_authors[:3])}")
        
        return {
            'total_papers': len(papers),
            'date_range': f"Last {days_back} days",
            'top_categories': top_categories,
            'top_authors': top_authors,
            'papers': papers
        }

# Common arXiv categories for different fields
ARXIV_CATEGORIES = {
    'computer_science': [
        'cs.AI',  # Artificial Intelligence
        'cs.LG',  # Machine Learning
        'cs.CL',  # Computation and Language
        'cs.CV',  # Computer Vision
        'cs.RO',  # Robotics
        'cs.NE',  # Neural and Evolutionary Computing
    ],
    'physics': [
        'physics.data-an',  # Data Analysis
        'physics.comp-ph',  # Computational Physics
        'cond-mat.stat-mech',  # Statistical Mechanics
    ],
    'mathematics': [
        'math.ST',  # Statistics Theory
        'math.OC',  # Optimization and Control
        'math.PR',  # Probability
    ],
    'quantitative_biology': [
        'q-bio.QM',  # Quantitative Methods
        'q-bio.GN',  # Genomics
        'q-bio.BM',  # Biomolecules
    ]
}

def get_categories_for_field(field: str) -> List[str]:
    """Get relevant arXiv categories for a research field"""
    return ARXIV_CATEGORIES.get(field.lower(), [])
