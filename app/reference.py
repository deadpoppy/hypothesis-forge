from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from .config import config
from .llm import LLMCallError, call_json
from .prompts import build_reference_summary_messages
from .utils import coerce_string_list, logger


SKILL_DIR = Path("/Users/milo/.codex/skills/arxiv2md-summarize")
SUMMARIZE_SCRIPT = SKILL_DIR / "scripts" / "summarize_paper.py"


class ReferencePaperError(RuntimeError):
    """Raised when the required reference paper cannot be prepared."""


def normalize_arxiv_reference(reference: str) -> tuple[str, str]:
    raw = str(reference or "").strip()
    if not raw:
        raise ReferencePaperError("Reference arXiv link is required.")

    match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", raw)
    if not match:
        raise ReferencePaperError(
            "Reference paper must be an arXiv URL or ID, for example "
            "https://arxiv.org/abs/2502.18864."
        )

    arxiv_id = "".join(part for part in match.groups() if part)
    return arxiv_id, f"https://arxiv.org/abs/{arxiv_id}"


def prepare_reference_paper(reference: str, model: str, temperature: float = 0.1) -> Dict[str, Any]:
    """Fetch, convert, and summarize the required reference paper."""

    arxiv_id, arxiv_url = normalize_arxiv_reference(reference)
    markdown = _convert_arxiv_to_markdown(reference, arxiv_id)
    summary = _summarize_reference_markdown(arxiv_id, arxiv_url, markdown, model, temperature)
    return {
        "arxiv_id": arxiv_id,
        "arxiv_url": arxiv_url,
        "source_input": str(reference).strip(),
        **summary,
    }


def _convert_arxiv_to_markdown(reference: str, arxiv_id: str) -> str:
    if not SUMMARIZE_SCRIPT.exists():
        raise ReferencePaperError(f"arxiv2md skill script not found: {SUMMARIZE_SCRIPT}")

    timeout = int(config.get("reference_arxiv_timeout_seconds", 180) or 180)
    with tempfile.TemporaryDirectory(prefix="co_scientist_reference_") as temp_dir:
        command = [
            sys.executable,
            str(SUMMARIZE_SCRIPT),
            str(reference).strip(),
            "--output",
            temp_dir,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(SKILL_DIR),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ReferencePaperError(f"Timed out converting reference paper {arxiv_id}.") from exc
        except OSError as exc:
            raise ReferencePaperError(f"Could not run arxiv2md skill script: {exc}") from exc

        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            raise ReferencePaperError(
                f"Failed to convert reference paper {arxiv_id} with arxiv2md skill script: {stderr[:1200]}"
            )

        markdown_paths = sorted(Path(temp_dir).glob("*.md"))
        if not markdown_paths:
            raise ReferencePaperError(f"arxiv2md skill script did not produce Markdown for {arxiv_id}.")

        markdown = markdown_paths[0].read_text(encoding="utf-8")
        if not markdown.strip():
            raise ReferencePaperError(f"Reference paper Markdown for {arxiv_id} is empty.")
        return markdown


def _summarize_reference_markdown(
    arxiv_id: str,
    arxiv_url: str,
    markdown: str,
    model: str,
    temperature: float,
) -> Dict[str, Any]:
    char_limit = int(config.get("reference_markdown_char_limit", 24000) or 24000)
    markdown_excerpt = markdown[:char_limit]
    try:
        payload = call_json(
            build_reference_summary_messages(
                arxiv_id=arxiv_id,
                arxiv_url=arxiv_url,
                markdown_excerpt=markdown_excerpt,
            ),
            model=model,
            temperature=temperature,
            profile="thinking",
        )
    except LLMCallError as exc:
        raise ReferencePaperError(f"Failed to summarize reference paper {arxiv_id}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ReferencePaperError(f"Reference paper summary for {arxiv_id} was not a JSON object.")

    return {
        "title": str(payload.get("title") or "").strip(),
        "concise_summary": str(payload.get("concise_summary") or "").strip(),
        "core_problem": str(payload.get("core_problem") or "").strip(),
        "core_mechanism": str(payload.get("core_mechanism") or "").strip(),
        "key_results": coerce_string_list(payload.get("key_results", [])),
        "limitations": coerce_string_list(payload.get("limitations", [])),
        "reusable_insights_for_new_ideas": coerce_string_list(payload.get("reusable_insights_for_new_ideas", [])),
        "avoid_copying": coerce_string_list(payload.get("avoid_copying", [])),
        "seed_queries": coerce_string_list(payload.get("seed_queries", [])),
        "important_figures": _coerce_figure_list(payload.get("important_figures", [])),
        "raw_summary": payload,
    }


def _coerce_figure_list(value: Any) -> list[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    figures = []
    for item in value:
        if isinstance(item, dict):
            figures.append(
                {
                    "label": str(item.get("label") or "").strip(),
                    "takeaway": str(item.get("takeaway") or "").strip(),
                    "url": str(item.get("url") or "").strip(),
                }
            )
        elif item:
            figures.append({"label": "", "takeaway": str(item).strip(), "url": ""})
    return figures


def compact_reference_context(reference_paper: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(reference_paper, dict):
        return {}
    return {
        "arxiv_id": reference_paper.get("arxiv_id"),
        "arxiv_url": reference_paper.get("arxiv_url"),
        "title": reference_paper.get("title"),
        "concise_summary": reference_paper.get("concise_summary"),
        "core_problem": reference_paper.get("core_problem"),
        "core_mechanism": reference_paper.get("core_mechanism"),
        "key_results": reference_paper.get("key_results", [])[:5],
        "limitations": reference_paper.get("limitations", [])[:5],
        "reusable_insights_for_new_ideas": reference_paper.get("reusable_insights_for_new_ideas", [])[:8],
        "avoid_copying": reference_paper.get("avoid_copying", [])[:6],
        "seed_queries": reference_paper.get("seed_queries", [])[:8],
    }


def reference_context_text(reference_paper: Dict[str, Any]) -> str:
    compact = compact_reference_context(reference_paper)
    if not compact:
        return "No reference paper context prepared."
    return json.dumps(compact, ensure_ascii=False, indent=2)


def log_reference_summary(reference_paper: Dict[str, Any]) -> None:
    compact = compact_reference_context(reference_paper)
    if compact:
        logger.info(
            "Prepared reference paper context: %s (%s)",
            compact.get("title") or compact.get("arxiv_id"),
            compact.get("arxiv_url"),
        )
