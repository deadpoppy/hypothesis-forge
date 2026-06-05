from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from app.config import config
from app.llm import get_configured_api_key
from app.models import ContextMemory, ResearchGoal
from app.reports import build_markdown_report
from app.utils import logger
from app.workflow import SupervisorAgent


class WorkflowRunError(RuntimeError):
    def __init__(self, message: str, written_outputs: Dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.written_outputs = written_outputs or {}


class InputValidationError(RuntimeError):
    """Raised when CLI inputs or required runtime prerequisites are invalid."""


def _load_constraints(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    constraints_path = Path(path)
    if not constraints_path.exists():
        raise InputValidationError(
            f"Constraints file not found: {constraints_path}. "
            "Pass a valid JSON file path with --constraints."
        )
    if constraints_path.is_dir():
        raise InputValidationError(
            f"Constraints path is a directory, not a file: {constraints_path}. "
            "Pass a JSON file path with --constraints."
        )
    try:
        with constraints_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            f"Constraints file is not valid JSON: {constraints_path} "
            f"(line {exc.lineno}, column {exc.colno})."
        ) from exc
    if not isinstance(payload, dict):
        raise InputValidationError(
            f"Constraints file must contain a top-level JSON object: {constraints_path}."
        )
    return payload


def _require_llm_api_key() -> None:
    if get_configured_api_key("thinking") and get_configured_api_key("critic"):
        return
    raise InputValidationError(
        "LLM API key is not configured. Set `THINKING_LLM_API_KEY` and "
        "`CRITIC_LLM_API_KEY`, set shared `LLM_API_KEY`, or provide "
        "`thinking_llm.api_key` and `critic_llm.api_key` in config.yaml."
    )


def _config_int(name: str, default: int, minimum: int | None = None) -> int:
    try:
        value = int(config.get(name, default))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _write_outputs(cycles: List[Dict[str, Any]], output_dir: str, context: ContextMemory) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"co_scientist_run_{timestamp}"

    json_path = output_path / f"{base_name}.json"
    md_path = output_path / f"{base_name}.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": {
                    "generated_at": timestamp,
                    "cycle_count": len(cycles),
                },
                "cycles": cycles,
                "final_context": context.to_dict(),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(build_markdown_report(cycles))

    return {"json": str(json_path), "markdown": str(md_path)}


def _write_checkpoint(cycles: List[Dict[str, Any]], output_dir: str, context: ContextMemory) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "checkpoint_latest.json"
    md_path = output_path / "checkpoint_latest.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": {
                    "generated_at": dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    "cycle_count": len(cycles),
                    "checkpoint": True,
                },
                "cycles": cycles,
                "final_context": context.to_dict(),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(build_markdown_report(cycles))

    return {"json": str(json_path), "markdown": str(md_path)}


def run_cycles(args: argparse.Namespace) -> Dict[str, str]:
    constraints = _load_constraints(args.constraints)
    _require_llm_api_key()
    research_goal = ResearchGoal(
        description=args.goal,
        constraints=constraints,
        llm_model=args.model,
        critic_llm_model=args.critic_model,
        num_hypotheses=args.num_hypotheses,
        generation_temperature=args.generation_temperature,
        evolution_temperature=args.evolution_temperature,
        reflection_temperature=args.reflection_temperature,
        elo_k_factor=args.elo_k_factor,
        top_k_hypotheses=args.top_k_hypotheses,
        max_literature_results=args.max_literature_results,
        enable_prior_art_check=False if args.disable_prior_art_check else None,
        prior_art_queries_per_idea=args.prior_art_queries_per_idea,
        prior_art_results_per_query=args.prior_art_results_per_query,
        prior_art_embedding_candidates=args.prior_art_embedding_candidates,
        prior_art_review_top_k=args.prior_art_review_top_k,
        prior_art_similarity_threshold=args.prior_art_similarity_threshold,
        prior_art_repair_attempts=args.prior_art_repair_attempts,
        ranking_matches_per_cycle=args.ranking_matches_per_cycle,
        proximity_similarity_threshold=args.proximity_similarity_threshold,
        deep_review_top_k=args.deep_review_top_k,
        hypothesis_decay_fraction=args.hypothesis_decay_fraction,
        enable_safety_review=False if args.disable_safety_review else None,
        max_concurrency=args.max_concurrency,
    )
    context = ContextMemory()
    context.reset_for_goal(research_goal)
    supervisor = SupervisorAgent()

    cycles = []
    for cycle_index in range(args.cycles):
        logger.info("Running requested cycle %d/%d", cycle_index + 1, args.cycles)
        def _progress_checkpoint(step_name: str, partial_cycle: Dict[str, Any], partial_context: ContextMemory) -> None:
            logger.info("Writing progress checkpoint after step: %s", step_name)
            _write_checkpoint(cycles + [partial_cycle], args.output_dir, partial_context)

        cycle_details = supervisor.run_cycle(research_goal, context, progress_callback=_progress_checkpoint)
        cycles.append(cycle_details)
        _write_checkpoint(cycles, args.output_dir, context)

        generation_step = cycle_details.get("steps", {}).get("generation", {})
        generated_count = len(generation_step.get("hypotheses", []))
        generation_errors = generation_step.get("errors", [])
        if (
            not args.allow_empty_cycles
            and generated_count == 0
            and not context.hypotheses
            and (generation_errors or cycle_details.get("errors"))
        ):
            written = _write_outputs(cycles, args.output_dir, context)
            raise WorkflowRunError(
                "Generation produced zero hypotheses. Aborting because the run would otherwise "
                "continue with fallback reviews/rankings and produce an empty report. "
                f"First errors: {generation_errors or cycle_details.get('errors')}",
                written_outputs=written,
            )

    return _write_outputs(cycles, args.output_dir, context)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the AI co-scientist workflow from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--goal", required=True, help="Research goal to investigate.")
    parser.add_argument("--constraints", help="Optional JSON file containing explicit constraints.")
    parser.add_argument(
        "--cycles",
        type=int,
        default=_config_int("cycles", 1, minimum=1),
        help="Number of workflow cycles to run in one process. Defaults to config.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default=config.get("output_dir", "results"),
        help="Directory where JSON and Markdown reports are written.",
    )
    parser.add_argument("--model", help="Override the configured thinking LLM model.")
    parser.add_argument("--critic-model", help="Override the configured LLM model used for review/ranking.")
    parser.add_argument("--num-hypotheses", type=int, default=None, help="New hypotheses to generate per cycle.")
    parser.add_argument("--top-k-hypotheses", type=int, default=None, help="Top hypotheses used as evolution sources.")
    parser.add_argument("--generation-temperature", type=float, default=None, help="Temperature for fresh hypothesis generation.")
    parser.add_argument("--evolution-temperature", type=float, default=None, help="Temperature for evolved hypothesis generation.")
    parser.add_argument("--reflection-temperature", type=float, default=None, help="Temperature for review/ranking/meta-review.")
    parser.add_argument("--elo-k-factor", type=int, default=None, help="Elo K-factor for tournament ranking.")
    parser.add_argument("--max-literature-results", type=int, default=None, help="Maximum literature notes to attach per search bundle.")
    parser.add_argument("--disable-prior-art-check", action="store_true", help="Skip large-pool prior-art duplicate checking.")
    parser.add_argument("--prior-art-queries-per-idea", type=int, default=None, help="Maximum prior-art search queries planned for each idea.")
    parser.add_argument("--prior-art-results-per-query", type=int, default=None, help="Large-pool papers retrieved per prior-art query.")
    parser.add_argument("--prior-art-embedding-candidates", type=int, default=None, help="Lexical prefilter size before embedding recall.")
    parser.add_argument("--prior-art-review-top-k", type=int, default=None, help="Top recalled prior-art papers sent to LLM audit.")
    parser.add_argument("--prior-art-similarity-threshold", type=float, default=None, help="Minimum recall score before LLM duplicate audit.")
    parser.add_argument("--prior-art-repair-attempts", type=int, default=None, help="Repair attempts allowed after a duplicate prior-art audit.")
    parser.add_argument("--ranking-matches-per-cycle", type=int, default=None, help="Maximum pairwise ranking matches per ranking stage.")
    parser.add_argument("--proximity-similarity-threshold", type=float, default=None, help="Threshold for proximity clusters.")
    parser.add_argument("--deep-review-top-k", type=int, default=None, help="Active hypotheses to run specialized reflection reviews on.")
    parser.add_argument(
        "--hypothesis-decay-fraction",
        type=float,
        default=None,
        help="Fraction of the lowest-Elo active hypotheses to deactivate at the end of each cycle.",
    )
    parser.add_argument("--max-concurrency", type=int, default=None, help="Maximum concurrent LLM/tool calls for parallel-safe stages.")
    parser.add_argument("--disable-safety-review", action="store_true", help="Skip automated research-goal safety review.")
    parser.add_argument(
        "--allow-empty-cycles",
        action="store_true",
        help="Continue even if the first generation stage returns zero hypotheses. Useful only for debugging.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cycles < 1:
        raise SystemExit("--cycles must be >= 1")

    try:
        written = run_cycles(args)
    except InputValidationError as exc:
        print(f"Input error: {exc}")
        raise SystemExit(1) from exc
    except WorkflowRunError as exc:
        print(f"Workflow aborted: {exc}")
        if exc.written_outputs:
            print(f"Partial JSON report: {exc.written_outputs['json']}")
            print(f"Partial Markdown report: {exc.written_outputs['markdown']}")
        raise SystemExit(1) from exc
    print("Workflow completed.")
    print(f"JSON report: {written['json']}")
    print(f"Markdown report: {written['markdown']}")


if __name__ == "__main__":
    # The configured API key is read by app.llm; this warning only helps script users spot local env issues.
    if not (
        os.getenv("LLM_API_KEY")
        or os.getenv("THINKING_LLM_API_KEY")
        or os.getenv("CRITIC_LLM_API_KEY")
    ):
        logger.info("No LLM API key env var detected; falling back to local/template config if present.")
    main()
