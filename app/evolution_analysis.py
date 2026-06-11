from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .trajectory import _focus_area_matches, _normalize_focus_area, _requested_focus_areas


def _safe_hypotheses(step: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(step, dict):
        return []
    hypotheses = step.get("hypotheses", [])
    return hypotheses if isinstance(hypotheses, list) else []


def _step_duration(step: Dict[str, Any] | None) -> float:
    if not isinstance(step, dict):
        return 0.0
    try:
        return float(step.get("duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _coverage_count(requested_focus_areas: Sequence[str], hypotheses: Sequence[Dict[str, Any]]) -> int:
    covered = 0
    for requested in requested_focus_areas:
        if any(
            _focus_area_matches(
                requested,
                str(hypothesis.get("focus_area") or hypothesis.get("title") or ""),
            )
            for hypothesis in hypotheses
        ):
            covered += 1
    return covered


def _lineage_path(hypothesis_id: str, hypotheses_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    path: List[Dict[str, Any]] = []
    seen = set()
    current_id = hypothesis_id
    while current_id and current_id not in seen:
        seen.add(current_id)
        hypothesis = hypotheses_by_id.get(current_id)
        if not hypothesis:
            break
        path.append(hypothesis)
        parent_ids = hypothesis.get("parent_ids") or []
        current_id = str(parent_ids[0]) if parent_ids else ""
    return path


def analyze_evolution_payload(data: Dict[str, Any], label: str = "") -> Dict[str, Any]:
    cycles = data.get("cycles", [])
    final_context = data.get("final_context", {}) if isinstance(data.get("final_context"), dict) else {}
    hypotheses_by_id = final_context.get("hypotheses", {}) if isinstance(final_context.get("hypotheses"), dict) else {}
    all_hypotheses = list(hypotheses_by_id.values())
    ranked_final = sorted(all_hypotheses, key=lambda item: item.get("elo_score", 0), reverse=True)
    top10 = ranked_final[:10]
    requested_focus_areas = _requested_focus_areas(data)

    cycle_summaries: List[Dict[str, Any]] = []
    focus_counter = Counter()
    bottleneck_counter = Counter()
    operator_counter = Counter()
    evolution_parent_counter = Counter()
    evolution_examples: Dict[str, List[str]] = defaultdict(list)

    for hypothesis in top10:
        focus_label = str(hypothesis.get("focus_area") or hypothesis.get("title") or "").strip()
        if focus_label:
            focus_counter[focus_label] += 1
        bottleneck = str(hypothesis.get("primary_bottleneck") or "").strip()
        if bottleneck:
            bottleneck_counter[bottleneck] += 1

    evolved = [hypothesis for hypothesis in all_hypotheses if hypothesis.get("origin") == "evolution"]
    for hypothesis in evolved:
        operator = str(hypothesis.get("mutation_operator") or hypothesis.get("generation_strategy") or "unknown").strip()
        operator_counter[operator] += 1
        evolution_parent_counter[len(hypothesis.get("parent_ids") or [])] += 1
        if len(evolution_examples[operator]) < 3:
            evolution_examples[operator].append(str(hypothesis.get("title") or "Untitled"))

    for cycle in cycles:
        steps = cycle.get("steps", {})
        generation_step = steps.get("generation") or {}
        evolution_step = steps.get("evolution") or {}
        final_step = (
            steps.get("opencode_reranking_final")
            or steps.get("codex_reranking_final")
            or steps.get("ranking_final")
            or steps.get("opencode_reranking")
            or steps.get("codex_reranking")
            or steps.get("ranking")
            or {}
        )
        ranked_hypotheses = sorted(_safe_hypotheses(final_step), key=lambda item: item.get("elo_score", 0), reverse=True)
        top3 = ranked_hypotheses[:3]
        cycle_summaries.append(
            {
                "iteration": cycle.get("iteration"),
                "generation_count": len(_safe_hypotheses(generation_step)),
                "generation_requested": generation_step.get("requested_hypotheses"),
                "generation_duration": round(_step_duration(generation_step), 2),
                "evolution_count": len(_safe_hypotheses(evolution_step)),
                "evolution_requested": evolution_step.get("requested_hypotheses"),
                "evolution_duration": round(_step_duration(evolution_step), 2),
                "review_evolved_duration": round(_step_duration(steps.get("review_evolved")), 2),
                "ranking_final_duration": round(_step_duration(steps.get("ranking_final")), 2),
                "opencode_reranking_final_duration": round(
                    _step_duration(steps.get("opencode_reranking_final") or steps.get("codex_reranking_final")),
                    2,
                ),
                "focus_coverage_top3": _coverage_count(requested_focus_areas, top3),
                "top3_titles": [str(item.get("title") or "") for item in top3],
                "top3_origins": [str(item.get("origin") or "") for item in top3],
                "trajectory_state_generation": generation_step.get("trajectory_state", {}),
                "trajectory_state_evolution": evolution_step.get("trajectory_state", {}),
                "adaptive_tuning": evolution_step.get("adaptive_tuning", {}),
            }
        )

    top_lineages = []
    for hypothesis in top10:
        path = _lineage_path(str(hypothesis.get("id") or ""), hypotheses_by_id)
        if not path:
            continue
        top_lineages.append(
            {
                "terminal_title": hypothesis.get("title", ""),
                "terminal_origin": hypothesis.get("origin", ""),
                "terminal_elo": round(float(hypothesis.get("elo_score", 0.0) or 0.0), 2),
                "path_titles": [str(item.get("title") or "") for item in path],
                "path_origins": [str(item.get("origin") or "") for item in path],
                "mutation_operators": [
                    str(item.get("mutation_operator") or item.get("generation_strategy") or "")
                    for item in path
                    if str(item.get("origin") or "") == "evolution"
                ],
            }
        )

    return {
        "label": label,
        "cycle_count": len(cycles),
        "requested_focus_areas": requested_focus_areas,
        "top10_focus_distribution": dict(focus_counter.most_common()),
        "top10_bottleneck_distribution": dict(bottleneck_counter.most_common()),
        "evolution_operator_distribution": dict(operator_counter.most_common()),
        "evolution_parent_arity_distribution": dict(sorted(evolution_parent_counter.items())),
        "evolution_operator_examples": dict(evolution_examples),
        "cycle_summaries": cycle_summaries,
        "top_lineages": top_lineages[:8],
    }


def analyze_evolution_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return analyze_evolution_payload(data, label=str(path))


def build_markdown_summary(reports: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = ["# Evolution Deep Analysis\n"]
    for report in reports:
        parts.append(f"## {report['label']}\n")
        parts.append(f"- cycles: `{report['cycle_count']}`\n")
        parts.append(f"- top10_focus_distribution: `{report['top10_focus_distribution']}`\n")
        parts.append(f"- top10_bottleneck_distribution: `{report['top10_bottleneck_distribution']}`\n")
        parts.append(f"- evolution_operator_distribution: `{report['evolution_operator_distribution']}`\n")
        parts.append(f"- evolution_parent_arity_distribution: `{report['evolution_parent_arity_distribution']}`\n")
        parts.append("\n### Cycle Summaries\n")
        for cycle in report["cycle_summaries"]:
            parts.append(
                f"- iter {cycle['iteration']}: "
                f"gen={cycle['generation_count']}/{cycle['generation_requested']} "
                f"({cycle['generation_duration']}s), "
                f"evo={cycle['evolution_count']}/{cycle['evolution_requested']} "
                f"({cycle['evolution_duration']}s), "
                f"review_evolved={cycle['review_evolved_duration']}s, "
                f"ranking_final={cycle['ranking_final_duration']}s, "
                f"opencode_reranking_final={cycle['opencode_reranking_final_duration']}s, "
                f"top3={cycle['top3_titles']}\n"
            )
        parts.append("\n### Top Lineages\n")
        for lineage in report["top_lineages"]:
            parts.append(
                f"- {lineage['terminal_title']} ({lineage['terminal_origin']}, elo={lineage['terminal_elo']}): "
                f"{' -> '.join(lineage['path_titles'])}\n"
            )
        parts.append("\n")
    return "".join(parts)


def _iter_paths(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.rglob("co_scientist_run_*.json")))
        else:
            paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep evolution analysis for co-scientist run JSON files.")
    parser.add_argument("paths", nargs="+", help="Run JSON files or directories containing them.")
    parser.add_argument("--markdown-out", help="Optional markdown output path.")
    parser.add_argument("--json-out", help="Optional JSON output path.")
    args = parser.parse_args()

    reports = [analyze_evolution_file(path) for path in _iter_paths(args.paths)]
    markdown = build_markdown_summary(reports)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_out:
        Path(args.markdown_out).write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
