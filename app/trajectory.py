from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _normalize_focus_area(text: str) -> str:
    return " ".join("".join(character if character.isalnum() else " " for character in text.casefold()).split())


def _focus_area_matches(reference: str, candidate: str) -> bool:
    left = _normalize_focus_area(reference)
    right = _normalize_focus_area(candidate)
    if not left or not right:
        return False
    if left == right or left in right or right in left:
        return True

    left_tokens = {token for token in left.split() if len(token) > 2}
    right_tokens = {token for token in right.split() if len(token) > 2}
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens)
    return overlap >= min(2, len(left_tokens), len(right_tokens))


def _unique_focus_ratio(items: Sequence[Dict[str, Any]]) -> float:
    if not items:
        return 0.0
    labels = {
        _normalize_focus_area(str(item.get("focus_area") or item.get("title") or ""))
        for item in items
        if str(item.get("focus_area") or item.get("title") or "").strip()
    }
    labels.discard("")
    return len(labels) / len(items) if items else 0.0


def _dominant_focus_share(items: Sequence[Dict[str, Any]]) -> float:
    if not items:
        return 0.0
    counts: Dict[str, int] = {}
    for item in items:
        label = _normalize_focus_area(str(item.get("focus_area") or item.get("title") or ""))
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return 0.0
    return max(counts.values()) / len(items)


def _top_hypotheses_from_cycle(cycle: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    final_step = (
        cycle.get("steps", {}).get("codex_reranking_final")
        or cycle.get("steps", {}).get("ranking_final")
        or cycle.get("steps", {}).get("codex_reranking")
        or cycle.get("steps", {}).get("ranking")
        or {}
    )
    hypotheses = final_step.get("hypotheses", [])
    return sorted(hypotheses, key=lambda item: item.get("elo_score", 0), reverse=True)[:limit]


def _adjacent_turnover(sets: Sequence[set[str]]) -> float:
    scores: List[float] = []
    for previous, current in zip(sets, sets[1:]):
        if not previous and not current:
            continue
        union = previous | current
        if not union:
            continue
        scores.append(1.0 - len(previous & current) / len(union))
    return sum(scores) / len(scores) if scores else 0.0


def _coverage_ratio(requested_focus_areas: Sequence[str], hypotheses: Sequence[Dict[str, Any]]) -> float:
    if not requested_focus_areas:
        return 0.0
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
    return covered / len(requested_focus_areas)


def _requested_focus_areas(data: Dict[str, Any]) -> List[str]:
    final_context = data.get("final_context", {})
    research_plan = final_context.get("research_plan") or {}
    focus_areas = research_plan.get("focus_areas") or []
    if focus_areas:
        return [str(item) for item in focus_areas if str(item).strip()]

    for cycle in data.get("cycles", []):
        focus_areas = (cycle.get("research_plan") or {}).get("focus_areas") or []
        if focus_areas:
            return [str(item) for item in focus_areas if str(item).strip()]
    return []


def analyze_run_payload(data: Dict[str, Any], label: str = "") -> Dict[str, Any]:
    cycles = data.get("cycles", [])
    final_context = data.get("final_context", {})
    hypotheses_map = final_context.get("hypotheses", {}) if isinstance(final_context, dict) else {}
    all_hypotheses = list(hypotheses_map.values()) if isinstance(hypotheses_map, dict) else []
    active_hypotheses = [hypothesis for hypothesis in all_hypotheses if hypothesis.get("is_active")]
    evolved_hypotheses = [hypothesis for hypothesis in all_hypotheses if hypothesis.get("origin") == "evolution"]
    evolved_with_parents = [hypothesis for hypothesis in evolved_hypotheses if hypothesis.get("parent_ids")]
    ranked_final = sorted(all_hypotheses, key=lambda item: item.get("elo_score", 0), reverse=True)
    top10 = ranked_final[:10]

    cycle_top1 = [_top_hypotheses_from_cycle(cycle, limit=1) for cycle in cycles]
    top1_titles = [items[0].get("title", "") for items in cycle_top1 if items]
    top1_change_count = sum(
        1
        for previous, current in zip(top1_titles, top1_titles[1:])
        if _normalize_focus_area(previous) != _normalize_focus_area(current)
    )
    top3_sets = [
        {
            str(item.get("id") or item.get("title") or "")
            for item in _top_hypotheses_from_cycle(cycle, limit=3)
            if str(item.get("id") or item.get("title") or "")
        }
        for cycle in cycles
    ]

    requested_focus_areas = _requested_focus_areas(data)
    active_focus_coverage = _coverage_ratio(requested_focus_areas, active_hypotheses)
    top10_focus_coverage = _coverage_ratio(requested_focus_areas, top10)
    top10_evolved_share = len([hypothesis for hypothesis in top10 if hypothesis.get("origin") == "evolution"]) / len(top10) if top10 else 0.0
    late_entry_threshold = max(1, len(cycles) // 2)
    late_entry_share = (
        len([hypothesis for hypothesis in top10 if int(hypothesis.get("created_in_iteration", 0) or 0) > late_entry_threshold]) / len(top10)
        if top10 else 0.0
    )
    lineage_coverage = len(evolved_with_parents) / len(evolved_hypotheses) if evolved_hypotheses else 0.0
    diversity_ratio = _unique_focus_ratio(top10)
    cluster_balance = 1.0 - _dominant_focus_share(top10)
    frontier_turnover = _adjacent_turnover(top3_sets)
    top1_change_ratio = top1_change_count / max(1, len(cycles) - 1)

    coverage_component = (active_focus_coverage + top10_focus_coverage) / 2 if requested_focus_areas else 0.0
    diversity_component = (diversity_ratio + cluster_balance) / 2 if top10 else 0.0
    dynamics_component = min(1.0, (top1_change_ratio * 1.5 + frontier_turnover) / 2)
    trajectory_score = round(
        100
        * (
            0.25 * coverage_component
            + 0.25 * diversity_component
            + 0.15 * lineage_coverage
            + 0.15 * top10_evolved_share
            + 0.10 * dynamics_component
            + 0.10 * late_entry_share
        ),
        2,
    )

    provider_error_cycles = 0
    generation_shortfall_cycles = 0
    evolution_shortfall_cycles = 0
    generation_schema_violation_cycles = 0
    evolution_schema_violation_cycles = 0
    generation_fill_ratios: List[float] = []
    evolution_fill_ratios: List[float] = []
    for cycle in cycles:
        errors = [str(error) for error in cycle.get("errors", [])]
        if any("502" in error or "Connection error" in error for error in errors):
            provider_error_cycles += 1
        steps = cycle.get("steps", {})
        generation_step = steps.get("generation") or {}
        evolution_step = steps.get("evolution") or {}

        generation_requested = int(generation_step.get("requested_hypotheses") or len(generation_step.get("hypotheses", [])) or 0)
        generation_count = len(generation_step.get("hypotheses", []))
        if generation_requested > 0:
            generation_fill_ratios.append(min(1.0, generation_count / generation_requested))
            if generation_count < generation_requested:
                generation_shortfall_cycles += 1
        generation_payload_keys = generation_step.get("raw_payload_keys") or []
        if generation_payload_keys:
            first_generation_keys = generation_payload_keys[0] if isinstance(generation_payload_keys[0], list) else generation_payload_keys
            if "hypotheses" not in first_generation_keys:
                generation_schema_violation_cycles += 1

        evolution_requested = int(evolution_step.get("requested_hypotheses") or len(evolution_step.get("hypotheses", [])) or 0)
        evolution_count = len(evolution_step.get("hypotheses", []))
        if evolution_requested > 0:
            evolution_fill_ratios.append(min(1.0, evolution_count / evolution_requested))
            if evolution_count < evolution_requested:
                evolution_shortfall_cycles += 1
        evolution_payload_keys = evolution_step.get("raw_payload_keys") or []
        if evolution_payload_keys:
            first_evolution_keys = evolution_payload_keys[0] if isinstance(evolution_payload_keys[0], list) else evolution_payload_keys
            if "hypotheses" not in first_evolution_keys:
                evolution_schema_violation_cycles += 1

    return {
        "label": label,
        "cycles_completed": len(cycles),
        "total_hypotheses": len(all_hypotheses),
        "active_hypotheses": len(active_hypotheses),
        "evolved_hypotheses": len(evolved_hypotheses),
        "lineage_coverage": round(lineage_coverage, 3),
        "requested_focus_areas": requested_focus_areas,
        "active_focus_coverage": round(active_focus_coverage, 3),
        "top10_focus_coverage": round(top10_focus_coverage, 3),
        "top10_unique_focus_ratio": round(diversity_ratio, 3),
        "top10_cluster_balance": round(cluster_balance, 3),
        "top10_evolved_share": round(top10_evolved_share, 3),
        "late_entry_share": round(late_entry_share, 3),
        "top1_change_count": top1_change_count,
        "top3_turnover": round(frontier_turnover, 3),
        "provider_error_cycles": provider_error_cycles,
        "generation_shortfall_cycles": generation_shortfall_cycles,
        "evolution_shortfall_cycles": evolution_shortfall_cycles,
        "generation_schema_violation_cycles": generation_schema_violation_cycles,
        "evolution_schema_violation_cycles": evolution_schema_violation_cycles,
        "avg_generation_fill_ratio": round(sum(generation_fill_ratios) / len(generation_fill_ratios), 3) if generation_fill_ratios else 0.0,
        "avg_evolution_fill_ratio": round(sum(evolution_fill_ratios) / len(evolution_fill_ratios), 3) if evolution_fill_ratios else 0.0,
        "trajectory_score": trajectory_score,
        "top1_titles": top1_titles,
        "top10_titles": [hypothesis.get("title", "") for hypothesis in top10],
        "cycle_generation_counts": [
            len((cycle.get("steps", {}).get("generation") or {}).get("hypotheses", []))
            for cycle in cycles
        ],
        "cycle_evolution_counts": [
            len((cycle.get("steps", {}).get("evolution") or {}).get("hypotheses", []))
            for cycle in cycles
        ],
    }


def analyze_run_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return analyze_run_payload(data, label=str(path))


def build_markdown_summary(reports: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = ["# Trajectory Benchmark\n"]
    for report in sorted(reports, key=lambda item: item["trajectory_score"], reverse=True):
        parts.append(f"## {report['label']}\n")
        parts.append(f"- trajectory_score: `{report['trajectory_score']}`\n")
        parts.append(f"- cycles_completed: `{report['cycles_completed']}`\n")
        parts.append(f"- total_hypotheses: `{report['total_hypotheses']}`\n")
        parts.append(f"- active_focus_coverage: `{report['active_focus_coverage']}`\n")
        parts.append(f"- top10_focus_coverage: `{report['top10_focus_coverage']}`\n")
        parts.append(f"- top10_unique_focus_ratio: `{report['top10_unique_focus_ratio']}`\n")
        parts.append(f"- top10_cluster_balance: `{report['top10_cluster_balance']}`\n")
        parts.append(f"- top10_evolved_share: `{report['top10_evolved_share']}`\n")
        parts.append(f"- lineage_coverage: `{report['lineage_coverage']}`\n")
        parts.append(f"- late_entry_share: `{report['late_entry_share']}`\n")
        parts.append(f"- top1_change_count: `{report['top1_change_count']}`\n")
        parts.append(f"- top3_turnover: `{report['top3_turnover']}`\n")
        parts.append(f"- provider_error_cycles: `{report['provider_error_cycles']}`\n")
        parts.append(f"- generation_shortfall_cycles: `{report['generation_shortfall_cycles']}`\n")
        parts.append(f"- evolution_shortfall_cycles: `{report['evolution_shortfall_cycles']}`\n")
        parts.append(f"- generation_schema_violation_cycles: `{report['generation_schema_violation_cycles']}`\n")
        parts.append(f"- evolution_schema_violation_cycles: `{report['evolution_schema_violation_cycles']}`\n")
        parts.append(f"- avg_generation_fill_ratio: `{report['avg_generation_fill_ratio']}`\n")
        parts.append(f"- avg_evolution_fill_ratio: `{report['avg_evolution_fill_ratio']}`\n")
        if report["requested_focus_areas"]:
            parts.append("- requested_focus_areas:\n")
            parts.extend(f"  - {item}\n" for item in report["requested_focus_areas"])
        if report["top10_titles"]:
            parts.append("- top10_titles:\n")
            parts.extend(f"  - {item}\n" for item in report["top10_titles"])
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
    parser = argparse.ArgumentParser(description="Analyze trajectory quality of co-scientist run JSON files.")
    parser.add_argument("paths", nargs="+", help="Run JSON files or directories containing them.")
    parser.add_argument("--markdown-out", help="Optional markdown output path.")
    parser.add_argument("--json-out", help="Optional JSON output path.")
    args = parser.parse_args()

    reports = [analyze_run_file(path) for path in _iter_paths(args.paths)]
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = build_markdown_summary(reports)
    if args.markdown_out:
        Path(args.markdown_out).write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
