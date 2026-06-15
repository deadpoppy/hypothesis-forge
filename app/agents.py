from __future__ import annotations

from collections import Counter
import hashlib
import json
import itertools
import math
import re
import subprocess
import time
from typing import Any, Dict, List, Sequence, Tuple

from .config import config
from .llm import LLMCallError, call_json
from .models import ContextMemory, Hypothesis, ResearchGoal, ResearchPlan, StepResult
from .prompts import (
    build_evolution_refill_messages,
    build_evolution_messages,
    build_generation_refill_messages,
    build_generation_messages,
    build_ranking_messages,
    build_ranking_reaudit_messages,
    build_research_plan_messages,
    build_unified_review_messages,
)
from .utils import (
    coerce_string_list,
    dedupe_preserve_order,
    generate_unique_id,
    generate_visjs_data,
    logger,
    run_concurrently,
    similarity_score,
)


def _run_opencode(prompt: str, timeout: int) -> subprocess.CompletedProcess:
    """Invoke `opencode run` feeding the prompt via stdin.

    Prompts can exceed MAX_ARG_STRLEN (128KB) when they carry full hypothesis
    context, which makes argv-based invocation fail with E2BIG. Passing the
    prompt through stdin avoids the per-argument hard limit entirely.
    """
    command = ["opencode", "run", "--dangerously-skip-permissions"]
    return subprocess.run(
        command,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _compose_hypothesis_text(payload: Dict[str, Any]) -> str:
    core_text = (
        payload.get("core_hypothesis")
        or payload.get("hypothesis")
        or payload.get("proposal")
        or payload.get("description")
        or payload.get("text")
        or payload.get("summary")
        or ""
    )
    parts = [str(core_text).strip()]
    problem_framing = str(payload.get("problem_framing") or payload.get("problem_frame") or "").strip()
    central_insight = str(payload.get("central_insight") or payload.get("key_insight") or "").strip()
    theoretical_story = str(
        payload.get("theoretical_story")
        or payload.get("causal_story")
        or payload.get("story")
        or payload.get("storyline")
        or ""
    ).strip()
    mechanism = str(payload.get("mechanism", "")).strip()
    novelty = str(payload.get("novelty_rationale") or payload.get("rationale") or "").strip()
    anti_combination = str(
        payload.get("why_not_simple_combination")
        or payload.get("why_not_just_combination")
        or payload.get("not_just_combination")
        or ""
    ).strip()
    if problem_framing:
        parts.append(f"Problem framing: {problem_framing}")
    if central_insight:
        parts.append(f"Central insight: {central_insight}")
    if theoretical_story:
        parts.append(f"Theoretical story: {theoretical_story}")
    if mechanism:
        parts.append(f"Mechanism: {mechanism}")
    if novelty:
        parts.append(f"Novelty rationale: {novelty}")
    if anti_combination:
        parts.append(f"Why not simple combination: {anti_combination}")
    return "\n".join(part for part in parts if part)


def _round_score(value: float) -> float:
    return round(float(value), 4)


def _normalize_paper_title(title: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(title or "").casefold())
    return re.sub(r"\s+", " ", normalized).strip()


def _paper_note_key(note: Dict[str, Any]) -> str:
    doi = str(note.get("doi") or "").strip().casefold()
    if doi:
        return f"doi:{doi}"
    arxiv_id = str(note.get("arxiv_id") or "").strip().casefold()
    if arxiv_id:
        normalized_arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        return f"arxiv:{normalized_arxiv_id}"
    url = str(note.get("url") or note.get("arxiv_url") or note.get("pdf_url") or "").strip().casefold()
    if url:
        return f"url:{url}"
    title = _normalize_paper_title(str(note.get("title") or ""))
    return f"title:{title}" if title else ""


def dedupe_notes(notes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for note in notes:
        if not isinstance(note, dict):
            continue
        key = _paper_note_key(note)
        if not key:
            continue
        if key not in merged:
            merged[key] = dict(note)
            continue
        existing = merged[key]
        sources = dedupe_preserve_order(
            coerce_string_list(existing.get("sources", []))
            + coerce_string_list(note.get("sources", []))
            + [str(existing.get("source") or ""), str(note.get("source") or "")]
        )
        existing["sources"] = [source for source in sources if source]
        for field in (
            "title",
            "summary",
            "abstract",
            "authors",
            "citation",
            "published",
            "year",
            "venue",
            "doi",
            "arxiv_id",
            "arxiv_url",
            "url",
            "pdf_url",
            "relevance",
            "novelty_risk",
        ):
            if not existing.get(field) and note.get(field):
                existing[field] = note[field]
    return list(merged.values())


def _build_review_artifact(hypothesis: Hypothesis) -> Dict[str, Any]:
    return {
        "id": hypothesis.hypothesis_id,
        "title": hypothesis.title,
        "summary": hypothesis.review_summary,
        "problem_framing": hypothesis.problem_framing,
        "central_insight": hypothesis.central_insight,
        "theoretical_story": hypothesis.theoretical_story,
        "mechanism": hypothesis.mechanism,
        "why_not_simple_combination": hypothesis.why_not_simple_combination,
        "scores": hypothesis.scores,
        "verdict": hypothesis.review_verdict,
        "weaknesses": hypothesis.failure_modes[:4] + hypothesis.improvement_actions[:4],
        "validation_experiments": hypothesis.validation_experiments[:4],
        "references": hypothesis.references[:5],
        "latest_review": hypothesis.review_artifacts[-1] if hypothesis.review_artifacts else {},
        "latest_opencode_rerank": hypothesis.opencode_rerank_reviews[-1] if hypothesis.opencode_rerank_reviews else {},
    }


def _coerce_mapping(payload: Any, default: Dict[str, Any]) -> Dict[str, Any]:
    return payload if isinstance(payload, dict) else dict(default)


def _extract_hypothesis_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in (
        "hypotheses",
        "new_hypotheses",
        "research_hypotheses",
        "proposals",
        "ideas",
        "candidates",
        "list",
        "items",
        "results",
        "directions",
        "research_directions",
        "hypothesis_list",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    list_values = [
        value
        for value in payload.values()
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value)
    ]
    if len(list_values) == 1:
        return list_values[0]

    if any(key in payload for key in ("title", "core_hypothesis", "hypothesis", "proposal", "description", "text")):
        return [payload]

    return []


def _payload_keys(payload: Any) -> List[str]:
    return list(payload.keys()) if isinstance(payload, dict) else [type(payload).__name__]


def _is_duplicate_candidate(
    candidate: Hypothesis,
    context: ContextMemory,
    staged: Sequence[Hypothesis] = (),
) -> bool:
    for existing in itertools.chain(context.hypotheses.values(), staged):
        if candidate.title.lower() == existing.title.lower():
            return True
        if similarity_score(candidate.text, existing.text) >= 0.94:
            return True
    return False


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


def _focus_area_guidance(research_plan: ResearchPlan, context: ContextMemory) -> Dict[str, List[str]]:
    counts: Counter[str] = Counter()
    labels: Dict[str, str] = {}
    for hypothesis in context.hypotheses.values():
        label = (hypothesis.focus_area or hypothesis.title).strip()
        normalized = _normalize_focus_area(label)
        if not normalized:
            continue
        counts[normalized] += 1
        labels.setdefault(normalized, label)

    uncovered = []
    for area in research_plan.focus_areas:
        if not any(_focus_area_matches(area, label) for label in labels.values()):
            uncovered.append(area)

    overexplored = [
        labels[normalized]
        for normalized, count in counts.most_common()
        if count >= 2
    ]

    return {
        "uncovered_focus_areas": uncovered[:6],
        "overexplored_areas": overexplored[:4],
    }


def _frontier_diversity_snapshot(context: ContextMemory, limit: int = 6) -> Dict[str, Any]:
    frontier = [hypothesis for hypothesis in context.get_ranked_hypotheses() if hypothesis.is_active][:limit]
    counts: Counter[str] = Counter()
    labels: Dict[str, str] = {}
    for hypothesis in frontier:
        label = (hypothesis.focus_area or hypothesis.title).strip()
        normalized = _normalize_focus_area(label)
        if not normalized:
            continue
        counts[normalized] += 1
        labels.setdefault(normalized, label)

    dominant_focus_area = ""
    dominant_share = 0.0
    if counts and frontier:
        normalized, count = counts.most_common(1)[0]
        dominant_focus_area = labels.get(normalized, "")
        dominant_share = round(count / len(frontier), 3)

    return {
        "frontier_size": len(frontier),
        "unique_focus_areas": len(counts),
        "dominant_focus_area": dominant_focus_area,
        "dominant_focus_share": dominant_share,
    }


def _trajectory_state_snapshot(
    research_plan: ResearchPlan,
    context: ContextMemory,
    coverage: Dict[str, List[str]] | None = None,
) -> Dict[str, Any]:
    coverage = coverage or _focus_area_guidance(research_plan, context)
    frontier = _frontier_diversity_snapshot(context)
    ranked = context.get_ranked_hypotheses()
    evolved = [hypothesis for hypothesis in context.hypotheses.values() if hypothesis.origin == "evolution"]
    lineage_coverage = (
        len([hypothesis for hypothesis in evolved if hypothesis.parent_ids]) / len(evolved)
        if evolved else 0.0
    )
    frontier_focus_gaps = [
        area
        for area in research_plan.focus_areas
        if not any(
            _focus_area_matches(area, str(hypothesis.focus_area or hypothesis.title))
            for hypothesis in ranked[:10]
        )
    ]
    top_titles = [hypothesis.title for hypothesis in ranked[:3]]
    recent_history = context.cycle_history[-3:]
    repeated_top1_cycles = 0
    if top_titles:
        current_top = _normalize_focus_area(top_titles[0])
        repeated_top1_cycles = sum(
            1
            for cycle in reversed(recent_history)
            if _normalize_focus_area(
                str((cycle.get("top_hypotheses") or [{}])[0].get("title", ""))
            ) == current_top
        )

    stagnation_signals = []
    if frontier["frontier_size"] >= 4 and frontier["dominant_focus_share"] >= 0.5:
        stagnation_signals.append("frontier_clustered")
    if repeated_top1_cycles >= 2:
        stagnation_signals.append("top1_stable")
    if lineage_coverage < 0.7 and evolved:
        stagnation_signals.append("lineage_loss")
    if coverage["uncovered_focus_areas"]:
        stagnation_signals.append("coverage_gap")

    return {
        "iteration": context.iteration_number,
        "frontier_diversity": frontier,
        "uncovered_focus_areas": coverage["uncovered_focus_areas"],
        "frontier_focus_gaps": frontier_focus_gaps[:4],
        "overexplored_areas": coverage["overexplored_areas"],
        "lineage_coverage": round(lineage_coverage, 3),
        "repeated_top1_cycles": repeated_top1_cycles,
        "stagnation_signals": stagnation_signals,
        "top_titles": top_titles,
    }


def _generation_temperature_for_cycle(
    research_goal: ResearchGoal,
    context: ContextMemory,
    coverage: Dict[str, List[str]],
) -> float:
    temperature = research_goal.generation_temperature
    if context.iteration_number == 0:
        temperature += 0.03
    if coverage["uncovered_focus_areas"]:
        temperature += 0.03

    diversity = _frontier_diversity_snapshot(context)
    if diversity["frontier_size"] >= 4 and diversity["dominant_focus_share"] >= 0.5:
        temperature += 0.02
    return min(max(temperature, 0.0), 0.92)


def _evolution_temperature_for_cycle(
    research_goal: ResearchGoal,
    trajectory_state: Dict[str, Any],
) -> Tuple[float, List[str]]:
    temperature = research_goal.evolution_temperature
    adjustments: List[str] = []
    stagnation_signals = set(coerce_string_list(trajectory_state.get("stagnation_signals", [])))

    if "coverage_gap" in stagnation_signals:
        temperature += 0.03
        adjustments.append("coverage_gap:+0.03")
    if "frontier_clustered" in stagnation_signals:
        temperature += 0.03
        adjustments.append("frontier_clustered:+0.03")
    if "top1_stable" in stagnation_signals:
        temperature += 0.02
        adjustments.append("top1_stable:+0.02")
    lineage_coverage = float(trajectory_state.get("lineage_coverage", 0.0) or 0.0)
    if lineage_coverage and lineage_coverage < 0.7:
        temperature -= 0.04
        adjustments.append("lineage_loss:-0.04")
    return min(max(temperature, 0.0), 0.78), adjustments


def _evolution_source_budget_for_cycle(
    research_goal: ResearchGoal,
    ranked_count: int,
    trajectory_state: Dict[str, Any],
) -> Tuple[int, List[str]]:
    budget = max(3, research_goal.top_k_hypotheses + 1)
    adjustments: List[str] = []
    stagnation_signals = set(coerce_string_list(trajectory_state.get("stagnation_signals", [])))

    if "coverage_gap" in stagnation_signals:
        budget += 1
        adjustments.append("coverage_gap:+1")
    if "frontier_clustered" in stagnation_signals:
        budget += 1
        adjustments.append("frontier_clustered:+1")
    if "lineage_loss" in stagnation_signals:
        budget += 1
        adjustments.append("lineage_loss:+1")

    budget = min(ranked_count, budget, max(6, research_goal.top_k_hypotheses + 2))
    return budget, adjustments


def _is_distinct_from_selected(candidate: Hypothesis, selected: Sequence[Hypothesis], similarity_threshold: float = 0.82) -> bool:
    for existing in selected:
        if similarity_score(candidate.text, existing.text) >= similarity_threshold:
            return False
        existing_area = existing.focus_area or existing.title
        candidate_area = candidate.focus_area or candidate.title
        if candidate_area and existing_area and _focus_area_matches(candidate_area, existing_area):
            return False
    return True


def _matches_any_focus_area(candidate_label: str, references: Sequence[str]) -> bool:
    return any(_focus_area_matches(reference, candidate_label) for reference in references if reference)


def _portfolio_selection_bonus(
    hypothesis: Hypothesis,
    selected: Sequence[Hypothesis],
    preferred_focus_areas: Sequence[str],
    penalized_focus_areas: Sequence[str],
) -> float:
    focus_label = str(hypothesis.focus_area or hypothesis.title).strip()
    selected_focus_labels = [str(item.focus_area or item.title).strip() for item in selected]
    selected_bottlenecks = {
        str(item.primary_bottleneck).strip().casefold()
        for item in selected
        if str(item.primary_bottleneck).strip()
    }
    bonus = 0.0

    if focus_label and _matches_any_focus_area(focus_label, preferred_focus_areas):
        bonus += 2.5
    if focus_label and not _matches_any_focus_area(focus_label, selected_focus_labels):
        bonus += 1.75
    if focus_label and _matches_any_focus_area(focus_label, penalized_focus_areas):
        bonus -= 1.25

    bottleneck = str(hypothesis.primary_bottleneck).strip().casefold()
    if bottleneck and bottleneck not in selected_bottlenecks:
        bonus += 0.35

    if hypothesis.origin == "evolution" and hypothesis.parent_ids:
        bonus += 0.15

    return bonus


def _select_evolution_sources(
    ranked: Sequence[Hypothesis],
    source_budget: int,
    preferred_focus_areas: Sequence[str] = (),
    penalized_focus_areas: Sequence[str] = (),
) -> List[Hypothesis]:
    if source_budget <= 0:
        return []

    frontier = list(ranked[: max(source_budget * 2, 6)])
    if not frontier:
        return []

    selected: List[Hypothesis] = [frontier[0]]
    rank_positions = {
        hypothesis.hypothesis_id: index
        for index, hypothesis in enumerate(ranked)
    }
    recent_outsiders = sorted(
        [hypothesis for hypothesis in ranked if hypothesis not in selected],
        key=lambda hypothesis: (
            hypothesis.created_in_iteration,
            hypothesis.scores.get("novelty", 0.0),
            hypothesis.elo_score,
        ),
        reverse=True,
    )
    candidate_pool: List[Hypothesis] = []
    seen_candidate_ids = set()
    for candidate in frontier[1:] + recent_outsiders + list(ranked):
        if candidate.hypothesis_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(candidate.hypothesis_id)
        candidate_pool.append(candidate)

    def candidate_priority(hypothesis: Hypothesis) -> tuple[float, float, float, int]:
        focus_label = str(hypothesis.focus_area or hypothesis.title).strip()
        frontier_focus_count = sum(
            1
            for item in frontier
            if focus_label and _focus_area_matches(focus_label, str(item.focus_area or item.title).strip())
        )
        frontier_scarcity_bonus = 0.4 if frontier_focus_count <= 1 else -0.15 * max(0, frontier_focus_count - 1)
        recency_bonus = min(hypothesis.created_in_iteration, 6) * 0.08
        portfolio_bonus = _portfolio_selection_bonus(
            hypothesis,
            selected=selected,
            preferred_focus_areas=preferred_focus_areas,
            penalized_focus_areas=penalized_focus_areas,
        )
        return (
            portfolio_bonus + frontier_scarcity_bonus + recency_bonus,
            hypothesis.elo_score,
            hypothesis.scores.get("novelty", 0.0),
            -rank_positions.get(hypothesis.hypothesis_id, len(ranked)),
        )

    while len(selected) < source_budget:
        remaining = [candidate for candidate in candidate_pool if candidate not in selected]
        if not remaining:
            break

        distinct_candidates = [
            candidate for candidate in remaining if _is_distinct_from_selected(candidate, selected)
        ]
        if distinct_candidates:
            selected.append(max(distinct_candidates, key=candidate_priority))
            continue

        selected.append(max(remaining, key=candidate_priority))

    return selected[:source_budget]


def _prioritize_diverse_hypotheses(
    hypotheses: Sequence[Hypothesis],
    target_count: int,
    preferred_focus_areas: Sequence[str] = (),
    penalized_focus_areas: Sequence[str] = (),
) -> List[Hypothesis]:
    if target_count <= 0:
        return []

    selected: List[Hypothesis] = []
    remaining = list(hypotheses)

    while remaining and len(selected) < target_count:
        best_index = 0
        best_score: tuple[float, int] | None = None
        for index, hypothesis in enumerate(remaining):
            focus_key = _normalize_focus_area(hypothesis.focus_area) or _normalize_focus_area(hypothesis.title)
            score = _portfolio_selection_bonus(
                hypothesis,
                selected=selected,
                preferred_focus_areas=preferred_focus_areas,
                penalized_focus_areas=penalized_focus_areas,
            )
            if focus_key and any(
                _focus_area_matches(focus_key, str(item.focus_area or item.title).strip())
                for item in selected
            ):
                score -= 0.5
            rank_key = (score, -index)
            if best_score is None or rank_key > best_score:
                best_score = rank_key
                best_index = index
        selected.append(remaining.pop(best_index))

    return selected


def _parse_json_object_from_text(text: str) -> Dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        return {}
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass

    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if code_block:
        try:
            payload = json.loads(code_block.group(1))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(stripped[start : end + 1])
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _opencode_literature_timeout() -> int:
    timeout_value = config.get(
        "opencode_literature_timeout_seconds",
        config.get("opencode_rerank_timeout_seconds", 900),
    )
    try:
        return max(1, int(timeout_value or 900))
    except (TypeError, ValueError):
        return 900


def _opencode_literature_max_papers(research_goal: ResearchGoal) -> int:
    try:
        goal_limit = max(0, int(research_goal.max_literature_results or 0))
    except (TypeError, ValueError):
        goal_limit = 0
    if goal_limit <= 0:
        return 0

    configured = config.get("opencode_literature_max_papers", goal_limit)
    try:
        value = int(configured)
    except (TypeError, ValueError):
        value = goal_limit
    return min(goal_limit, max(0, value))


class ResearchPlanAgent:
    def create_plan(self, research_goal: ResearchGoal) -> ResearchPlan:
        fallback = ResearchPlan.from_goal(research_goal)
        try:
            payload = call_json(
                build_research_plan_messages(research_goal.prompt_context(), research_goal.constraints),
                model=research_goal.llm_model,
                temperature=0.2,
                profile="thinking",
            )
            plan = ResearchPlan.from_dict(payload, research_goal)
            logger.info("Structured research plan created for goal: %s", research_goal.description)
            return plan
        except LLMCallError as exc:
            logger.warning("Falling back to local research plan: %s", exc)
            return fallback


class GenerationAgent:
    def _payload_to_hypothesis(self, payload: Dict[str, Any], iteration: int) -> Hypothesis:
        title = str(payload.get("title") or payload.get("name") or payload.get("short_title") or "Untitled hypothesis").strip()
        return Hypothesis(
            hypothesis_id=generate_unique_id("G"),
            title=title,
            text=_compose_hypothesis_text(payload),
            focus_area=str(payload.get("focus_area", "")).strip(),
            primary_bottleneck=str(payload.get("primary_bottleneck", "")).strip(),
            rationale=str(payload.get("novelty_rationale") or payload.get("rationale") or "").strip(),
            mechanism=str(payload.get("mechanism", "")).strip(),
            problem_framing=str(payload.get("problem_framing") or payload.get("problem_frame") or "").strip(),
            central_insight=str(payload.get("central_insight") or payload.get("key_insight") or "").strip(),
            theoretical_story=str(
                payload.get("theoretical_story")
                or payload.get("causal_story")
                or payload.get("story")
                or payload.get("storyline")
                or ""
            ).strip(),
            why_not_simple_combination=str(
                payload.get("why_not_simple_combination")
                or payload.get("why_not_just_combination")
                or payload.get("not_just_combination")
                or ""
            ).strip(),
            generation_strategy=str(payload.get("generation_strategy", "generation")).strip(),
            mutation_operator=str(payload.get("mutation_operator") or "").strip(),
            evolution_delta=str(payload.get("delta_from_parents") or payload.get("diversity_reason") or "").strip(),
            origin="generation",
            predictions=dedupe_preserve_order(payload.get("predictions", [])),
            key_assumptions=dedupe_preserve_order(payload.get("key_assumptions") or payload.get("assumptions") or []),
            validation_experiments=dedupe_preserve_order(
                payload.get("test_plan") or payload.get("validation_experiments") or payload.get("experiments") or []
            ),
            references=dedupe_preserve_order(payload.get("references", [])),
            created_in_iteration=iteration,
        )

    def generate_new_hypotheses(self, research_goal: ResearchGoal, context: ContextMemory) -> StepResult:
        start_time = time.time()
        research_plan = context.research_plan or ResearchPlan.from_goal(research_goal)
        context_memory = context.summarize_hypotheses(limit=8)
        coverage = _focus_area_guidance(research_plan, context)
        generation_temperature = _generation_temperature_for_cycle(research_goal, context, coverage)
        diversity_snapshot = _frontier_diversity_snapshot(context)
        trajectory_state = _trajectory_state_snapshot(research_plan, context, coverage)
        preferred_focus_areas = dedupe_preserve_order(
            coverage["uncovered_focus_areas"] + coerce_string_list(trajectory_state.get("frontier_focus_gaps", []))
        )
        meta_feedback = context.latest_meta_review()
        research_overview = context.latest_research_overview()
        errors: List[str] = []

        try:
            payload = call_json(
                build_generation_messages(
                    research_goal=research_goal.prompt_context(),
                    research_plan=research_plan.to_dict(),
                    context_hypotheses=context_memory,
                    meta_feedback=meta_feedback,
                    research_overview=research_overview,
                    trajectory_state=trajectory_state,
                    num_hypotheses=research_goal.num_hypotheses,
                    uncovered_focus_areas=coverage["uncovered_focus_areas"],
                    overexplored_areas=coverage["overexplored_areas"],
                ),
                model=research_goal.llm_model,
                temperature=generation_temperature,
                profile="thinking",
            )
        except LLMCallError as exc:
            payload = {"hypotheses": []}
            errors.append(str(exc))

        new_hypotheses: List[Hypothesis] = []
        raw_payload_keys = [_payload_keys(payload)]
        hypothesis_items = _extract_hypothesis_items(payload)
        if not hypothesis_items and not errors:
            errors.append("Generation returned valid JSON but no hypothesis list was found.")

        for item in hypothesis_items:
            candidate = self._payload_to_hypothesis(item, context.iteration_number + 1)
            if not candidate.text:
                candidate.text = candidate.title
            if _is_duplicate_candidate(candidate, context, new_hypotheses):
                continue
            new_hypotheses.append(candidate)

        refill_attempts = 0
        refill_added = 0
        while len(new_hypotheses) < research_goal.num_hypotheses and refill_attempts < 2:
            missing_hypotheses = research_goal.num_hypotheses - len(new_hypotheses)
            refill_attempts += 1
            try:
                refill_payload = call_json(
                    build_generation_refill_messages(
                        research_goal=research_goal.prompt_context(),
                        research_plan=research_plan.to_dict(),
                        accepted_hypotheses=[hypothesis.compact_summary() for hypothesis in new_hypotheses],
                        meta_feedback=meta_feedback,
                        research_overview=research_overview,
                        trajectory_state=trajectory_state,
                        missing_hypotheses=missing_hypotheses,
                        uncovered_focus_areas=coverage["uncovered_focus_areas"],
                        overexplored_areas=coverage["overexplored_areas"],
                    ),
                    model=research_goal.llm_model,
                    temperature=min(max(generation_temperature, 0.0) + 0.02, 0.95),
                    profile="thinking",
                )
            except LLMCallError as exc:
                errors.append(f"Generation refill {refill_attempts} failed: {exc}")
                break

            raw_payload_keys.append(_payload_keys(refill_payload))
            refill_items = _extract_hypothesis_items(refill_payload)
            if not refill_items:
                errors.append(f"Generation refill {refill_attempts} returned valid JSON but no hypothesis list was found.")
                break

            added_this_attempt = 0
            for item in refill_items:
                candidate = self._payload_to_hypothesis(item, context.iteration_number + 1)
                if not candidate.text:
                    candidate.text = candidate.title
                if _is_duplicate_candidate(candidate, context, new_hypotheses):
                    continue
                new_hypotheses.append(candidate)
                added_this_attempt += 1

            refill_added += added_this_attempt
            if added_this_attempt == 0:
                errors.append(f"Generation refill {refill_attempts} did not add any distinct hypotheses.")
                break

        new_hypotheses = _prioritize_diverse_hypotheses(
            new_hypotheses,
            research_goal.num_hypotheses,
            preferred_focus_areas=preferred_focus_areas,
            penalized_focus_areas=coverage["overexplored_areas"],
        )

        return StepResult(
            name="generation",
            hypotheses=new_hypotheses,
            data={
                "temperature_used": generation_temperature,
                "frontier_diversity": diversity_snapshot,
                "trajectory_state": trajectory_state,
                "uncovered_focus_areas": coverage["uncovered_focus_areas"],
                "overexplored_areas": coverage["overexplored_areas"],
                "requested_hypotheses": research_goal.num_hypotheses,
                "accepted_hypotheses": len(new_hypotheses),
                "refill_attempts": refill_attempts,
                "refill_added": refill_added,
                "raw_payload_keys": raw_payload_keys,
            },
            errors=errors,
            duration=time.time() - start_time,
        )

    def _looks_duplicate(self, candidate: Hypothesis, context: ContextMemory) -> bool:
        return _is_duplicate_candidate(candidate, context)


class IdeaLiteratureAgent:
    def check_hypotheses(
        self,
        hypotheses: List[Hypothesis],
        research_goal: ResearchGoal,
        context: ContextMemory,
        step_name: str = "idea_literature",
    ) -> StepResult:
        start_time = time.time()
        if not research_goal.enable_idea_literature:
            return StepResult(
                name=step_name,
                hypotheses=hypotheses,
                data={"status": "disabled", "checked_count": 0, "skipped_count": len(hypotheses)},
                duration=time.time() - start_time,
            )

        candidates = [hypothesis for hypothesis in hypotheses if hypothesis.is_active]
        max_papers = _opencode_literature_max_papers(research_goal)
        if max_papers <= 0:
            return StepResult(
                name=step_name,
                hypotheses=hypotheses,
                data={
                    "status": "disabled",
                    "reason": "max_literature_results_or_opencode_literature_max_papers_is_zero",
                    "checked_count": 0,
                    "skipped_count": len(hypotheses),
                    "retrieval_mode": "opencode_per_idea_once",
                },
                duration=time.time() - start_time,
            )

        def check_one(hypothesis: Hypothesis) -> Tuple[Hypothesis, Dict[str, Any], str | None]:
            try:
                return hypothesis, self._fetch_one(hypothesis, research_goal, context, step_name, max_papers), None
            except Exception as exc:  # noqa: BLE001
                return (
                    hypothesis,
                    {
                        "status": "error",
                        "paper_count": len(hypothesis.literature_notes),
                        "short_summary": f"OpenCode literature retrieval failed: {exc}",
                    },
                    f"{hypothesis.hypothesis_id}: {exc}",
                )

        errors: List[str] = []
        results = run_concurrently(candidates, check_one, max_workers=research_goal.max_concurrency)
        audits: Dict[str, Dict[str, Any]] = {}
        papers_by_hypothesis: Dict[str, List[Dict[str, Any]]] = {}
        checked_count = 0
        skipped_count = max(0, len(hypotheses) - len(candidates))
        fetched_count = 0
        error_count = 0

        for hypothesis, audit, error in results:
            if error:
                errors.append(error)
            audits[hypothesis.hypothesis_id] = audit
            papers_by_hypothesis[hypothesis.hypothesis_id] = list(hypothesis.literature_notes)
            status = audit.get("status")
            if status == "skipped_unchanged":
                skipped_count += 1
            elif status in {"fetched", "empty"}:
                checked_count += 1
                if status == "fetched":
                    fetched_count += 1
            elif status == "error":
                error_count += 1

        return StepResult(
            name=step_name,
            hypotheses=hypotheses,
            data={
                "status": "completed",
                "audits": audits,
                "papers_by_hypothesis": papers_by_hypothesis,
                "checked_count": checked_count,
                "skipped_count": skipped_count,
                "fetched_count": fetched_count,
                "error_count": error_count,
                "kept_count": checked_count + skipped_count,
                "repaired_count": 0,
                "rejected_count": 0,
                "retrieval_mode": "opencode_per_idea_once",
                "command": "opencode run --dangerously-skip-permissions",
                "max_papers_per_idea": max_papers,
            },
            errors=errors,
            duration=time.time() - start_time,
        )

    def _fetch_one(
        self,
        hypothesis: Hypothesis,
        research_goal: ResearchGoal,
        context: ContextMemory,
        step_name: str,
        max_papers: int,
    ) -> Dict[str, Any]:
        current_signature = hypothesis.idea_signature()
        if hypothesis.literature_fetch_signature == current_signature and (
            hypothesis.literature_notes
            or hypothesis.literature_fetch_audit.get("method") == "opencode_per_idea_literature"
        ):
            return {
                "status": "skipped_unchanged",
                "paper_count": len(hypothesis.literature_notes),
                "short_summary": "OpenCode literature retrieval skipped because the idea signature is unchanged.",
                "similar_papers": hypothesis.literature_notes,
            }

        timeout = _opencode_literature_timeout()
        prompt = self._build_opencode_literature_prompt(research_goal, context, hypothesis, max_papers)
        raw_stdout = ""
        raw_stderr = ""
        errors: List[str] = []
        payload: Dict[str, Any] = {}
        try:
            completed = _run_opencode(prompt, timeout)
            raw_stdout = completed.stdout or ""
            raw_stderr = completed.stderr or ""
            if completed.returncode != 0:
                errors.append(f"opencode run exited with {completed.returncode}: {(raw_stderr or raw_stdout)[:1200]}")
            else:
                payload = _parse_json_object_from_text(raw_stdout)
                if not payload:
                    errors.append("opencode run returned no parseable JSON payload.")
        except subprocess.TimeoutExpired as exc:
            errors.append(f"opencode run timed out after {timeout}s: {exc}")
        except OSError as exc:
            errors.append(f"could not run opencode run: {exc}")

        notes = self._normalize_literature_notes(payload, hypothesis, max_papers)
        status = "fetched" if notes else "empty"
        if errors and not notes:
            status = "error"
        hypothesis.literature_fetch_signature = current_signature
        hypothesis.literature_notes = notes
        search_summary = str(payload.get("search_summary") or payload.get("summary") or "").strip() if payload else ""
        audit_payload = {
            "status": status,
            "method": "opencode_per_idea_literature",
            "paper_count": len(notes),
            "short_summary": search_summary or f"OpenCode returned {len(notes)} related papers for this idea.",
            "search_terms": coerce_string_list(payload.get("search_terms", [])) if payload else [],
            "errors": errors,
            "stdout_tail": raw_stdout[-2000:],
            "stderr_tail": raw_stderr[-2000:],
        }
        hypothesis.literature_fetch_audit = audit_payload
        if search_summary:
            hypothesis.review_comments = dedupe_preserve_order(
                hypothesis.review_comments + [f"{step_name}: {search_summary}"]
            )
        return audit_payload

    def _build_opencode_literature_prompt(
        self,
        research_goal: ResearchGoal,
        context: ContextMemory,
        hypothesis: Hypothesis,
        max_papers: int,
    ) -> str:
        research_plan = context.research_plan or ResearchPlan.from_goal(research_goal)
        payload = {
            "task": "Find related academic papers for one research idea.",
            "instructions": [
                "Use your paper/web search ability once for this idea.",
                "Return papers useful for novelty checking, evaluation, and future evolution of this exact idea.",
                "Prefer real, citable papers. Include close prior art, relevant benchmarks, and adjacent methods.",
                "Do not rank or reject the idea. Only provide concise paper metadata and why each paper matters.",
                "Return strict JSON only, with no markdown or commentary outside JSON.",
            ],
            "research_goal": research_goal.prompt_context(),
            "research_plan": research_plan.to_dict(),
            "reference_paper": context.reference_paper_context or research_goal.reference_paper_context,
            "idea": hypothesis.to_prompt_dict(),
            "max_papers": max_papers,
            "required_schema": {
                "papers": [
                    {
                        "title": "string",
                        "authors": ["string"],
                        "year": "string",
                        "venue": "string",
                        "url": "string",
                        "doi": "string",
                        "arxiv_id": "string",
                        "summary": "1-2 sentence description",
                        "relevance": "why this paper matters for the idea",
                        "novelty_risk": "low|medium|high|unknown",
                    }
                ],
                "search_terms": ["string"],
                "search_summary": "string",
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _normalize_literature_notes(
        self,
        payload: Dict[str, Any],
        hypothesis: Hypothesis,
        max_papers: int,
    ) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict) or max_papers <= 0:
            return []
        raw_papers = (
            payload.get("papers")
            or payload.get("related_papers")
            or payload.get("literature")
            or payload.get("literature_notes")
            or []
        )
        if not isinstance(raw_papers, list):
            return []

        notes: List[Dict[str, Any]] = []
        for item in raw_papers:
            if isinstance(item, str):
                paper = {"title": item}
            elif isinstance(item, dict):
                paper = item
            else:
                continue
            title = str(paper.get("title") or paper.get("paper_title") or "").strip()
            if not title:
                continue
            summary = str(paper.get("summary") or paper.get("abstract") or paper.get("description") or "").strip()
            year = str(paper.get("year") or paper.get("published") or "").strip()
            venue = str(paper.get("venue") or paper.get("journal") or paper.get("conference") or "").strip()
            citation_parts = [title]
            if year:
                citation_parts.append(year)
            if venue:
                citation_parts.append(venue)
            notes.append(
                {
                    "source": "opencode",
                    "sources": ["opencode"],
                    "selection_reason": "opencode_per_idea_literature",
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "title": title,
                    "summary": summary,
                    "abstract": str(paper.get("abstract") or summary),
                    "authors": coerce_string_list(paper.get("authors", [])),
                    "citation": " - ".join(citation_parts),
                    "published": str(paper.get("published") or ""),
                    "year": year,
                    "venue": venue,
                    "doi": str(paper.get("doi") or "").strip(),
                    "arxiv_id": str(paper.get("arxiv_id") or "").strip(),
                    "arxiv_url": str(paper.get("arxiv_url") or ""),
                    "url": str(paper.get("url") or paper.get("arxiv_url") or paper.get("pdf_url") or "").strip(),
                    "pdf_url": str(paper.get("pdf_url") or ""),
                    "relevance": str(paper.get("relevance") or paper.get("why_relevant") or "").strip(),
                    "novelty_risk": str(paper.get("novelty_risk") or "unknown").strip().lower(),
                }
            )
        return dedupe_notes(notes)[:max_papers]

class ReflectionAgent:

    def _seed_literature_for_review(
        self,
        hypothesis: Hypothesis,
        context: ContextMemory,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        if max_results <= 0:
            return []
        return dedupe_notes(hypothesis.literature_notes)[:max_results]

    def review_hypotheses(
        self,
        hypotheses: List[Hypothesis],
        research_goal: ResearchGoal,
        context: ContextMemory,
        step_name: str = "review",
    ) -> StepResult:
        start_time = time.time()
        research_plan = context.research_plan or ResearchPlan.from_goal(research_goal)
        meta_feedback = context.latest_meta_review()
        reference_paper = context.reference_paper_context or research_goal.reference_paper_context

        def review_one(hypothesis: Hypothesis) -> Tuple[Hypothesis, Dict[str, Any], List[Dict[str, Any]], str | None]:
            literature_notes = self._seed_literature_for_review(hypothesis, context, research_goal.max_literature_results)

            try:
                payload = _coerce_mapping(call_json(
                    build_unified_review_messages(
                        research_goal=research_goal.prompt_context(),
                        research_plan=research_plan.to_dict(),
                        reference_paper=reference_paper,
                        hypothesis=hypothesis.to_dict(),
                        literature_notes=literature_notes,
                        tournament_history=context.tournament_results + context.opencode_rerank_history,
                        meta_feedback=meta_feedback,
                    ),
                    model=research_goal.critic_llm_model,
                    temperature=research_goal.reflection_temperature,
                    profile="critic",
                ), self._fallback_review(hypothesis))
                return hypothesis, payload, literature_notes, None
            except LLMCallError as exc:
                return hypothesis, self._fallback_review(hypothesis), literature_notes, f"{hypothesis.hypothesis_id}: {exc}"

        errors: List[str] = []
        reviews = []
        literature_by_hypothesis: Dict[str, List[Dict[str, Any]]] = {}
        results = run_concurrently(hypotheses, review_one, max_workers=research_goal.max_concurrency)
        for hypothesis, payload, literature_notes, error in results:
            if error:
                errors.append(error)
            hypothesis.literature_notes = literature_notes
            literature_by_hypothesis[hypothesis.hypothesis_id] = literature_notes
            hypothesis.apply_review(payload, stage_label=step_name)
            hypothesis.review_artifacts.append(payload)
            reviews.append({"id": hypothesis.hypothesis_id, **payload})

        return StepResult(
            name=step_name,
            hypotheses=hypotheses,
            data={
                "reviews": reviews,
                "literature_by_hypothesis": literature_by_hypothesis,
                "reviewed_count": len(reviews),
                "eligible_count": len([hypothesis for hypothesis in hypotheses if hypothesis.is_active]),
            },
            errors=errors,
            duration=time.time() - start_time,
        )

    def _fallback_review(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        summary = hypothesis.text[:220]
        return {
            "alignment_score": 3,
            "novelty_score": 3,
            "plausibility_score": 3,
            "feasibility_score": 3,
            "correctness_score": 3,
            "testability_score": 3,
            "story_coherence_score": 3,
            "theoretical_depth_score": 3,
            "non_combination_score": 3,
            "verdict": "revise",
            "reject": False,
            "feasible": True,
            "short_summary": summary,
            "reflection": "Automatic review failed, so this hypothesis needs manual reflection before being trusted.",
            "feasibility_rationale": "No fatal feasibility issue was proven by the fallback path.",
            "reference_connection": "",
            "strengths": ["Fallback review used because the structured LLM review failed."],
            "weaknesses": ["Needs manual inspection because automatic review failed."],
            "critical_assumptions": hypothesis.key_assumptions[:3],
            "story_diagnosis": {
                "core_story": hypothesis.theoretical_story or hypothesis.mechanism,
                "combination_risk": "medium" if hypothesis.generation_strategy == "combination" else "low",
                "missing_theory": [],
                "repair_to_story": ["Retry review to diagnose whether the hypothesis has a coherent paper story."],
            },
            "supporting_observations": [],
            "contradicting_observations": [],
            "failure_modes": [],
            "validation_experiments": hypothesis.validation_experiments[:4],
            "improvement_actions": ["Retry review with more specific literature grounding."],
            "references": hypothesis.references[:5],
        }


class ProximityAgent:
    def build_proximity_graph(self, research_goal: ResearchGoal, context: ContextMemory, step_name: str) -> StepResult:
        start_time = time.time()
        active_hypotheses = context.get_active_hypotheses()
        adjacency: Dict[str, List[Dict[str, Any]]] = {}
        duplicate_candidates = []

        for hypothesis in active_hypotheses:
            adjacency.setdefault(hypothesis.hypothesis_id, [])

        for hypothesis_a, hypothesis_b in itertools.combinations(active_hypotheses, 2):
            similarity = similarity_score(hypothesis_a.text, hypothesis_b.text)
            adjacency.setdefault(hypothesis_a.hypothesis_id, []).append(
                {"other_id": hypothesis_b.hypothesis_id, "similarity": _round_score(similarity)}
            )
            adjacency.setdefault(hypothesis_b.hypothesis_id, []).append(
                {"other_id": hypothesis_a.hypothesis_id, "similarity": _round_score(similarity)}
            )
            if similarity >= 0.85:
                duplicate_candidates.append(
                    {
                        "hypothesis_a": hypothesis_a.hypothesis_id,
                        "hypothesis_b": hypothesis_b.hypothesis_id,
                        "similarity": _round_score(similarity),
                    }
                )

        threshold = research_goal.proximity_similarity_threshold
        visjs_data = generate_visjs_data(adjacency, threshold=threshold)
        clusters = self._build_clusters(adjacency, threshold)
        data = {
            "adjacency_graph": adjacency,
            "nodes": visjs_data["nodes"],
            "edges": visjs_data["edges"],
            "clusters": clusters,
            "duplicate_candidates": duplicate_candidates,
            "threshold": threshold,
        }
        return StepResult(
            name=step_name,
            hypotheses=active_hypotheses,
            data=data,
            duration=time.time() - start_time,
        )

    def _build_clusters(self, adjacency: Dict[str, List[Dict[str, Any]]], threshold: float) -> List[List[str]]:
        visited = set()
        clusters = []

        for node_id in adjacency:
            if node_id in visited:
                continue
            stack = [node_id]
            cluster = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(current)
                for neighbor in adjacency.get(current, []):
                    if neighbor.get("similarity", 0.0) >= threshold:
                        stack.append(neighbor.get("other_id"))
            clusters.append(sorted(cluster))
        return sorted(clusters, key=len, reverse=True)


class RankingAgent:
    def run_tournament(
        self,
        research_goal: ResearchGoal,
        context: ContextMemory,
        proximity_data: Dict[str, Any],
        step_name: str,
    ) -> StepResult:
        start_time = time.time()
        research_plan = context.research_plan or ResearchPlan.from_goal(research_goal)
        active_hypotheses = context.get_active_hypotheses()
        if len(active_hypotheses) < 2:
            return StepResult(name=step_name, hypotheses=active_hypotheses, data={"matches": []}, duration=time.time() - start_time)

        skipped_cached_pairs = self._count_cached_pairs(active_hypotheses, context)
        match_budget = self._match_budget(
            research_goal=research_goal,
            active_hypotheses=active_hypotheses,
            skipped_cached_pairs=skipped_cached_pairs,
            step_name=step_name,
        )
        pairs = self._select_pairs(
            active_hypotheses,
            proximity_data.get("adjacency_graph", {}),
            match_budget,
            research_goal.proximity_similarity_threshold,
            context,
        )
        top_ids = {hypothesis.hypothesis_id for hypothesis in sorted(active_hypotheses, key=lambda item: item.elo_score, reverse=True)[:4]}
        latest_iteration = max((hypothesis.created_in_iteration for hypothesis in active_hypotheses), default=0)
        fresh_ids = {
            hypothesis.hypothesis_id
            for hypothesis in active_hypotheses
            if latest_iteration > 0 and hypothesis.created_in_iteration == latest_iteration
        }
        uncached_full_debate_keys = []
        for hypothesis_a, hypothesis_b, _similarity in pairs:
            if self._get_cached_decision(context, hypothesis_a, hypothesis_b):
                continue
            uncached_full_debate_keys.append(self._pair_key(hypothesis_a, hypothesis_b))
            if len(uncached_full_debate_keys) >= 3:
                break
        uncached_full_debate_key_set = set(uncached_full_debate_keys)
        reaudit_keys = self._select_reaudit_keys(
            pairs=pairs,
            context=context,
            research_goal=research_goal,
            step_name=step_name,
            top_ids=top_ids,
            fresh_ids=fresh_ids,
        )
        errors: List[str] = []
        matches: List[Dict[str, Any]] = []

        def judge_pair(indexed_pair: Tuple[int, Tuple[Hypothesis, Hypothesis, float]]) -> Dict[str, Any]:
            index, (hypothesis_a, hypothesis_b, similarity) = indexed_pair
            pair_ids = {hypothesis_a.hypothesis_id, hypothesis_b.hypothesis_id}
            pair_key = self._pair_key(hypothesis_a, hypothesis_b)
            challenger_vs_frontier = bool(pair_ids & fresh_ids) and bool(pair_ids & top_ids)
            mode = (
                "full_debate"
                if pair_key in uncached_full_debate_key_set or pair_ids <= top_ids or challenger_vs_frontier
                else "quick_compare"
            )
            scheduled_because = (
                "fresh_challenger_vs_frontier"
                if challenger_vs_frontier
                else "frontier_debate"
                if pair_ids <= top_ids
                else "proximity_and_priority"
            )
            proximity_note = {"similarity": similarity, "scheduled_because": scheduled_because}
            cached = self._get_cached_decision(context, hypothesis_a, hypothesis_b)
            should_reaudit = cached is not None and pair_key in reaudit_keys
            if cached and not should_reaudit:
                payload = _coerce_mapping(cached.get("payload"), {})
                if not payload:
                    payload = {
                        "winner_id": cached.get("winner"),
                        "loser_id": cached.get("loser"),
                        "comparison_summary": cached.get("comparison_summary", ""),
                        "winner_reason": cached.get("winner_reason", ""),
                        "confidence": cached.get("confidence", 0.5),
                    }
                return {
                    "index": index,
                    "hypothesis_a": hypothesis_a,
                    "hypothesis_b": hypothesis_b,
                    "similarity": similarity,
                    "mode": mode,
                    "scheduled_because": "cached_replay",
                    "payload": payload,
                    "cached": True,
                    "counted_for_elo": False,
                    "adjudication_source": "cache_replay",
                    "error": None,
                }
            try:
                if should_reaudit:
                    messages = build_ranking_reaudit_messages(
                        research_goal=research_goal.prompt_context(),
                        research_plan=research_plan.to_dict(),
                        hypothesis_a=hypothesis_a.to_dict(),
                        hypothesis_b=hypothesis_b.to_dict(),
                        proximity_note=proximity_note,
                        previous_decision=cached,
                        mode=mode,
                    )
                    adjudication_source = "cache_reaudit"
                else:
                    messages = build_ranking_messages(
                        research_goal=research_goal.prompt_context(),
                        research_plan=research_plan.to_dict(),
                        hypothesis_a=hypothesis_a.to_dict(),
                        hypothesis_b=hypothesis_b.to_dict(),
                        proximity_note=proximity_note,
                        mode=mode,
                    )
                    adjudication_source = "new_llm"
                payload = _coerce_mapping(call_json(
                    messages,
                    model=research_goal.critic_llm_model,
                    temperature=max(0.1, research_goal.reflection_temperature - 0.2),
                    profile="critic",
                ), self._heuristic_decision(hypothesis_a, hypothesis_b, research_plan))
                error = None
            except LLMCallError as exc:
                payload = self._heuristic_decision(hypothesis_a, hypothesis_b, research_plan)
                error = f"{hypothesis_a.hypothesis_id} vs {hypothesis_b.hypothesis_id}: {exc}"
            return {
                "index": index,
                "hypothesis_a": hypothesis_a,
                "hypothesis_b": hypothesis_b,
                "similarity": similarity,
                "mode": mode,
                "scheduled_because": scheduled_because,
                "payload": payload,
                "cached": cached is not None,
                "counted_for_elo": True,
                "adjudication_source": adjudication_source,
                "error": error,
            }

        judged_pairs = run_concurrently(list(enumerate(pairs)), judge_pair, max_workers=research_goal.max_concurrency)
        judged_pairs.sort(key=lambda item: item["index"])

        for judged in judged_pairs:
            hypothesis_a = judged["hypothesis_a"]
            hypothesis_b = judged["hypothesis_b"]
            similarity = judged["similarity"]
            mode = judged["mode"]
            payload = judged["payload"]
            if judged["error"]:
                errors.append(judged["error"])

            winner, loser = self._resolve_winner(payload, hypothesis_a, hypothesis_b, research_plan)
            counted_for_elo = bool(judged.get("counted_for_elo"))
            if counted_for_elo:
                self._update_elo(winner, loser, research_goal.elo_k_factor)
            match_record = {
                "iteration": context.iteration_number + 1,
                "winner": winner.hypothesis_id,
                "loser": loser.hypothesis_id,
                "mode": mode,
                "cached": judged["cached"],
                "counted_for_elo": counted_for_elo,
                "adjudication_source": judged.get("adjudication_source", "new_llm"),
                "scheduled_because": judged["scheduled_because"],
                "similarity": similarity,
                "comparison_summary": payload.get("comparison_summary", ""),
                "winner_reason": payload.get("winner_reason", ""),
                "confidence": payload.get("confidence", 0.5),
                "winner_score_after": round(winner.elo_score, 2),
                "loser_score_after": round(loser.elo_score, 2),
            }
            if counted_for_elo:
                winner.record_debate_outcome(match_record)
                loser.record_debate_outcome(match_record)
                context.tournament_results.append(match_record)
                self._store_pairwise_decision(context, hypothesis_a, hypothesis_b, payload, match_record)
            matches.append(match_record)

        ranked = sorted(active_hypotheses, key=lambda hypothesis: hypothesis.elo_score, reverse=True)
        cached_match_count = sum(1 for match in matches if match.get("cached"))
        new_match_count = len(matches) - cached_match_count
        reaudit_match_count = sum(1 for match in matches if match.get("adjudication_source") == "cache_reaudit")
        elo_counted_match_count = sum(1 for match in matches if match.get("counted_for_elo"))
        fresh_match_count = sum(
            1
            for match in matches
            if match.get("winner") in fresh_ids or match.get("loser") in fresh_ids
        )
        fresh_new_match_count = sum(
            1
            for match in matches
            if not match.get("cached") and (match.get("winner") in fresh_ids or match.get("loser") in fresh_ids)
        )
        return StepResult(
            name=step_name,
            hypotheses=ranked,
            data={
                "matches": matches,
                "pairs_considered": len(pairs),
                "cached_pairs_considered": cached_match_count,
                "new_pairs_considered": new_match_count,
                "reaudited_pairs_considered": reaudit_match_count,
                "elo_counted_matches": elo_counted_match_count,
                "fresh_hypothesis_ids": sorted(fresh_ids),
                "fresh_pairs_considered": fresh_match_count,
                "fresh_new_pairs_considered": fresh_new_match_count,
                "match_budget": match_budget,
                "skipped_cached_pairs": skipped_cached_pairs,
            },
            errors=errors,
            duration=time.time() - start_time,
        )

    def _match_budget(
        self,
        research_goal: ResearchGoal,
        active_hypotheses: Sequence[Hypothesis],
        skipped_cached_pairs: int,
        step_name: str,
    ) -> int:
        return max(1, research_goal.ranking_matches_per_cycle)

    def _should_reaudit_pair(
        self,
        cached: Dict[str, Any],
        hypothesis_a: Hypothesis,
        hypothesis_b: Hypothesis,
        top_ids: set[str],
        fresh_ids: set[str],
        current_iteration: int,
    ) -> bool:
        created_iteration = int(cached.get("created_iteration") or 0)
        if created_iteration >= current_iteration:
            return False

        pair_ids = {hypothesis_a.hypothesis_id, hypothesis_b.hypothesis_id}
        confidence = float(cached.get("confidence", 0.5) or 0.5)
        age = current_iteration - created_iteration
        return pair_ids <= top_ids or bool(pair_ids & fresh_ids) or confidence < 0.55 or age >= 2

    def _select_reaudit_keys(
        self,
        pairs: Sequence[Tuple[Hypothesis, Hypothesis, float]],
        context: ContextMemory,
        research_goal: ResearchGoal,
        step_name: str,
        top_ids: set[str],
        fresh_ids: set[str],
    ) -> set[str]:
        current_iteration = context.iteration_number + 1
        cached_candidates: List[Tuple[str, Hypothesis, Hypothesis, Dict[str, Any], float]] = []
        for hypothesis_a, hypothesis_b, similarity in pairs:
            cached = self._get_cached_decision(context, hypothesis_a, hypothesis_b)
            if not cached:
                continue
            created_iteration = int(cached.get("created_iteration") or 0)
            if created_iteration >= current_iteration:
                continue
            cached_candidates.append((self._pair_key(hypothesis_a, hypothesis_b), hypothesis_a, hypothesis_b, cached, similarity))

        def priority(candidate: Tuple[str, Hypothesis, Hypothesis, Dict[str, Any], float]) -> Tuple[int, float, str]:
            key, hypothesis_a, hypothesis_b, cached, similarity = candidate
            pair_ids = {hypothesis_a.hypothesis_id, hypothesis_b.hypothesis_id}
            confidence = float(cached.get("confidence", 0.5) or 0.5)
            age = current_iteration - int(cached.get("created_iteration") or 0)
            score = 0
            if pair_ids <= top_ids:
                score += 8
            if pair_ids & fresh_ids:
                score += 6
            if confidence < 0.55:
                score += 4
            if age >= 2:
                score += 2
            return (score, similarity, key)

        selected: set[str] = set()
        priority_candidates = [
            candidate
            for candidate in cached_candidates
            if self._should_reaudit_pair(candidate[3], candidate[1], candidate[2], top_ids, fresh_ids, current_iteration)
        ]
        for key, _hypothesis_a, _hypothesis_b, _cached, _similarity in sorted(priority_candidates, key=priority, reverse=True)[:2]:
            selected.add(key)

        random_limit = min(1, math.ceil(len(cached_candidates) * 0.05)) if cached_candidates else 0
        if random_limit <= 0:
            return selected

        random_candidates = [
            candidate
            for candidate in cached_candidates
            if candidate[0] not in selected
            and {candidate[1].hypothesis_id, candidate[2].hypothesis_id} - top_ids
            and not ({candidate[1].hypothesis_id, candidate[2].hypothesis_id} & fresh_ids)
        ]
        seed_prefix = f"{research_goal.signature()}::{current_iteration}::{step_name}"

        def stable_random_key(candidate: Tuple[str, Hypothesis, Hypothesis, Dict[str, Any], float]) -> str:
            return hashlib.sha256(f"{seed_prefix}::{candidate[0]}".encode("utf-8")).hexdigest()

        for key, _hypothesis_a, _hypothesis_b, _cached, _similarity in sorted(random_candidates, key=stable_random_key)[:random_limit]:
            selected.add(key)
        return selected

    def _select_pairs(
        self,
        hypotheses: List[Hypothesis],
        adjacency_graph: Dict[str, List[Dict[str, Any]]],
        max_matches: int,
        similarity_threshold: float,
        context: ContextMemory,
    ) -> List[Tuple[Hypothesis, Hypothesis, float]]:
        by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in hypotheses}
        cached_pairs: List[Tuple[Hypothesis, Hypothesis, float]] = []
        new_pairs: List[Tuple[Hypothesis, Hypothesis, float]] = []
        seen_similar_pairs = set()
        scheduled_pairs = set()
        similarity_by_key: Dict[Tuple[str, str], float] = {}

        similar_pairs = []
        for source_id, neighbors in adjacency_graph.items():
            for neighbor in neighbors:
                other_id = str(neighbor.get("other_id") or "").strip()
                if not other_id:
                    continue
                pair_key = tuple(sorted([source_id, other_id]))
                if len(pair_key) != 2 or pair_key in seen_similar_pairs:
                    continue
                seen_similar_pairs.add(pair_key)
                similarity = float(neighbor.get("similarity", 0.0))
                if pair_key[0] in by_id and pair_key[1] in by_id:
                    similarity_by_key[pair_key] = max(similarity_by_key.get(pair_key, 0.0), similarity)
                if similarity >= similarity_threshold:
                    similar_pairs.append((pair_key[0], pair_key[1], similarity))
        similar_pairs.sort(key=lambda item: item[2], reverse=True)

        ranked = sorted(hypotheses, key=lambda item: item.elo_score, reverse=True)
        latest_iteration = max((hypothesis.created_in_iteration for hypothesis in hypotheses), default=0)
        fresh_hypotheses = [
            hypothesis
            for hypothesis in hypotheses
            if latest_iteration > 0 and hypothesis.created_in_iteration == latest_iteration
        ]
        fresh_hypotheses.sort(
            key=lambda hypothesis: (
                0 if hypothesis.origin == "evolution" else 1,
                -hypothesis.scores.get("novelty", 0.0),
                -hypothesis.elo_score,
                hypothesis.hypothesis_id,
            )
        )
        fresh_ids = {hypothesis.hypothesis_id for hypothesis in fresh_hypotheses}
        incumbents = [hypothesis for hypothesis in ranked if hypothesis.hypothesis_id not in fresh_ids]
        frontier = incumbents if incumbents else ranked
        top_frontier = frontier[: min(6, len(frontier))]

        for hypothesis_a, hypothesis_b in itertools.combinations(hypotheses, 2):
            if not self._get_cached_decision(context, hypothesis_a, hypothesis_b):
                continue
            key = tuple(sorted([hypothesis_a.hypothesis_id, hypothesis_b.hypothesis_id]))
            scheduled_pairs.add(key)
            cached_pairs.append(
                (
                    hypothesis_a,
                    hypothesis_b,
                    _round_score(similarity_by_key.get(key, 0.0)),
                )
            )

        def add_pair(left_id: str, right_id: str, similarity: float) -> None:
            key = tuple(sorted([left_id, right_id]))
            if left_id == right_id or key in scheduled_pairs:
                return
            if left_id not in by_id or right_id not in by_id:
                return
            if self._get_cached_decision(context, by_id[left_id], by_id[right_id]):
                return
            scheduled_pairs.add(key)
            new_pairs.append((by_id[left_id], by_id[right_id], _round_score(similarity)))

        def add_round_robin_challenger_pairs(opponent_groups: List[List[Hypothesis]]) -> None:
            max_rounds = max((len(group) for group in opponent_groups), default=0)
            if max_rounds <= 0:
                return

            def challenger_match_count(challenger: Hypothesis) -> int:
                return sum(
                    1
                    for hypothesis_a, hypothesis_b, _similarity in new_pairs
                    if challenger.hypothesis_id in {hypothesis_a.hypothesis_id, hypothesis_b.hypothesis_id}
                )

            for round_index in range(max_rounds):
                ordered_groups = sorted(
                    enumerate(zip(fresh_hypotheses, opponent_groups)),
                    key=lambda item: (challenger_match_count(item[1][0]), item[0]),
                )
                for _group_index, (challenger, opponents) in ordered_groups:
                    if round_index >= len(opponents):
                        continue
                    opponent = opponents[round_index]
                    add_pair(
                        challenger.hypothesis_id,
                        opponent.hypothesis_id,
                        similarity_by_key.get(tuple(sorted([challenger.hypothesis_id, opponent.hypothesis_id])), 0.0),
                    )
                    if len(new_pairs) >= max_matches:
                        return

        if fresh_hypotheses:
            parent_groups = [
                [
                    by_id[parent_id]
                    for parent_id in hypothesis.parent_ids
                    if parent_id in by_id and parent_id != hypothesis.hypothesis_id
                ]
                for hypothesis in fresh_hypotheses
            ]
            add_round_robin_challenger_pairs(parent_groups)
            if len(new_pairs) >= max_matches:
                return new_pairs[:max_matches] + cached_pairs

            frontier_groups = [
                [candidate for candidate in top_frontier if candidate.hypothesis_id != hypothesis.hypothesis_id]
                for hypothesis in fresh_hypotheses
            ]
            add_round_robin_challenger_pairs(frontier_groups)
            if len(new_pairs) >= max_matches:
                return new_pairs[:max_matches] + cached_pairs

            similar_groups: List[List[Hypothesis]] = []
            for hypothesis in fresh_hypotheses:
                neighbors: List[Hypothesis] = []
                for left_id, right_id, similarity in similar_pairs:
                    if similarity < similarity_threshold:
                        continue
                    other_id = right_id if left_id == hypothesis.hypothesis_id else left_id if right_id == hypothesis.hypothesis_id else ""
                    if other_id in by_id and other_id != hypothesis.hypothesis_id:
                        neighbors.append(by_id[other_id])
                similar_groups.append(neighbors)
            add_round_robin_challenger_pairs(similar_groups)
            if len(new_pairs) >= max_matches:
                return new_pairs[:max_matches] + cached_pairs

            benchmark_groups: List[List[Hypothesis]] = []
            if frontier:
                benchmark_candidates = dedupe_preserve_order(
                    [
                        frontier[len(frontier) // 2].hypothesis_id,
                        frontier[-1].hypothesis_id,
                    ]
                )
                benchmarks = [by_id[candidate_id] for candidate_id in benchmark_candidates if candidate_id in by_id]
                benchmark_groups = [
                    [candidate for candidate in benchmarks if candidate.hypothesis_id != hypothesis.hypothesis_id]
                    for hypothesis in fresh_hypotheses
                ]
            add_round_robin_challenger_pairs(benchmark_groups)
            if len(new_pairs) >= max_matches:
                return new_pairs[:max_matches] + cached_pairs

            for first, second in itertools.combinations(fresh_hypotheses, 2):
                add_pair(
                    first.hypothesis_id,
                    second.hypothesis_id,
                    similarity_by_key.get(tuple(sorted([first.hypothesis_id, second.hypothesis_id])), 0.0),
                )
                if len(new_pairs) >= max_matches:
                    return new_pairs[:max_matches] + cached_pairs

        for left_id, right_id, similarity in similar_pairs:
            add_pair(left_id, right_id, similarity)
            if len(new_pairs) >= max_matches:
                return new_pairs[:max_matches] + cached_pairs

        for first, second in itertools.combinations(ranked[:5], 2):
            add_pair(first.hypothesis_id, second.hypothesis_id, 0.0)
            if len(new_pairs) >= max_matches:
                return new_pairs[:max_matches] + cached_pairs

        for new_hypothesis in fresh_hypotheses:
            for incumbent in ranked[:4]:
                add_pair(new_hypothesis.hypothesis_id, incumbent.hypothesis_id, 0.0)
                if len(new_pairs) >= max_matches:
                    return new_pairs[:max_matches] + cached_pairs

        remaining_pairs = []
        for source_id, neighbors in adjacency_graph.items():
            for neighbor in neighbors:
                other_id = str(neighbor.get("other_id") or "").strip()
                if not other_id:
                    continue
                pair_key = tuple(sorted([source_id, other_id]))
                if len(pair_key) != 2:
                    continue
                remaining_pairs.append((pair_key[0], pair_key[1], float(neighbor.get("similarity", 0.0))))
        remaining_pairs.sort(key=lambda item: item[2], reverse=True)
        for left_id, right_id, similarity in remaining_pairs:
            add_pair(left_id, right_id, similarity)
            if len(new_pairs) >= max_matches:
                return new_pairs[:max_matches] + cached_pairs

        for first, second in itertools.combinations(hypotheses, 2):
            add_pair(first.hypothesis_id, second.hypothesis_id, 0.0)
            if len(new_pairs) >= max_matches:
                return new_pairs[:max_matches] + cached_pairs

        return new_pairs[:max_matches] + cached_pairs

    def _pair_key(self, hypothesis_a: Hypothesis, hypothesis_b: Hypothesis) -> str:
        return "::".join(sorted([hypothesis_a.hypothesis_id, hypothesis_b.hypothesis_id]))

    def _pair_state_signature(self, hypothesis_a: Hypothesis, hypothesis_b: Hypothesis) -> str:
        payload = {
            hypothesis_a.hypothesis_id: hypothesis_a.comparison_signature(),
            hypothesis_b.hypothesis_id: hypothesis_b.comparison_signature(),
        }
        return json.dumps(payload, sort_keys=True)

    def _get_cached_decision(
        self,
        context: ContextMemory,
        hypothesis_a: Hypothesis,
        hypothesis_b: Hypothesis,
    ) -> Dict[str, Any] | None:
        cached = context.pairwise_decisions.get(self._pair_key(hypothesis_a, hypothesis_b))
        if not cached:
            return None
        if cached.get("state_signature") != self._pair_state_signature(hypothesis_a, hypothesis_b):
            return None
        return cached

    def _store_pairwise_decision(
        self,
        context: ContextMemory,
        hypothesis_a: Hypothesis,
        hypothesis_b: Hypothesis,
        payload: Dict[str, Any],
        match_record: Dict[str, Any],
    ) -> None:
        context.pairwise_decisions[self._pair_key(hypothesis_a, hypothesis_b)] = {
            "state_signature": self._pair_state_signature(hypothesis_a, hypothesis_b),
            "hypothesis_ids": sorted([hypothesis_a.hypothesis_id, hypothesis_b.hypothesis_id]),
            "payload": payload,
            "winner": match_record.get("winner"),
            "loser": match_record.get("loser"),
            "comparison_summary": match_record.get("comparison_summary", ""),
            "winner_reason": match_record.get("winner_reason", ""),
            "confidence": match_record.get("confidence", 0.5),
            "created_iteration": match_record.get("iteration"),
        }

    def _count_cached_pairs(self, hypotheses: List[Hypothesis], context: ContextMemory) -> int:
        count = 0
        for hypothesis_a, hypothesis_b in itertools.combinations(hypotheses, 2):
            if self._get_cached_decision(context, hypothesis_a, hypothesis_b):
                count += 1
        return count

    def _weighted_score(self, hypothesis: Hypothesis, research_plan: ResearchPlan) -> float:
        weights = research_plan.evaluation_criteria
        score_map = {
            "alignment": hypothesis.scores.get("alignment", hypothesis.scores.get("correctness", 0.0)),
            "novelty": hypothesis.scores.get("novelty", 0.0),
            "plausibility": hypothesis.scores.get("plausibility", hypothesis.scores.get("feasibility", 0.0)),
            "testability": hypothesis.scores.get("testability", 0.0),
        }
        weighted = sum(score_map.get(metric, 0.0) * weight for metric, weight in weights.items())
        story_scores = [
            hypothesis.scores.get("story_coherence", 0.0),
            hypothesis.scores.get("theoretical_depth", 0.0),
            hypothesis.scores.get("non_combination", 0.0),
        ]
        story_bonus = (sum(score for score in story_scores if score) / (5.0 * len(story_scores))) * 0.75 if any(story_scores) else 0.0
        return weighted + story_bonus

    def _heuristic_decision(self, hypothesis_a: Hypothesis, hypothesis_b: Hypothesis, research_plan: ResearchPlan) -> Dict[str, Any]:
        score_a = self._weighted_score(hypothesis_a, research_plan)
        score_b = self._weighted_score(hypothesis_b, research_plan)
        winner = hypothesis_a if score_a >= score_b else hypothesis_b
        loser = hypothesis_b if winner is hypothesis_a else hypothesis_a
        return {
            "winner_id": winner.hypothesis_id,
            "loser_id": loser.hypothesis_id,
            "comparison_summary": "Heuristic fallback used because pairwise LLM ranking failed.",
            "winner_reason": "The winner had the stronger weighted review profile under the shared evaluation criteria.",
            "novelty_edge": "A" if hypothesis_a.scores.get("novelty", 0) > hypothesis_b.scores.get("novelty", 0) else "B",
            "plausibility_edge": "A" if hypothesis_a.scores.get("plausibility", 0) > hypothesis_b.scores.get("plausibility", 0) else "B",
            "testability_edge": "A" if hypothesis_a.scores.get("testability", 0) > hypothesis_b.scores.get("testability", 0) else "B",
            "confidence": 0.35,
        }

    def _resolve_winner(
        self,
        payload: Dict[str, Any],
        hypothesis_a: Hypothesis,
        hypothesis_b: Hypothesis,
        research_plan: ResearchPlan,
    ) -> Tuple[Hypothesis, Hypothesis]:
        winner_id = str(payload.get("winner_id", "")).strip()
        loser_id = str(payload.get("loser_id", "")).strip()
        by_id = {
            hypothesis_a.hypothesis_id: hypothesis_a,
            hypothesis_b.hypothesis_id: hypothesis_b,
        }
        winner = by_id.get(winner_id)
        loser = by_id.get(loser_id)
        if winner and loser and winner is not loser:
            return winner, loser

        fallback = self._heuristic_decision(hypothesis_a, hypothesis_b, research_plan)
        return self._resolve_winner(fallback, hypothesis_a, hypothesis_b, research_plan)

    def _update_elo(self, winner: Hypothesis, loser: Hypothesis, k_factor: int) -> None:
        expected_winner = 1 / (1 + math.pow(10, (loser.elo_score - winner.elo_score) / 400))
        expected_loser = 1 - expected_winner
        winner.elo_score += k_factor * (1 - expected_winner)
        loser.elo_score += k_factor * (0 - expected_loser)


class OpenCodeRerankingAgent:
    """Use parallel OpenCode CLI reviews to rerank the current top-four ideas."""

    def rerank_top_hypotheses(
        self,
        research_goal: ResearchGoal,
        context: ContextMemory,
        step_name: str = "opencode_reranking",
        top_n: int = 4,
    ) -> StepResult:
        start_time = time.time()
        active = [hypothesis for hypothesis in context.get_ranked_hypotheses() if hypothesis.is_active]
        candidates = active[:top_n]
        if len(candidates) < 2:
            return StepResult(
                name=step_name,
                hypotheses=candidates,
                data={"status": "skipped", "reason": "fewer_than_two_active_hypotheses"},
                duration=time.time() - start_time,
            )
        enabled = config.get("enable_opencode_reranking")
        if enabled is None:
            enabled = config.get("enable_codex_reranking", True)
        if not bool(enabled):
            return StepResult(
                name=step_name,
                hypotheses=candidates,
                data={"status": "disabled"},
                duration=time.time() - start_time,
            )

        timeout_value = config.get(
            "opencode_rerank_timeout_seconds",
            config.get("codex_rerank_timeout_seconds", 900),
        )
        timeout = int(timeout_value or 900)
        errors: List[str] = []
        raw_reviews = run_concurrently(
            candidates,
            lambda hypothesis: self._review_one_candidate(research_goal, context, candidates, hypothesis, timeout),
            max_workers=min(len(candidates), research_goal.max_concurrency),
        )
        for review in raw_reviews:
            errors.extend(review.get("errors", []))

        normalized = self._normalize_parallel_reviews(raw_reviews, candidates)
        if any(review.get("payload") for review in raw_reviews) and normalized["ranking"]:
            self._apply_ranking(candidates, normalized)
            status = "applied"
        else:
            status = "fallback_current_order"

        record = {
            "iteration": context.iteration_number + 1,
            "step": step_name,
            "status": status,
            "ranking": normalized["ranking"],
            "evaluations": normalized["evaluations"],
            "overall_rationale": self._parallel_overall_rationale(normalized["ranking"], normalized["evaluations"]),
            "parallel_reviews": raw_reviews,
        }
        context.opencode_rerank_history.append(record)

        ranked_candidates = sorted(candidates, key=lambda hypothesis: hypothesis.elo_score, reverse=True)
        return StepResult(
            name=step_name,
            hypotheses=ranked_candidates,
            data={
                **record,
                "command": "opencode run --dangerously-skip-permissions",
                "candidate_ids": [hypothesis.hypothesis_id for hypothesis in candidates],
                "parallel_review_count": len(raw_reviews),
            },
            errors=errors,
            duration=time.time() - start_time,
        )

    def _review_one_candidate(
        self,
        research_goal: ResearchGoal,
        context: ContextMemory,
        candidates: Sequence[Hypothesis],
        candidate: Hypothesis,
        timeout: int,
    ) -> Dict[str, Any]:
        prompt = self._build_candidate_prompt(research_goal, context, candidates, candidate)
        raw_stdout = ""
        raw_stderr = ""
        payload: Dict[str, Any] = {}
        errors: List[str] = []
        try:
            completed = _run_opencode(prompt, timeout)
            raw_stdout = completed.stdout or ""
            raw_stderr = completed.stderr or ""
            if completed.returncode != 0:
                errors.append(
                    f"{candidate.hypothesis_id}: opencode run exited with {completed.returncode}: "
                    f"{(raw_stderr or raw_stdout)[:1200]}"
                )
            else:
                payload = self._parse_json_payload(raw_stdout)
                if not payload:
                    errors.append(f"{candidate.hypothesis_id}: opencode run returned no parseable JSON payload.")
        except subprocess.TimeoutExpired as exc:
            errors.append(f"{candidate.hypothesis_id}: opencode run timed out after {timeout}s: {exc}")
        except OSError as exc:
            errors.append(f"{candidate.hypothesis_id}: could not run opencode run: {exc}")

        return {
            "id": candidate.hypothesis_id,
            "payload": payload,
            "errors": errors,
            "stdout_tail": raw_stdout[-2000:],
            "stderr_tail": raw_stderr[-2000:],
        }

    def _build_candidate_prompt(
        self,
        research_goal: ResearchGoal,
        context: ContextMemory,
        candidates: Sequence[Hypothesis],
        candidate: Hypothesis,
    ) -> str:
        research_plan = context.research_plan or ResearchPlan.from_goal(research_goal)
        competing_candidates = [
            {
                "id": hypothesis.hypothesis_id,
                "title": hypothesis.title,
                "elo_score": hypothesis.elo_score,
                "review_summary": hypothesis.review_summary,
                "scores": hypothesis.scores,
                "latest_review": hypothesis.review_artifacts[-1] if hypothesis.review_artifacts else {},
                "latest_opencode_rerank": hypothesis.opencode_rerank_reviews[-1] if hypothesis.opencode_rerank_reviews else {},
            }
            for hypothesis in candidates
            if hypothesis.hypothesis_id != candidate.hypothesis_id
        ]
        payload = {
            "task": "Independently review one candidate from the current top-four research ideas.",
            "instructions": [
                "Use only the included per-idea literature_notes. Do not run another paper search.",
                "Score this candidate against the user's goal, the research plan, the required reference paper, and the competing candidates.",
                "Return a calibrated standalone score so local code can merge four parallel reviews into one ranking.",
                "Prefer ideas that are inspired by the reference paper but do not copy its exact contribution.",
                "Return strict JSON only, with no markdown or commentary outside JSON.",
            ],
            "research_goal": research_goal.prompt_context(),
            "research_plan": research_plan.to_dict(),
            "reference_paper": context.reference_paper_context or research_goal.reference_paper_context,
            "candidate": candidate.to_prompt_dict(),
            "competing_candidates": competing_candidates,
            "required_schema": {
                "id": candidate.hypothesis_id,
                "overall_score": 0.0,
                "keep": True,
                "summary": "short judgment",
                "strengths": ["string"],
                "weaknesses": ["string"],
                "novelty_risks": ["string"],
                "feasibility_risks": ["string"],
                "literature_notes": ["string"],
                "reference_connection": "string",
                "recommended_action": "keep|revise|drop",
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _parse_json_payload(self, text: str) -> Dict[str, Any]:
        stripped = text.strip()
        if not stripped:
            return {}
        try:
            payload = json.loads(stripped)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            pass

        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
        if code_block:
            try:
                payload = json.loads(code_block.group(1))
                return payload if isinstance(payload, dict) else {}
            except json.JSONDecodeError:
                pass

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(stripped[start : end + 1])
                return payload if isinstance(payload, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    def _normalize_parallel_reviews(
        self,
        raw_reviews: Sequence[Dict[str, Any]],
        candidates: Sequence[Hypothesis],
    ) -> Dict[str, Any]:
        by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in candidates}
        evaluations: Dict[str, Dict[str, Any]] = {}
        scored: List[Tuple[float, float, str]] = []

        for review in raw_reviews:
            hypothesis_id = str(review.get("id") or "").strip()
            if hypothesis_id not in by_id:
                continue
            payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
            evaluation = self._normalize_candidate_evaluation(payload, hypothesis_id)
            evaluations[hypothesis_id] = evaluation
            scored.append(
                (
                    evaluation["overall_score"],
                    by_id[hypothesis_id].elo_score,
                    hypothesis_id,
                )
            )

        for hypothesis in candidates:
            if hypothesis.hypothesis_id not in evaluations:
                evaluations[hypothesis.hypothesis_id] = self._fallback_candidate_evaluation(hypothesis)
                scored.append((0.0, hypothesis.elo_score, hypothesis.hypothesis_id))

        ranking = []
        for rank, (_score, _elo, hypothesis_id) in enumerate(sorted(scored, reverse=True), start=1):
            evaluation = evaluations[hypothesis_id]
            ranking.append(
                {
                    "id": hypothesis_id,
                    "rank": rank,
                    "overall_score": evaluation["overall_score"],
                    "keep": evaluation["keep"],
                    "summary": evaluation["summary"],
                }
            )
        return {"ranking": ranking, "evaluations": evaluations}

    def _normalize_candidate_evaluation(self, payload: Dict[str, Any], expected_id: str) -> Dict[str, Any]:
        hypothesis_id = str(payload.get("id") or payload.get("hypothesis_id") or expected_id).strip()
        if hypothesis_id != expected_id:
            hypothesis_id = expected_id
        recommended_action = str(payload.get("recommended_action") or "revise").strip().lower()
        keep = payload.get("keep")
        if keep is None:
            keep = recommended_action != "drop"
        return {
            "id": hypothesis_id,
            "summary": str(payload.get("summary") or "").strip(),
            "strengths": coerce_string_list(payload.get("strengths", [])),
            "weaknesses": coerce_string_list(payload.get("weaknesses", [])),
            "novelty_risks": coerce_string_list(payload.get("novelty_risks", [])),
            "feasibility_risks": coerce_string_list(payload.get("feasibility_risks", [])),
            "literature_notes": coerce_string_list(payload.get("literature_notes", [])),
            "reference_connection": str(payload.get("reference_connection") or "").strip(),
            "recommended_action": recommended_action,
            "overall_score": self._coerce_float(payload.get("overall_score")),
            "keep": bool(keep),
        }

    def _fallback_candidate_evaluation(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        return {
            "id": hypothesis.hypothesis_id,
            "summary": "",
            "strengths": [],
            "weaknesses": [],
            "novelty_risks": [],
            "feasibility_risks": [],
            "literature_notes": [],
            "reference_connection": "",
            "recommended_action": "revise",
            "overall_score": 0.0,
            "keep": True,
        }

    def _parallel_overall_rationale(
        self,
        ranking: Sequence[Dict[str, Any]],
        evaluations: Dict[str, Dict[str, Any]],
    ) -> str:
        if not ranking:
            return ""
        top_id = str(ranking[0].get("id") or "")
        summary = str(evaluations.get(top_id, {}).get("summary") or "").strip()
        if summary:
            return f"{top_id} ranked first by parallel OpenCode review: {summary}"
        return f"{top_id} ranked first by parallel OpenCode review."

    def _normalize_payload(self, payload: Dict[str, Any], candidates: Sequence[Hypothesis]) -> Dict[str, Any]:
        candidate_ids = [hypothesis.hypothesis_id for hypothesis in candidates]
        seen: set[str] = set()
        ranking: List[Dict[str, Any]] = []
        raw_ranking = payload.get("ranking") if isinstance(payload, dict) else []
        if isinstance(raw_ranking, list):
            for item in raw_ranking:
                hypothesis_id = ""
                if isinstance(item, dict):
                    hypothesis_id = str(item.get("id") or item.get("hypothesis_id") or "").strip()
                elif item:
                    hypothesis_id = str(item).strip()
                if hypothesis_id not in candidate_ids or hypothesis_id in seen:
                    continue
                seen.add(hypothesis_id)
                ranking.append(
                    {
                        "id": hypothesis_id,
                        "rank": len(ranking) + 1,
                        "overall_score": self._coerce_float(item.get("overall_score") if isinstance(item, dict) else None),
                        "keep": bool(item.get("keep", True)) if isinstance(item, dict) else True,
                        "summary": str(item.get("summary") or "").strip() if isinstance(item, dict) else "",
                    }
                )
        for hypothesis_id in candidate_ids:
            if hypothesis_id not in seen:
                ranking.append({"id": hypothesis_id, "rank": len(ranking) + 1, "overall_score": 0.0, "keep": True, "summary": ""})

        raw_evaluations = payload.get("evaluations") if isinstance(payload, dict) else {}
        evaluations: Dict[str, Dict[str, Any]] = {}
        for hypothesis_id in candidate_ids:
            evaluation = raw_evaluations.get(hypothesis_id, {}) if isinstance(raw_evaluations, dict) else {}
            if not isinstance(evaluation, dict):
                evaluation = {}
            evaluations[hypothesis_id] = {
                "summary": str(evaluation.get("summary") or "").strip(),
                "strengths": coerce_string_list(evaluation.get("strengths", [])),
                "weaknesses": coerce_string_list(evaluation.get("weaknesses", [])),
                "novelty_risks": coerce_string_list(evaluation.get("novelty_risks", [])),
                "feasibility_risks": coerce_string_list(evaluation.get("feasibility_risks", [])),
                "literature_notes": coerce_string_list(evaluation.get("literature_notes", [])),
                "reference_connection": str(evaluation.get("reference_connection") or "").strip(),
                "recommended_action": str(evaluation.get("recommended_action") or "revise").strip().lower(),
            }
        return {"ranking": ranking, "evaluations": evaluations}

    def _coerce_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _apply_ranking(self, candidates: Sequence[Hypothesis], normalized: Dict[str, Any]) -> None:
        by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in candidates}
        original_elo_values = sorted([hypothesis.elo_score for hypothesis in candidates], reverse=True)
        ranking = normalized["ranking"]
        evaluations = normalized["evaluations"]
        for index, item in enumerate(ranking):
            hypothesis = by_id.get(item["id"])
            if hypothesis is None:
                continue
            if index < len(original_elo_values):
                hypothesis.elo_score = original_elo_values[index]
            evaluation = {
                "rank": index + 1,
                "overall_score": item.get("overall_score", 0.0),
                "keep": item.get("keep", True),
                **evaluations.get(hypothesis.hypothesis_id, {}),
            }
            if item.get("summary") and not evaluation.get("summary"):
                evaluation["summary"] = item["summary"]
            hypothesis.opencode_rerank_reviews.append(evaluation)
            hypothesis.review_comments = dedupe_preserve_order(
                hypothesis.review_comments
                + coerce_string_list(evaluation.get("weaknesses", []))
                + coerce_string_list(evaluation.get("novelty_risks", []))
                + coerce_string_list(evaluation.get("feasibility_risks", []))
            )
            if evaluation.get("summary"):
                hypothesis.review_comments = dedupe_preserve_order(
                    hypothesis.review_comments + [f"opencode_reranking: {evaluation['summary']}"]
                )


class EvolutionAgent:
    def evolve_hypotheses(self, research_goal: ResearchGoal, context: ContextMemory) -> StepResult:
        start_time = time.time()
        research_plan = context.research_plan or ResearchPlan.from_goal(research_goal)
        ranked = [hypothesis for hypothesis in context.get_ranked_hypotheses() if hypothesis.is_active]
        coverage = _focus_area_guidance(research_plan, context)
        trajectory_state = _trajectory_state_snapshot(research_plan, context, coverage)
        diversity_snapshot = trajectory_state["frontier_diversity"]
        preferred_focus_areas = dedupe_preserve_order(
            coverage["uncovered_focus_areas"] + coerce_string_list(trajectory_state.get("frontier_focus_gaps", []))
        )
        source_budget, source_budget_adjustments = _evolution_source_budget_for_cycle(
            research_goal,
            len(ranked),
            trajectory_state,
        )
        source_hypotheses = _select_evolution_sources(
            ranked,
            source_budget,
            preferred_focus_areas=preferred_focus_areas,
            penalized_focus_areas=coverage["overexplored_areas"],
        )
        if len(source_hypotheses) < 2:
            return StepResult(
                name="evolution",
                hypotheses=[],
                data={
                    "source_hypotheses": [],
                    "trajectory_state": trajectory_state,
                    "adaptive_tuning": {
                        "base_source_budget": max(3, research_goal.top_k_hypotheses + 1),
                        "source_budget": source_budget,
                        "source_budget_adjustments": source_budget_adjustments,
                    },
                },
                duration=time.time() - start_time,
            )
        literature_notes = dedupe_notes(
            note
            for hypothesis in source_hypotheses
            for note in hypothesis.literature_notes
        )[: research_goal.max_literature_results]
        meta_feedback = context.latest_meta_review()
        research_overview = context.latest_research_overview()
        errors: List[str] = []
        evolution_temperature, temperature_adjustments = _evolution_temperature_for_cycle(
            research_goal,
            trajectory_state,
        )
        target_evolved_hypotheses = max(2, min(research_goal.top_k_hypotheses, 5))

        try:
            payload = call_json(
                build_evolution_messages(
                    research_goal=research_goal.prompt_context(),
                    research_plan=research_plan.to_dict(),
                    top_hypotheses=[hypothesis.compact_summary() for hypothesis in source_hypotheses],
                    literature_notes=literature_notes,
                    meta_feedback=meta_feedback,
                    research_overview=research_overview,
                    trajectory_state=trajectory_state,
                    num_hypotheses=target_evolved_hypotheses,
                    uncovered_focus_areas=coverage["uncovered_focus_areas"],
                    overexplored_areas=coverage["overexplored_areas"],
                ),
                model=research_goal.llm_model,
                temperature=evolution_temperature,
                profile="thinking",
            )
        except LLMCallError as exc:
            payload = {"hypotheses": []}
            errors.append(str(exc))

        evolved_hypotheses = []
        raw_payload_keys = [_payload_keys(payload)]
        valid_source_ids = {hypothesis.hypothesis_id for hypothesis in source_hypotheses}
        source_title_lookup = {
            hypothesis.title.casefold(): hypothesis.hypothesis_id for hypothesis in source_hypotheses if hypothesis.title
        }
        hypothesis_items = _extract_hypothesis_items(payload)
        if not hypothesis_items and not errors:
            errors.append("Evolution returned valid JSON but no hypothesis list was found.")

        for item in hypothesis_items:
            parent_ids = [parent_id for parent_id in item.get("parent_ids", []) if parent_id in valid_source_ids]
            if not parent_ids:
                for parent_title in coerce_string_list(item.get("parent_titles") or item.get("parent_hypotheses") or []):
                    matched_id = source_title_lookup.get(parent_title.casefold())
                    if matched_id and matched_id not in parent_ids:
                        parent_ids.append(matched_id)
            if not parent_ids:
                candidate_text = "\n".join(
                    [
                        str(item.get("title") or ""),
                        str(item.get("core_hypothesis") or item.get("hypothesis") or item.get("description") or ""),
                        str(item.get("problem_framing") or item.get("problem_frame") or ""),
                        str(item.get("central_insight") or item.get("key_insight") or ""),
                        str(item.get("theoretical_story") or item.get("causal_story") or item.get("story") or ""),
                        str(item.get("mechanism") or ""),
                        str(item.get("novelty_rationale") or item.get("rationale") or ""),
                    ]
                ).strip()
                ranked_parent_matches = sorted(
                    source_hypotheses,
                    key=lambda hypothesis: (
                        similarity_score(candidate_text, hypothesis.text),
                        hypothesis.elo_score,
                    ),
                    reverse=True,
                )
                parent_ids = [hypothesis.hypothesis_id for hypothesis in ranked_parent_matches[:2] if candidate_text]
            candidate = Hypothesis(
                hypothesis_id=generate_unique_id("E"),
                title=str(item.get("title") or item.get("name") or item.get("short_title") or "Untitled evolved hypothesis").strip(),
                text=_compose_hypothesis_text(item),
                focus_area=str(item.get("focus_area", "")).strip(),
                primary_bottleneck=str(item.get("primary_bottleneck", "")).strip(),
                rationale=str(item.get("novelty_rationale") or item.get("rationale") or "").strip(),
                mechanism=str(item.get("mechanism", "")).strip(),
                problem_framing=str(item.get("problem_framing") or item.get("problem_frame") or "").strip(),
                central_insight=str(item.get("central_insight") or item.get("key_insight") or "").strip(),
                theoretical_story=str(
                    item.get("theoretical_story")
                    or item.get("causal_story")
                    or item.get("story")
                    or item.get("storyline")
                    or ""
                ).strip(),
                why_not_simple_combination=str(
                    item.get("why_not_simple_combination")
                    or item.get("why_not_just_combination")
                    or item.get("not_just_combination")
                    or ""
                ).strip(),
                generation_strategy=str(item.get("generation_strategy", "combination")).strip(),
                mutation_operator=str(item.get("mutation_operator") or item.get("generation_strategy") or "combination").strip(),
                evolution_delta=str(item.get("delta_from_parents") or item.get("diversity_reason") or "").strip(),
                origin="evolution",
                parent_ids=dedupe_preserve_order(parent_ids)[:2],
                predictions=dedupe_preserve_order(item.get("predictions", [])),
                key_assumptions=dedupe_preserve_order(item.get("key_assumptions") or item.get("assumptions") or []),
                validation_experiments=dedupe_preserve_order(
                    item.get("test_plan") or item.get("validation_experiments") or item.get("experiments") or []
                ),
                references=dedupe_preserve_order(item.get("references", [])),
                created_in_iteration=context.iteration_number + 1,
            )
            if not candidate.text:
                candidate.text = candidate.title
            if _is_duplicate_candidate(candidate, context, evolved_hypotheses):
                continue
            evolved_hypotheses.append(candidate)

        refill_attempts = 0
        refill_added = 0
        while len(evolved_hypotheses) < target_evolved_hypotheses and refill_attempts < 2:
            missing_hypotheses = target_evolved_hypotheses - len(evolved_hypotheses)
            refill_attempts += 1
            try:
                refill_payload = call_json(
                    build_evolution_refill_messages(
                        research_goal=research_goal.prompt_context(),
                        research_plan=research_plan.to_dict(),
                        top_hypotheses=[hypothesis.compact_summary() for hypothesis in source_hypotheses],
                        accepted_hypotheses=[hypothesis.compact_summary() for hypothesis in evolved_hypotheses],
                        literature_notes=literature_notes,
                        meta_feedback=meta_feedback,
                        research_overview=research_overview,
                        trajectory_state=trajectory_state,
                        missing_hypotheses=missing_hypotheses,
                        uncovered_focus_areas=coverage["uncovered_focus_areas"],
                        overexplored_areas=coverage["overexplored_areas"],
                    ),
                    model=research_goal.llm_model,
                    temperature=min(max(evolution_temperature, 0.0) + 0.02, 0.82),
                    profile="thinking",
                )
            except LLMCallError as exc:
                errors.append(f"Evolution refill {refill_attempts} failed: {exc}")
                break

            raw_payload_keys.append(_payload_keys(refill_payload))
            refill_items = _extract_hypothesis_items(refill_payload)
            if not refill_items:
                errors.append(f"Evolution refill {refill_attempts} returned valid JSON but no hypothesis list was found.")
                break

            added_this_attempt = 0
            for item in refill_items:
                parent_ids = [parent_id for parent_id in item.get("parent_ids", []) if parent_id in valid_source_ids]
                if not parent_ids:
                    for parent_title in coerce_string_list(item.get("parent_titles") or item.get("parent_hypotheses") or []):
                        matched_id = source_title_lookup.get(parent_title.casefold())
                        if matched_id and matched_id not in parent_ids:
                            parent_ids.append(matched_id)
                if not parent_ids:
                    continue
                candidate = Hypothesis(
                    hypothesis_id=generate_unique_id("E"),
                    title=str(item.get("title") or item.get("name") or item.get("short_title") or "Untitled evolved hypothesis").strip(),
                    text=_compose_hypothesis_text(item),
                    focus_area=str(item.get("focus_area", "")).strip(),
                    primary_bottleneck=str(item.get("primary_bottleneck", "")).strip(),
                    rationale=str(item.get("novelty_rationale") or item.get("rationale") or "").strip(),
                    mechanism=str(item.get("mechanism", "")).strip(),
                    problem_framing=str(item.get("problem_framing") or item.get("problem_frame") or "").strip(),
                    central_insight=str(item.get("central_insight") or item.get("key_insight") or "").strip(),
                    theoretical_story=str(
                        item.get("theoretical_story")
                        or item.get("causal_story")
                        or item.get("story")
                        or item.get("storyline")
                        or ""
                    ).strip(),
                    why_not_simple_combination=str(
                        item.get("why_not_simple_combination")
                        or item.get("why_not_just_combination")
                        or item.get("not_just_combination")
                        or ""
                    ).strip(),
                    generation_strategy=str(item.get("generation_strategy", "combination")).strip(),
                    mutation_operator=str(item.get("mutation_operator") or item.get("generation_strategy") or "combination").strip(),
                    evolution_delta=str(item.get("delta_from_parents") or item.get("diversity_reason") or "").strip(),
                    origin="evolution",
                    parent_ids=dedupe_preserve_order(parent_ids)[:2],
                    predictions=dedupe_preserve_order(item.get("predictions", [])),
                    key_assumptions=dedupe_preserve_order(item.get("key_assumptions") or item.get("assumptions") or []),
                    validation_experiments=dedupe_preserve_order(
                        item.get("test_plan") or item.get("validation_experiments") or item.get("experiments") or []
                    ),
                    references=dedupe_preserve_order(item.get("references", [])),
                    created_in_iteration=context.iteration_number + 1,
                )
                if not candidate.text:
                    candidate.text = candidate.title
                if _is_duplicate_candidate(candidate, context, evolved_hypotheses):
                    continue
                evolved_hypotheses.append(candidate)
                added_this_attempt += 1

            refill_added += added_this_attempt
            if added_this_attempt == 0:
                errors.append(f"Evolution refill {refill_attempts} did not add any distinct hypotheses.")
                break

        evolved_hypotheses = _prioritize_diverse_hypotheses(
            evolved_hypotheses,
            target_evolved_hypotheses,
            preferred_focus_areas=preferred_focus_areas,
            penalized_focus_areas=coverage["overexplored_areas"],
        )

        return StepResult(
            name="evolution",
            hypotheses=evolved_hypotheses,
            data={
                "source_hypotheses": [hypothesis.compact_summary() for hypothesis in source_hypotheses],
                "source_selection": {
                    "source_budget": source_budget,
                    "selected_ids": [hypothesis.hypothesis_id for hypothesis in source_hypotheses],
                    "selected_titles": [hypothesis.title for hypothesis in source_hypotheses],
                },
                "adaptive_tuning": {
                    "base_evolution_temperature": research_goal.evolution_temperature,
                    "temperature_used": evolution_temperature,
                    "temperature_adjustments": temperature_adjustments,
                    "base_source_budget": max(3, research_goal.top_k_hypotheses + 1),
                    "source_budget": source_budget,
                    "source_budget_adjustments": source_budget_adjustments,
                },
                "literature": literature_notes,
                "temperature_used": evolution_temperature,
                "frontier_diversity": diversity_snapshot,
                "trajectory_state": trajectory_state,
                "uncovered_focus_areas": coverage["uncovered_focus_areas"],
                "overexplored_areas": coverage["overexplored_areas"],
                "literature_mode": "reuse_source_idea_opencode_notes",
                "requested_hypotheses": target_evolved_hypotheses,
                "accepted_hypotheses": len(evolved_hypotheses),
                "refill_attempts": refill_attempts,
                "refill_added": refill_added,
                "raw_payload_keys": raw_payload_keys,
            },
            errors=errors,
            duration=time.time() - start_time,
        )

    def _looks_duplicate(self, candidate: Hypothesis, context: ContextMemory) -> bool:
        return _is_duplicate_candidate(candidate, context)


class MetaReviewAgent:
    def summarize_and_feedback(
        self,
        research_goal: ResearchGoal,
        context: ContextMemory,
        proximity_data: Dict[str, Any],
    ) -> StepResult:
        start_time = time.time()
        research_plan = context.research_plan or ResearchPlan.from_goal(research_goal)
        ranked = context.get_ranked_hypotheses()[:6]
        recent_reviews = [_build_review_artifact(hypothesis) for hypothesis in ranked]
        coverage = _focus_area_guidance(research_plan, context)
        trajectory_state = _trajectory_state_snapshot(research_plan, context, coverage)
        seeded_notes = dedupe_notes(
            note
            for hypothesis in ranked
            for note in hypothesis.literature_notes
        )[: research_goal.max_literature_results]
        literature_notes = list(seeded_notes)
        errors: List[str] = []
        opencode_audit: Dict[str, Any] = {}

        try:
            payload, opencode_audit = self._fetch_opencode_meta_review(
                research_goal=research_goal,
                research_plan=research_plan,
                context=context,
                ranked=ranked,
                recent_reviews=recent_reviews,
                proximity_data=proximity_data,
                trajectory_state=trajectory_state,
                literature_notes=literature_notes,
            )
            payload = _coerce_mapping(payload, self._fallback_meta_review(context))
            errors.extend(coerce_string_list(opencode_audit.get("errors", [])))
        except Exception as exc:  # noqa: BLE001
            payload = self._fallback_meta_review(context)
            errors.append(str(exc))

        critique = dedupe_preserve_order(payload.get("meta_review_critique", []))
        raw_story_guidance = payload.get("story_guidance", {})
        story_guidance = raw_story_guidance if isinstance(raw_story_guidance, dict) else {}
        story_guidance = {
            "frontier_story": str(story_guidance.get("frontier_story") or "").strip(),
            "method_stacking_patterns": dedupe_preserve_order(story_guidance.get("method_stacking_patterns", [])),
            "theory_gaps": dedupe_preserve_order(story_guidance.get("theory_gaps", [])),
            "next_generation_story_targets": dedupe_preserve_order(story_guidance.get("next_generation_story_targets", [])),
            "combination_policy": str(story_guidance.get("combination_policy") or "").strip(),
        }
        overview = payload.get("research_overview", {})
        if isinstance(overview, dict):
            top_titles = []
            for hypothesis in ranked[:3]:
                top_titles.append(hypothesis.title)
            overview.setdefault("top_ranked_hypotheses", top_titles)
            overview.setdefault("frontier_storyline", "")
            overview.setdefault("anti_combination_guidance", [])
            overview.setdefault("theory_gaps", [])

        context.meta_review_feedback.append(
            {
                "meta_review_critique": critique,
                "story_guidance": story_guidance,
                "generation_guidance": dedupe_preserve_order(payload.get("generation_guidance", [])),
                "reflection_guidance": dedupe_preserve_order(payload.get("reflection_guidance", [])),
                "ranking_guidance": dedupe_preserve_order(payload.get("ranking_guidance", [])),
                "trajectory_state": trajectory_state,
                "research_overview": overview,
                "expert_profiles": dedupe_preserve_order(payload.get("expert_profiles", [])),
            }
        )
        context.research_overviews.append(overview)

        return StepResult(
            name="meta_review",
            hypotheses=ranked[:3],
            data={
                **context.latest_meta_review(),
                "literature_reuse_count": len(seeded_notes),
                "meta_review_mode": "opencode",
                "opencode_audit": opencode_audit,
            },
            errors=errors,
            duration=time.time() - start_time,
        )

    def _fetch_opencode_meta_review(
        self,
        research_goal: ResearchGoal,
        research_plan: ResearchPlan,
        context: ContextMemory,
        ranked: Sequence[Hypothesis],
        recent_reviews: List[Dict[str, Any]],
        proximity_data: Dict[str, Any],
        trajectory_state: Dict[str, Any],
        literature_notes: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        timeout_value = config.get(
            "opencode_meta_review_timeout_seconds",
            config.get("opencode_rerank_timeout_seconds", 900),
        )
        try:
            timeout = max(1, int(timeout_value or 900))
        except (TypeError, ValueError):
            timeout = 900
        prompt = self._build_opencode_meta_review_prompt(
            research_goal=research_goal,
            research_plan=research_plan,
            context=context,
            ranked=ranked,
            recent_reviews=recent_reviews,
            proximity_data=proximity_data,
            trajectory_state=trajectory_state,
            literature_notes=literature_notes,
        )
        command = ["opencode", "run", "--dangerously-skip-permissions"]
        raw_stdout = ""
        raw_stderr = ""
        errors: List[str] = []
        payload: Dict[str, Any] = {}
        try:
            completed = _run_opencode(prompt, timeout)
            raw_stdout = completed.stdout or ""
            raw_stderr = completed.stderr or ""
            if completed.returncode != 0:
                errors.append(f"opencode meta-review exited with {completed.returncode}: {(raw_stderr or raw_stdout)[:1200]}")
            else:
                payload = _parse_json_object_from_text(raw_stdout)
                if not payload:
                    errors.append("opencode meta-review returned no parseable JSON payload.")
        except subprocess.TimeoutExpired as exc:
            errors.append(f"opencode meta-review timed out after {timeout}s: {exc}")
        except OSError as exc:
            errors.append(f"could not run opencode meta-review: {exc}")
        return payload, {
            "command": "opencode run --dangerously-skip-permissions",
            "timeout_seconds": timeout,
            "errors": errors,
            "stdout_tail": raw_stdout[-2000:],
            "stderr_tail": raw_stderr[-2000:],
        }

    def _build_opencode_meta_review_prompt(
        self,
        research_goal: ResearchGoal,
        research_plan: ResearchPlan,
        context: ContextMemory,
        ranked: Sequence[Hypothesis],
        recent_reviews: List[Dict[str, Any]],
        proximity_data: Dict[str, Any],
        trajectory_state: Dict[str, Any],
        literature_notes: List[Dict[str, Any]],
    ) -> str:
        payload = {
            "task": "Perform a meta-review over the current research state.",
            "instructions": [
                "Use the included hypotheses, review artifacts, tournament history, OpenCode rerank history, proximity diagnostics, and literature notes.",
                "Do not run another paper search; this is a synthesis and guidance pass.",
                "Identify recurring critique, blind spots, frontier story quality, method-stacking risk, and next-iteration guidance.",
                "Return strict JSON only, with no markdown or commentary outside JSON.",
            ],
            "research_goal": research_goal.prompt_context(),
            "research_plan": research_plan.to_dict(),
            "ranked_hypotheses": [hypothesis.compact_summary() for hypothesis in ranked],
            "recent_reviews": recent_reviews[-10:],
            "tournament_history": (context.tournament_results + context.opencode_rerank_history)[-12:],
            "proximity_summary": {
                "clusters": proximity_data.get("clusters", []),
                "duplicate_candidates": proximity_data.get("duplicate_candidates", []),
            },
            "trajectory_state": trajectory_state,
            "literature_notes": literature_notes,
            "required_schema": {
                "meta_review_critique": ["string"],
                "story_guidance": {
                    "frontier_story": "string",
                    "method_stacking_patterns": ["string"],
                    "theory_gaps": ["string"],
                    "next_generation_story_targets": ["string"],
                    "combination_policy": "string",
                },
                "generation_guidance": ["string"],
                "reflection_guidance": ["string"],
                "ranking_guidance": ["string"],
                "research_overview": {
                    "summary": "string",
                    "frontier_storyline": "string",
                    "anti_combination_guidance": ["string"],
                    "theory_gaps": ["string"],
                    "priority_areas": ["string"],
                    "top_ranked_hypotheses": ["string"],
                    "suggested_next_steps": ["string"],
                    "suggested_experiments": ["string"],
                },
                "expert_profiles": ["string"],
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _fallback_meta_review(self, context: ContextMemory) -> Dict[str, Any]:
        ranked = context.get_ranked_hypotheses()
        research_plan = context.research_plan or ResearchPlan(objective="fallback")
        trajectory_state = _trajectory_state_snapshot(research_plan, context)
        critique = []
        if len([hypothesis for hypothesis in ranked if not hypothesis.is_active]) > 0:
            critique.append("Several hypotheses were rejected or gated out; the search space may be too noisy.")
        if context.compute_statistics().get("average_review_score", 0) < 3.2:
            critique.append("Average review quality is only moderate; the next round should focus on stronger grounding.")
        if "coverage_gap" in trajectory_state.get("stagnation_signals", []):
            critique.append("Some requested focus areas are still uncovered; the next round should widen the frontier.")
        if "frontier_clustered" in trajectory_state.get("stagnation_signals", []):
            critique.append("The frontier is clustering around one family; prioritize divergence over another incremental refinement.")
        if not critique:
            critique.append("The current frontier looks reasonably healthy; focus on sharper experiments and differentiation.")
        return {
            "meta_review_critique": critique,
            "story_guidance": {
                "frontier_story": "No reliable LLM meta-review was available; preserve only hypotheses with a clear problem-to-mechanism story.",
                "method_stacking_patterns": ["Watch for proposals that only combine named techniques without a unifying causal claim."],
                "theory_gaps": ["Clarify the central insight and falsifiable prediction for the next generation batch."],
                "next_generation_story_targets": ["Generate hypotheses from a problem tension first, then derive the mechanism."],
                "combination_policy": "Combination is acceptable only when one causal story explains why the ingredients must interact.",
            },
            "generation_guidance": ["Use stronger literature grounding and avoid near-duplicate proposals."],
            "reflection_guidance": ["Be explicit about failure modes, falsifiable experiments, and story coherence."],
            "ranking_guidance": ["Prefer direct pairwise comparisons among similar hypotheses and newer challengers."],
            "trajectory_state": trajectory_state,
            "research_overview": {
                "summary": "Fallback meta-review generated locally.",
                "frontier_storyline": "",
                "anti_combination_guidance": ["Do not advance ideas whose novelty is only method aggregation."],
                "theory_gaps": ["Need explicit central insight and causal story."],
                "priority_areas": [hypothesis.focus_area for hypothesis in ranked[:3] if hypothesis.focus_area],
                "top_ranked_hypotheses": [hypothesis.title for hypothesis in ranked[:3]],
                "suggested_next_steps": dedupe_preserve_order(
                    step for hypothesis in ranked[:3] for step in hypothesis.validation_experiments[:2]
                )[:5],
                "suggested_experiments": dedupe_preserve_order(
                    step for hypothesis in ranked[:3] for step in hypothesis.validation_experiments[:3]
                )[:5],
            },
            "expert_profiles": [],
        }
