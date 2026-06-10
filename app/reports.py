from __future__ import annotations

from typing import Any, Dict, List


def _line(text: str = "") -> str:
    return f"{text}\n"


def _format_list(items: List[Any]) -> str:
    if not items:
        return "- None\n"
    return "".join(f"- {item}\n" for item in items)


def _top_hypotheses(cycle: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    final_step = (
        cycle.get("steps", {}).get("codex_reranking_final")
        or cycle.get("steps", {}).get("ranking_final")
        or cycle.get("steps", {}).get("codex_reranking")
        or cycle.get("steps", {}).get("ranking")
        or {}
    )
    hypotheses = final_step.get("hypotheses", [])
    return sorted(hypotheses, key=lambda item: item.get("elo_score", 0), reverse=True)[:limit]


def _references(cycle: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
    collected = {}
    for step in cycle.get("steps", {}).values():
        for note in step.get("literature", []):
            key = note.get("arxiv_id") or note.get("title")
            if key:
                collected[key] = note

        literature_by_hypothesis = step.get("literature_by_hypothesis", {})
        if isinstance(literature_by_hypothesis, dict):
            for notes in literature_by_hypothesis.values():
                for note in notes:
                    key = note.get("arxiv_id") or note.get("title")
                    if key:
                        collected[key] = note

        for hypothesis in step.get("hypotheses", []):
            for note in hypothesis.get("literature_notes", []):
                key = note.get("arxiv_id") or note.get("title")
                if key:
                    collected[key] = note
    return list(collected.values())[:limit]


def build_markdown_report(cycles: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    parts.append(_line("# AI Co-Scientist Run"))

    for cycle in cycles:
        parts.append(_line(f"## Cycle {cycle.get('iteration', 'Unknown')}"))
        parts.append(_line(f"Duration: {cycle.get('cycle_duration', 0):.2f}s"))
        parts.append(_line("### Statistics"))
        parts.append(_line(f"Before: `{cycle.get('statistics_before', {})}`"))
        parts.append(_line(f"After: `{cycle.get('statistics_after', {})}`"))

        errors = cycle.get("errors", [])
        if errors:
            parts.append(_line("### Errors"))
            parts.append(_format_list(errors))

        research_plan = cycle.get("research_plan", {})
        reference_paper = cycle.get("reference_paper", {})
        if isinstance(reference_paper, dict) and reference_paper:
            parts.append(_line("### Reference Paper"))
            title = reference_paper.get("title") or reference_paper.get("arxiv_id") or "Reference paper"
            url = reference_paper.get("arxiv_url", "")
            if url:
                parts.append(_line(f"- [{title}]({url})"))
            else:
                parts.append(_line(f"- {title}"))
            if reference_paper.get("concise_summary"):
                parts.append(_line(f"Summary: {reference_paper.get('concise_summary')}"))
            if reference_paper.get("limitations"):
                parts.append(_line("Reference limitations to build on:"))
                parts.append(_format_list(reference_paper.get("limitations", [])[:5]))

        parts.append(_line("### Research Plan"))
        parts.append(_line(f"Objective: {research_plan.get('objective', '')}"))
        parts.append(_line(f"Domain: {research_plan.get('domain', '')}"))
        parts.append(_line("Focus areas:"))
        parts.append(_format_list(research_plan.get("focus_areas", [])))
        parts.append(_line("Success criteria:"))
        parts.append(_format_list(research_plan.get("success_criteria", [])))

        parts.append(_line("### Top Hypotheses"))
        top_hypotheses = _top_hypotheses(cycle)
        if not top_hypotheses:
            parts.append(_line("No hypotheses were ranked."))
        for index, hypothesis in enumerate(top_hypotheses, start=1):
            parts.append(_line(f"#### {index}. {hypothesis.get('title', 'Untitled')}"))
            parts.append(_line(f"ID: `{hypothesis.get('id')}`"))
            parts.append(_line(f"Elo: `{hypothesis.get('elo_score', 0):.2f}`"))
            parts.append(_line(f"Verdict: `{hypothesis.get('review_verdict', 'n/a')}`"))
            parts.append(_line(f"Focus: {hypothesis.get('focus_area', '')}"))
            if hypothesis.get("primary_bottleneck"):
                parts.append(_line(f"Bottleneck: `{hypothesis.get('primary_bottleneck')}`"))
            if hypothesis.get("problem_framing"):
                parts.append(_line(f"Problem framing: {hypothesis.get('problem_framing')}"))
            if hypothesis.get("central_insight"):
                parts.append(_line(f"Central insight: {hypothesis.get('central_insight')}"))
            if hypothesis.get("theoretical_story"):
                parts.append(_line(f"Story: {hypothesis.get('theoretical_story')}"))
            if hypothesis.get("origin") == "evolution":
                parts.append(_line(f"Parents: `{hypothesis.get('parent_ids', [])}`"))
                if hypothesis.get("mutation_operator"):
                    parts.append(_line(f"Mutation operator: `{hypothesis.get('mutation_operator')}`"))
                if hypothesis.get("evolution_delta"):
                    parts.append(_line(f"Evolution delta: {hypothesis.get('evolution_delta')}"))
            if hypothesis.get("why_not_simple_combination"):
                parts.append(_line(f"Why not a simple combination: {hypothesis.get('why_not_simple_combination')}"))
            parts.append(_line(hypothesis.get("text", "")))
            parts.append(_line(f"Scores: `{hypothesis.get('scores', {})}`"))
            review_artifacts = hypothesis.get("review_artifacts", [])
            if review_artifacts:
                latest_review = review_artifacts[-1]
                parts.append(_line("Latest review:"))
                if latest_review.get("short_summary"):
                    parts.append(_line(latest_review.get("short_summary", "")))
                if latest_review.get("reflection"):
                    parts.append(_line(f"Reflection: {latest_review.get('reflection')}"))
                if latest_review.get("feasibility_rationale"):
                    parts.append(_line(f"Feasibility: {latest_review.get('feasibility_rationale')}"))
                if latest_review.get("weaknesses"):
                    parts.append(_line("Weaknesses:"))
                    parts.append(_format_list(latest_review.get("weaknesses", [])[:5]))
            codex_reviews = hypothesis.get("codex_rerank_reviews", [])
            if codex_reviews:
                latest_codex = codex_reviews[-1]
                parts.append(_line("Codex rerank review:"))
                parts.append(_line(f"Rank: `{latest_codex.get('rank', 'n/a')}`"))
                if latest_codex.get("summary"):
                    parts.append(_line(latest_codex.get("summary", "")))
                if latest_codex.get("weaknesses"):
                    parts.append(_line("Codex weaknesses:"))
                    parts.append(_format_list(latest_codex.get("weaknesses", [])[:4]))
                if latest_codex.get("novelty_risks"):
                    parts.append(_line("Codex novelty risks:"))
                    parts.append(_format_list(latest_codex.get("novelty_risks", [])[:4]))
            prior_art_audit = hypothesis.get("prior_art_audit", {})
            if isinstance(prior_art_audit, dict) and prior_art_audit:
                parts.append(_line(f"Prior-art risk: `{prior_art_audit.get('novelty_risk', 'unknown')}`"))
                if prior_art_audit.get("short_summary"):
                    parts.append(_line(f"Prior-art summary: {prior_art_audit.get('short_summary')}"))
                similar_papers = hypothesis.get("prior_art_similar_papers", [])
                if similar_papers:
                    parts.append(_line("Closest prior art:"))
                    for paper in similar_papers[:3]:
                        title = paper.get("title", "Untitled")
                        score = paper.get("recall_score", "n/a")
                        url = paper.get("arxiv_url") or paper.get("url") or ""
                        if url:
                            parts.append(_line(f"- [{title}]({url}) score={score}"))
                        else:
                            parts.append(_line(f"- {title} score={score}"))
            experiments = hypothesis.get("validation_experiments", [])
            if experiments:
                parts.append(_line("Validation experiments:"))
                parts.append(_format_list(experiments))
        parts.append(_line("### Ranking Diagnostics"))
        for step_name in ("ranking", "codex_reranking", "ranking_final", "codex_reranking_final"):
            step = cycle.get("steps", {}).get(step_name, {})
            if step:
                if step_name.startswith("codex"):
                    parts.append(
                        _line(
                            f"- {step_name}: status={step.get('status', 'n/a')}, "
                            f"candidates={step.get('candidate_ids', [])}"
                        )
                    )
                else:
                    parts.append(
                        _line(
                            f"- {step_name}: matches={len(step.get('matches', []))}, "
                            f"skipped_cached_pairs={step.get('skipped_cached_pairs', 0)}"
                        )
                    )

        parts.append(_line("### Generation Diagnostics"))
        for step_name in ("generation", "prior_art_check", "evolution", "evolution_replacement_pruning", "prior_art_check_evolved"):
            step = cycle.get("steps", {}).get(step_name, {})
            if not step:
                continue
            if step_name == "evolution_replacement_pruning":
                parts.append(
                    _line(
                        f"- {step_name}: new_active_evolved={step.get('new_active_evolved_count', 0)}, "
                        f"pruned={step.get('pruned_count', 0)}"
                    )
                )
            elif step_name.startswith("prior_art_check"):
                parts.append(
                    _line(
                        f"- {step_name}: checked={step.get('checked_count', 0)}, "
                        f"skipped={step.get('skipped_count', 0)}, "
                        f"repaired={step.get('repaired_count', 0)}, "
                        f"rejected={step.get('rejected_count', 0)}"
                    )
                )
            else:
                parts.append(
                    _line(
                        f"- {step_name}: hypotheses={len(step.get('hypotheses', []))}, "
                        f"temperature={step.get('temperature_used', 'n/a')}, "
                        f"frontier_diversity={step.get('frontier_diversity', {})}"
                    )
                )
            if step.get("trajectory_state"):
                parts.append(_line(f"  trajectory_state={step.get('trajectory_state')}"))
            if step.get("adaptive_tuning"):
                parts.append(_line(f"  adaptive_tuning={step.get('adaptive_tuning')}"))
            if step_name == "evolution" and step.get("source_selection"):
                parts.append(_line(f"  source_selection={step.get('source_selection')}"))

        meta_review = cycle.get("meta_review", {})
        overview = meta_review.get("research_overview", {}) if isinstance(meta_review, dict) else {}
        story_guidance = meta_review.get("story_guidance", {}) if isinstance(meta_review, dict) else {}
        parts.append(_line("### Meta-Review"))
        parts.append(_line("Critique:"))
        parts.append(_format_list(meta_review.get("meta_review_critique", []) if isinstance(meta_review, dict) else []))
        if isinstance(story_guidance, dict) and story_guidance:
            parts.append(_line(f"Frontier story: {story_guidance.get('frontier_story', '')}"))
            parts.append(_line("Method-stacking patterns:"))
            parts.append(_format_list(story_guidance.get("method_stacking_patterns", [])))
            parts.append(_line("Theory gaps:"))
            parts.append(_format_list(story_guidance.get("theory_gaps", [])))
        parts.append(_line(f"Overview: {overview.get('summary', '')}"))
        if overview.get("frontier_storyline"):
            parts.append(_line(f"Frontier storyline: {overview.get('frontier_storyline')}"))
        if overview.get("anti_combination_guidance"):
            parts.append(_line("Anti-combination guidance:"))
            parts.append(_format_list(overview.get("anti_combination_guidance", [])))
        if overview.get("theory_gaps"):
            parts.append(_line("Theory gaps:"))
            parts.append(_format_list(overview.get("theory_gaps", [])))
        parts.append(_line("Suggested next steps:"))
        parts.append(_format_list(overview.get("suggested_next_steps", [])))
        parts.append(_line("Suggested experiments:"))
        parts.append(_format_list(overview.get("suggested_experiments", [])))

        parts.append(_line("### Grounding References"))
        references = _references(cycle)
        if not references:
            parts.append(_line("No references collected."))
        for reference in references:
            title = reference.get("title", "Untitled")
            url = reference.get("arxiv_url", "")
            citation = reference.get("citation", "")
            if url:
                parts.append(_line(f"- [{title}]({url}) - {citation}"))
            else:
                parts.append(_line(f"- {title} - {citation}"))

    return "".join(parts)
