from __future__ import annotations

import json
from typing import Any, Dict, List


BASE_SYSTEM_PROMPT = """You are an AI co-scientist.
You are rigorous, literature-aware, and conservative about unsupported claims.
Always return strict JSON with no markdown, no code fences, and no extra commentary.
When uncertain, make the uncertainty explicit inside the JSON fields instead of hedging outside the schema."""


def _json_block(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _serialize_literature(literature_notes: List[Dict[str, Any]]) -> str:
    concise = []
    for note in literature_notes:
        concise.append(
            {
                "title": note.get("title"),
                "summary": note.get("summary"),
                "citation": note.get("citation"),
                "source": note.get("source"),
                "sources": note.get("sources"),
                "published": note.get("published"),
                "updated": note.get("updated"),
                "year": note.get("year"),
                "venue": note.get("venue"),
                "citation_count": note.get("citation_count"),
                "selection_reason": note.get("selection_reason"),
                "arxiv_url": note.get("arxiv_url"),
            }
        )
    return _json_block({"literature": concise})


def _serialize_hypotheses(hypotheses: List[Dict[str, Any]], limit: int = 8) -> str:
    return _json_block({"hypotheses": hypotheses[:limit]})


def _serialize_feedback(meta_feedback: Dict[str, Any], research_overview: Dict[str, Any], trajectory_state: Dict[str, Any] | None = None) -> str:
    payload = {
        "meta_review_critique": meta_feedback.get("meta_review_critique", []),
        "generation_guidance": meta_feedback.get("generation_guidance", []),
        "reflection_guidance": meta_feedback.get("reflection_guidance", []),
        "ranking_guidance": meta_feedback.get("ranking_guidance", []),
        "trajectory_state": trajectory_state or meta_feedback.get("trajectory_state", {}),
        "research_overview": research_overview or {},
    }
    return _json_block(payload)


def build_research_plan_messages(research_goal: str, constraints: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt = f"""
Convert the research goal into a structured research plan that downstream agents can share.

Research goal:
{research_goal}

Explicit constraints:
{_json_block(constraints or {})}

Please infer and structure:
- the scientific objective
- the likely domain
- focus areas
- key questions
- success criteria
- evaluation criteria weights shared across all later stages
- constraints and things to avoid
- preferred evidence and useful literature search queries
- tool preferences and any short notes for future iterations

Required JSON schema:
{{
  "objective": "string",
  "domain": "string",
  "focus_areas": ["string"],
  "key_questions": ["string"],
  "success_criteria": ["string"],
  "evaluation_criteria": {{
    "alignment": 0.0,
    "novelty": 0.0,
    "plausibility": 0.0,
    "testability": 0.0
  }},
  "constraints": ["string"],
  "avoid": ["string"],
  "preferred_evidence": ["string"],
  "seed_queries": ["string"],
  "tool_preferences": ["string"],
  "notes": ["string"]
}}
"""
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
    ]


def build_goal_safety_review_messages(research_goal: str, constraints: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt = f"""
Review whether this research goal is safe and ethically appropriate for an autonomous scientific ideation workflow.
Focus on dual-use risk, harmful enablement, unethical experimentation, and requests that should require explicit human governance.

Research goal:
{research_goal}

Explicit constraints:
{_json_block(constraints or {})}

Return a conservative decision. Safe basic science, literature review, benchmarking, or defensive safety work can be allowed,
but goals that directly enable harm, evasion, misuse, or unsafe wet-lab execution should be blocked or escalated.

Required JSON schema:
{{
  "allowed": true,
  "risk_level": "low|medium|high|blocked",
  "decision": "allow|allow_with_human_review|block",
  "reasons": ["string"],
  "required_mitigations": ["string"]
}}
"""
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
    ]


def build_literature_query_planning_messages(
    research_goal: str,
    research_plan: Dict[str, Any],
    search_context: Dict[str, Any],
    candidate_queries: List[str],
    max_queries: int,
) -> List[Dict[str, str]]:
    prompt = f"""
Construct academic literature search queries before live API search.
The raw goal or hypothesis title may be too specific, so convert it into compact keyword bundles that can retrieve relevant prior work.

Research goal:
{research_goal}

Research plan:
{_json_block(research_plan)}

Search context:
{_json_block(search_context)}

Candidate raw queries:
{_json_block({"queries": candidate_queries})}

Guidelines:
- Produce at most {max_queries} queries.
- Each query should combine 3 to 6 high-signal academic keywords or phrases with OR.
- Prefer field vocabulary used in paper titles and abstracts, not full hypothesis prose.
- Include synonyms and adjacent terms that broaden retrieval without drifting from the scientific objective.
- Do not include markdown, explanations, or source-specific API syntax such as cat: filters.

Required JSON schema:
{{
  "queries": [
    {{
      "query": "\"phrase one\" OR keyword OR \"phrase two\"",
      "keywords": ["string"],
      "rationale": "string"
    }}
  ]
}}
"""
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
    ]


def build_generation_messages(
    research_goal: str,
    research_plan: Dict[str, Any],
    context_hypotheses: List[Dict[str, Any]],
    literature_notes: List[Dict[str, Any]],
    meta_feedback: Dict[str, Any],
    research_overview: Dict[str, Any],
    trajectory_state: Dict[str, Any],
    num_hypotheses: int,
    uncovered_focus_areas: List[str],
    overexplored_areas: List[str],
) -> List[Dict[str, str]]:
    prompt = f"""
Generate {num_hypotheses} new research hypotheses.
Follow the shared research plan, use the literature as grounding, and use meta-review feedback to explore missing or weak areas.

Research goal:
{research_goal}

Research plan:
{_json_block(research_plan)}

Current hypothesis memory:
{_serialize_hypotheses(context_hypotheses)}

Recent literature context:
{_serialize_literature(literature_notes)}

Recent meta-review feedback and research overview:
{_serialize_feedback(meta_feedback, research_overview, trajectory_state)}

Undercovered focus areas that should be prioritized:
{_json_block({"uncovered_focus_areas": uncovered_focus_areas})}

Overexplored clusters to avoid repeating unless the mechanism is clearly different:
{_json_block({"overexplored_areas": overexplored_areas})}

Requirements:
- Produce genuinely distinct hypotheses, not paraphrases.
- Each hypothesis must contain a mechanism or causal story.
- Each hypothesis must include an actionable validation path.
- Prefer ideas that are novel but still testable.
- Treat the returned set as a portfolio. Avoid collapsing into one mechanism family.
- Before writing the JSON, internally assign each output slot to a different focus area or bottleneck so the batch is intentionally spread out.
- If generating 4 or more hypotheses, cover at least 3 mechanism families unless the goal explicitly forbids it.
- No single mechanism family should occupy more than half of the returned hypotheses unless there is overwhelming literature evidence that the family dominates the search space.
- You may draw from multiple generation modes such as literature-grounded synthesis, debate-inspired proposals, decomposition into sub-hypotheses, or expansion into uncovered areas.
- Avoid ideas that directly duplicate the current hypothesis memory.
- When undercovered focus areas are provided, cover as many of them as possible before revisiting crowded clusters.
- Include the primary bottleneck for each idea such as compute, memory bandwidth, latency, cache pressure, or verification overhead.
- If two hypotheses still land in the same focus area, they must attack different primary bottlenecks and propose different experiments.
- If `trajectory_state.stagnation_signals` indicates clustering or coverage gaps, explicitly use some hypotheses to widen the frontier rather than only polishing incumbents.
- If `trajectory_state.frontier_focus_gaps` is non-empty, reserve at least one hypothesis for one of those missing leaderboard areas unless the area is clearly invalidated by the current review evidence.
- Return exactly one top-level JSON object with the key "hypotheses". Do not rename the top-level key to "list", "items", or "results".
- Returning a single hypothesis object instead of a `hypotheses` array is invalid.
- Returning fewer than {num_hypotheses} hypotheses is invalid. Finish the full portfolio even if one idea looks strongest.

Required JSON schema:
{{
  "hypotheses": [
    {{
      "title": "string",
      "focus_area": "string",
      "primary_bottleneck": "compute|memory bandwidth|latency|cache pressure|verification overhead|other",
      "core_hypothesis": "string",
      "mechanism": "string",
      "novelty_rationale": "string",
      "generation_strategy": "literature_grounding|debate|decomposition|expansion|other",
      "key_assumptions": ["string"],
      "predictions": ["string"],
      "test_plan": ["string"],
      "references": ["string"],
      "search_queries": ["string"]
    }}
  ]
}}
"""
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
    ]


def build_generation_refill_messages(
    research_goal: str,
    research_plan: Dict[str, Any],
    accepted_hypotheses: List[Dict[str, Any]],
    literature_notes: List[Dict[str, Any]],
    meta_feedback: Dict[str, Any],
    research_overview: Dict[str, Any],
    trajectory_state: Dict[str, Any],
    missing_hypotheses: int,
    uncovered_focus_areas: List[str],
    overexplored_areas: List[str],
) -> List[Dict[str, str]]:
    prompt = f"""
The previous generation response under-filled the batch.
Produce exactly {missing_hypotheses} additional research hypotheses so the portfolio reaches the intended size.

Research goal:
{research_goal}

Research plan:
{_json_block(research_plan)}

Already accepted hypotheses that must not be repeated:
{_serialize_hypotheses(accepted_hypotheses)}

Recent literature context:
{_serialize_literature(literature_notes)}

Recent meta-review feedback and research overview:
{_serialize_feedback(meta_feedback, research_overview, trajectory_state)}

Undercovered focus areas that should be prioritized:
{_json_block({"uncovered_focus_areas": uncovered_focus_areas})}

Overexplored clusters to avoid repeating unless the mechanism is clearly different:
{_json_block({"overexplored_areas": overexplored_areas})}

Requirements:
- Return exactly {missing_hypotheses} additional hypotheses.
- Do not repeat titles, focus areas, or mechanisms that already appear in the accepted set unless the causal story is materially different.
- Prefer uncovered focus areas first, then use a clearly different mechanism family.
- Use each remaining slot to cover a different focus area or bottleneck whenever possible.
- Returning a single hypothesis object instead of a `hypotheses` array is invalid.
- Return exactly one top-level JSON object with the key "hypotheses".

Required JSON schema:
{{
  "hypotheses": [
    {{
      "title": "string",
      "focus_area": "string",
      "primary_bottleneck": "compute|memory bandwidth|latency|cache pressure|verification overhead|other",
      "core_hypothesis": "string",
      "mechanism": "string",
      "novelty_rationale": "string",
      "generation_strategy": "literature_grounding|debate|decomposition|expansion|other",
      "key_assumptions": ["string"],
      "predictions": ["string"],
      "test_plan": ["string"],
      "references": ["string"],
      "search_queries": ["string"]
    }}
  ]
}}
"""
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
    ]


def build_initial_review_messages(
    research_goal: str,
    research_plan: Dict[str, Any],
    hypothesis: Dict[str, Any],
    meta_feedback: Dict[str, Any],
) -> List[Dict[str, str]]:
    prompt = f"""
Perform a fast initial review of this hypothesis without external tools.
This is a gate for whether the idea should advance to a more literature-grounded review.

Research goal:
{research_goal}

Research plan:
{_json_block(research_plan)}

Hypothesis:
{_json_block(hypothesis)}

Recent meta-review feedback:
{_json_block(meta_feedback or {})}

Judge the hypothesis using the shared evaluation contract:
- alignment with the goal
- novelty
- plausibility
- testability
- use "reject" only for ideas that are out of scope, fundamentally invalid, or not worth keeping in the tournament pool

Required JSON schema:
{{
  "alignment_score": 1,
  "novelty_score": 1,
  "plausibility_score": 1,
  "feasibility_score": 1,
  "testability_score": 1,
  "verdict": "advance|revise|reject",
  "short_summary": "string",
  "strengths": ["string"],
  "weaknesses": ["string"],
  "critical_assumptions": ["string"],
  "improvement_actions": ["string"]
}}
"""
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
    ]


def build_full_review_messages(
    research_goal: str,
    research_plan: Dict[str, Any],
    hypothesis: Dict[str, Any],
    literature_notes: List[Dict[str, Any]],
    tournament_history: List[Dict[str, Any]],
    meta_feedback: Dict[str, Any],
) -> List[Dict[str, str]]:
    prompt = f"""
Perform a full review of this hypothesis using the literature notes as external grounding.
Be concrete about support, contradictions, failure modes, and experiments.

Research goal:
{research_goal}

Research plan:
{_json_block(research_plan)}

Hypothesis:
{_json_block(hypothesis)}

Literature notes:
{_serialize_literature(literature_notes)}

Recent tournament history:
{_json_block({"recent_matches": tournament_history[-6:]})}

Recent meta-review feedback:
{_json_block(meta_feedback or {})}

Required JSON schema:
{{
  "alignment_score": 1,
  "novelty_score": 1,
  "plausibility_score": 1,
  "feasibility_score": 1,
  "correctness_score": 1,
  "testability_score": 1,
  "verdict": "advance|revise|reject",
  "short_summary": "string",
  "strengths": ["string"],
  "weaknesses": ["string"],
  "critical_assumptions": ["string"],
  "supporting_observations": ["string"],
  "contradicting_observations": ["string"],
  "failure_modes": ["string"],
  "validation_experiments": ["string"],
  "improvement_actions": ["string"],
  "references": ["string"]
}}
"""
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
    ]


def build_specialized_review_messages(
    research_goal: str,
    research_plan: Dict[str, Any],
    hypothesis: Dict[str, Any],
    literature_notes: List[Dict[str, Any]],
    tournament_history: List[Dict[str, Any]],
    meta_feedback: Dict[str, Any],
) -> List[Dict[str, str]]:
    prompt = f"""
Perform the specialized Reflection-agent reviews for this hypothesis in one structured pass.
Cover three review types:
1. Deep verification: decompose assumptions into sub-assumptions and identify fundamental invalidating errors.
2. Observation review: assess whether the hypothesis explains long-tail or under-explained observations in the literature.
3. Simulation review: simulate the mechanism or proposed experiment step by step and summarize failure scenarios.

Research goal:
{research_goal}

Research plan:
{_json_block(research_plan)}

Hypothesis:
{_json_block(hypothesis)}

Literature notes:
{_serialize_literature(literature_notes)}

Recent tournament history:
{_json_block({"recent_matches": tournament_history[-8:]})}

Recent meta-review feedback:
{_json_block(meta_feedback or {})}

Required JSON schema:
{{
  "correctness_score": 1,
  "plausibility_score": 1,
  "testability_score": 1,
  "verdict": "advance|revise|reject",
  "short_summary": "string",
  "deep_verification": {{
    "assumption_tree": [
      {{
        "assumption": "string",
        "sub_assumptions": ["string"],
        "assessment": "supported|uncertain|contradicted",
        "fundamental": true,
        "reason": "string"
      }}
    ],
    "invalidating_assumptions": ["string"],
    "non_fundamental_repairs": ["string"]
  }},
  "observation_review": {{
    "explained_observations": ["string"],
    "unexplained_or_contradictory_observations": ["string"],
    "needed_observations": ["string"]
  }},
  "simulation_review": {{
    "simulation_trace": ["string"],
    "failure_scenarios": ["string"],
    "protocol_risks": ["string"]
  }},
  "failure_modes": ["string"],
  "validation_experiments": ["string"],
  "improvement_actions": ["string"],
  "references": ["string"]
}}
"""
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
    ]


def build_prior_art_audit_messages(
    research_goal: str,
    research_plan: Dict[str, Any],
    hypothesis: Dict[str, Any],
    similar_papers: List[Dict[str, Any]],
    attempt_number: int,
) -> List[Dict[str, str]]:
    prompt = f"""
Audit this hypothesis against recalled prior art.
Use the papers as evidence, not as inspiration to copy. The goal is to distinguish true duplicate methods from normal related work.

Research goal:
{research_goal}

Research plan:
{_json_block(research_plan)}

Hypothesis:
{_json_block(hypothesis)}

Recalled prior-art candidates:
{_json_block({"papers": similar_papers})}

Repair attempt number:
{attempt_number}

Decision policy:
- A strong duplicate means the hypothesis overlaps the same problem, the same core mechanism, and substantially the same novelty claim as at least one paper.
- A medium risk means the direction is close, but there is a plausible gap or materially different mechanism.
- A low risk means the papers are background or adjacent work rather than duplicates.
- If risk is high and a crisp gap exists, return decision "repair" and rewrite the hypothesis around that gap.
- If risk is high and no defensible gap exists, return decision "reject".
- If risk is low or medium, return decision "keep" unless a simple repair would materially improve novelty.
- Do not call something duplicate only because it uses the same benchmark, model family, or broad problem setting.

Required JSON schema:
{{
  "novelty_risk": "low|medium|high",
  "decision": "keep|repair|reject",
  "short_summary": "string",
  "overlap_assessment": [
    {{
      "paper_title": "string",
      "problem_overlap": "none|partial|high",
      "mechanism_overlap": "none|partial|high",
      "claim_overlap": "none|partial|high",
      "duplicate_level": "none|related|medium|strong",
      "rationale": "string"
    }}
  ],
  "gap_analysis": ["string"],
  "repair_strategy": "string",
  "revised_hypothesis": {{
    "title": "string",
    "focus_area": "string",
    "primary_bottleneck": "compute|memory bandwidth|latency|cache pressure|verification overhead|other",
    "core_hypothesis": "string",
    "mechanism": "string",
    "novelty_rationale": "string",
    "key_assumptions": ["string"],
    "predictions": ["string"],
    "test_plan": ["string"],
    "search_queries": ["string"]
  }}
}}
"""
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
    ]


def build_ranking_messages(
    research_goal: str,
    research_plan: Dict[str, Any],
    hypothesis_a: Dict[str, Any],
    hypothesis_b: Dict[str, Any],
    proximity_note: Dict[str, Any],
    mode: str,
) -> List[Dict[str, str]]:
    prompt = f"""
Compare the two hypotheses and decide which should win this tournament match.
Use pairwise scientific judgment. Do not simply add up any prior review scores. Those scores are only local signals and are not directly comparable across hypotheses.

Comparison mode:
{mode}

Research goal:
{research_goal}

Research plan:
{_json_block(research_plan)}

Hypothesis A:
{_json_block(hypothesis_a)}

Hypothesis B:
{_json_block(hypothesis_b)}

Proximity note:
{_json_block(proximity_note)}

If mode is "full_debate", internally weigh the strongest case for each side before deciding.
If mode is "quick_compare", still give a concrete winner and rationale.

Required JSON schema:
{{
  "winner_id": "string",
  "loser_id": "string",
  "comparison_summary": "string",
  "winner_reason": "string",
  "novelty_edge": "A|B|tie",
  "plausibility_edge": "A|B|tie",
  "testability_edge": "A|B|tie",
  "confidence": 0.0
}}
"""
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
    ]


def build_ranking_reaudit_messages(
    research_goal: str,
    research_plan: Dict[str, Any],
    hypothesis_a: Dict[str, Any],
    hypothesis_b: Dict[str, Any],
    proximity_note: Dict[str, Any],
    previous_decision: Dict[str, Any],
    mode: str,
) -> List[Dict[str, str]]:
    prompt = f"""
Re-audit this cached tournament decision from a different angle.
Do not defer to the previous winner. Stress-test the previous winner, steelman the previous loser, and check whether novelty, feasibility, testability, or risk-aware utility was underweighted.

Comparison mode:
{mode}

Research goal:
{research_goal}

Research plan:
{_json_block(research_plan)}

Hypothesis A:
{_json_block(hypothesis_a)}

Hypothesis B:
{_json_block(hypothesis_b)}

Previous cached decision:
{_json_block(previous_decision)}

Proximity note:
{_json_block(proximity_note)}

Give one final tournament winner after the adversarial re-audit.

Required JSON schema:
{{
  "winner_id": "string",
  "loser_id": "string",
  "comparison_summary": "string",
  "winner_reason": "string",
  "novelty_edge": "A|B|tie",
  "plausibility_edge": "A|B|tie",
  "testability_edge": "A|B|tie",
  "confidence": 0.0
}}
"""
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
    ]


def build_evolution_messages(
    research_goal: str,
    research_plan: Dict[str, Any],
    top_hypotheses: List[Dict[str, Any]],
    literature_notes: List[Dict[str, Any]],
    meta_feedback: Dict[str, Any],
    research_overview: Dict[str, Any],
    trajectory_state: Dict[str, Any],
    num_hypotheses: int,
    uncovered_focus_areas: List[str],
    overexplored_areas: List[str],
) -> List[Dict[str, str]]:
    prompt = f"""
Generate {num_hypotheses} new evolved hypotheses.
Important: create new hypotheses only. Do not overwrite or rewrite existing ones.
The goal is not merely to make variants. Each evolved hypothesis must be a credible improvement over the source hypotheses provided to you.

Research goal:
{research_goal}

Research plan:
{_json_block(research_plan)}

Top ranked source hypotheses:
{_serialize_hypotheses(top_hypotheses)}

Literature notes:
{_serialize_literature(literature_notes)}

Meta-review feedback and research overview:
{_serialize_feedback(meta_feedback, research_overview, trajectory_state)}

Undercovered focus areas that should be expanded:
{_json_block({"uncovered_focus_areas": uncovered_focus_areas})}

Overexplored areas that should only be revisited with a clearly different mechanism:
{_json_block({"overexplored_areas": overexplored_areas})}

Use at least one of these operator families where appropriate:
- grounded enhancement
- feasibility or coherence repair
- inspiration from adjacent literature
- combination
- simplification
- divergence away from over-explored clusters
- Treat evolution as improvement, not paraphrase: every evolved hypothesis should be expected to rank above at least one of its provided source hypotheses under the shared criteria.
- Do not output a mutation that is only a narrower, more complex, or differently worded version of a parent unless it is clearly better.
- At least one evolved hypothesis should be an ambitious challenger intended to improve on the current top-ranked source, not just a safe refinement.
- Prefer divergence or simplification when the current top hypotheses cluster around the same mechanism family.
- Every evolved hypothesis must cite 1 or 2 `parent_ids` drawn from the provided source hypotheses. Do not leave `parent_ids` empty.
- Make the mutation legible: describe the concrete delta from the parent idea instead of rewriting the same idea with new wording.
- At least one evolved hypothesis should deliberately widen the search frontier when overexplored areas are listed.
- Use the batch as a portfolio: when generating 3 or more mutations, include a mix of repair/simplification and frontier-widening moves instead of only polishing the same family.
- Prefer fixing a specific overhead, invalidating assumption, or experiment weakness that appears in the parent line.
- If `trajectory_state.lineage_coverage` is low, be extra careful to name valid parents and preserve mutation traceability.
- If `trajectory_state.frontier_focus_gaps` is non-empty, use at least one evolved hypothesis to challenge the current leaderboard from one of those missing areas when a plausible parent path exists.
- If multiple evolved hypotheses stay in the same focus area, they must use different mutation operators or different parent combinations.
- Return exactly one top-level JSON object with the key "hypotheses". Do not rename the top-level key to "list", "items", or "results".
- Returning a single evolved hypothesis object instead of a `hypotheses` array is invalid.
- Returning fewer than {num_hypotheses} evolved hypotheses is invalid. Finish the full mutation batch.

Required JSON schema:
{{
  "hypotheses": [
    {{
      "title": "string",
      "focus_area": "string",
      "primary_bottleneck": "compute|memory bandwidth|latency|cache pressure|verification overhead|other",
      "core_hypothesis": "string",
      "mechanism": "string",
      "novelty_rationale": "string",
      "generation_strategy": "grounded_enhancement|repair|inspiration|combination|simplification|divergence",
      "mutation_operator": "grounded_enhancement|repair|inspiration|combination|simplification|divergence",
      "parent_ids": ["string"],
      "delta_from_parents": "string",
      "diversity_reason": "string",
      "key_assumptions": ["string"],
      "predictions": ["string"],
      "test_plan": ["string"],
      "references": ["string"],
      "search_queries": ["string"]
    }}
  ]
}}
"""
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
    ]


def build_evolution_refill_messages(
    research_goal: str,
    research_plan: Dict[str, Any],
    top_hypotheses: List[Dict[str, Any]],
    accepted_hypotheses: List[Dict[str, Any]],
    literature_notes: List[Dict[str, Any]],
    meta_feedback: Dict[str, Any],
    research_overview: Dict[str, Any],
    trajectory_state: Dict[str, Any],
    missing_hypotheses: int,
    uncovered_focus_areas: List[str],
    overexplored_areas: List[str],
) -> List[Dict[str, str]]:
    prompt = f"""
The previous evolution response under-filled the mutation batch.
Produce exactly {missing_hypotheses} additional evolved hypotheses so the frontier reaches the intended size.

Research goal:
{research_goal}

Research plan:
{_json_block(research_plan)}

Top ranked source hypotheses:
{_serialize_hypotheses(top_hypotheses)}

Already accepted evolved hypotheses that must not be repeated:
{_serialize_hypotheses(accepted_hypotheses)}

Literature notes:
{_serialize_literature(literature_notes)}

Meta-review feedback and research overview:
{_serialize_feedback(meta_feedback, research_overview, trajectory_state)}

Undercovered focus areas that should be expanded:
{_json_block({"uncovered_focus_areas": uncovered_focus_areas})}

Overexplored areas that should only be revisited with a clearly different mechanism:
{_json_block({"overexplored_areas": overexplored_areas})}

Requirements:
- Return exactly {missing_hypotheses} additional evolved hypotheses.
- Every hypothesis must cite 1 or 2 `parent_ids` drawn from the provided source hypotheses.
- Do not repeat titles, mechanisms, or parent combinations that already appear in the accepted evolved set unless the mutation delta is materially different.
- Every additional hypothesis must be a credible improvement over its parent hypotheses, not just a variant or paraphrase.
- Prefer divergence toward uncovered focus areas when possible.
- Spread the refill across different focus areas or mutation operators whenever possible.
- Returning a single evolved hypothesis object instead of a `hypotheses` array is invalid.
- Return exactly one top-level JSON object with the key "hypotheses".

Required JSON schema:
{{
  "hypotheses": [
    {{
      "title": "string",
      "focus_area": "string",
      "primary_bottleneck": "compute|memory bandwidth|latency|cache pressure|verification overhead|other",
      "core_hypothesis": "string",
      "mechanism": "string",
      "novelty_rationale": "string",
      "generation_strategy": "grounded_enhancement|repair|inspiration|combination|simplification|divergence",
      "mutation_operator": "grounded_enhancement|repair|inspiration|combination|simplification|divergence",
      "parent_ids": ["string"],
      "delta_from_parents": "string",
      "diversity_reason": "string",
      "key_assumptions": ["string"],
      "predictions": ["string"],
      "test_plan": ["string"],
      "references": ["string"],
      "search_queries": ["string"]
    }}
  ]
}}
"""
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
    ]


def build_meta_review_messages(
    research_goal: str,
    research_plan: Dict[str, Any],
    ranked_hypotheses: List[Dict[str, Any]],
    recent_reviews: List[Dict[str, Any]],
    tournament_history: List[Dict[str, Any]],
    proximity_summary: Dict[str, Any],
    trajectory_state: Dict[str, Any],
    literature_notes: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    prompt = f"""
Perform a meta-review over the current research state.
The job here is not only to pick the best hypothesis, but to identify recurring critique, blind spots, and next-iteration guidance for every agent.

Research goal:
{research_goal}

Research plan:
{_json_block(research_plan)}

Ranked hypotheses:
{_serialize_hypotheses(ranked_hypotheses)}

Recent review artifacts:
{_json_block({"reviews": recent_reviews[-10:]})}

Tournament history:
{_json_block({"matches": tournament_history[-12:]})}

Proximity summary:
{_json_block(proximity_summary)}

Current trajectory diagnostics:
{_json_block(trajectory_state)}

Literature notes:
{_serialize_literature(literature_notes)}

Required JSON schema:
{{
  "meta_review_critique": ["string"],
  "generation_guidance": ["string"],
  "reflection_guidance": ["string"],
  "ranking_guidance": ["string"],
  "research_overview": {{
    "summary": "string",
    "priority_areas": ["string"],
    "top_ranked_hypotheses": ["string"],
    "suggested_next_steps": ["string"],
    "suggested_experiments": ["string"]
  }},
  "expert_profiles": ["string"]
}}
"""
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
    ]
