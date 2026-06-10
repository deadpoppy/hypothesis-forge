from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import config
from .utils import coerce_string_list


DEFAULT_EVALUATION_CRITERIA = {
    "alignment": 0.25,
    "novelty": 0.25,
    "plausibility": 0.25,
    "testability": 0.25,
}


def _dedupe(items: Any) -> List[str]:
    return coerce_string_list(items)


def _stringify_constraint_values(prefix: str, items: Any) -> List[str]:
    values = _dedupe(items)
    if not values:
        return []
    return [f"{prefix}: {value}" for value in values]


def _bounded_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _bounded_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, minimum), maximum)


def _score_value(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(parsed, 1.0), 5.0)


def _normalize_weights(weights: Optional[Dict[str, Any]]) -> Dict[str, float]:
    merged = dict(DEFAULT_EVALUATION_CRITERIA)
    if isinstance(weights, dict):
        for key, value in weights.items():
            key = str(key)
            if key not in DEFAULT_EVALUATION_CRITERIA:
                continue
            try:
                merged[key] = max(0.0, float(value))
            except (TypeError, ValueError):
                continue

    total = sum(merged.values())
    if total <= 0:
        return dict(DEFAULT_EVALUATION_CRITERIA)
    return {key: value / total for key, value in merged.items()}


def _dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _configured_llm_model(profile: str, default: str) -> str:
    prefixes = [profile]
    if profile == "thinking":
        prefixes.extend(["think", "generation"])
    elif profile == "critic":
        prefixes.extend(["review", "reflection", "rank", "ranking"])

    for prefix in prefixes:
        block = config.get(f"{prefix}_llm")
        if isinstance(block, list):
            for item in block:
                if not isinstance(item, dict):
                    continue
                for key in ("model", "llm_model"):
                    value = str(item.get(key) or "").strip()
                    if value:
                        return value
        elif isinstance(block, dict):
            providers = block.get("providers")
            if isinstance(providers, list):
                for item in providers:
                    if not isinstance(item, dict):
                        continue
                    for key in ("model", "llm_model"):
                        value = str(item.get(key) or "").strip()
                        if value:
                            return value
            for key in ("model", "llm_model"):
                value = str(block.get(key) or "").strip()
                if value:
                    return value
        for key in ("model", "llm_model"):
            value = str(config.get(f"{prefix}_{key}") or "").strip()
            if value:
                return value

    value = str(config.get("llm_model") or "").strip()
    return value or default


def _rating_bucket(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    if score >= 4.0:
        return "HIGH"
    if score >= 2.5:
        return "MEDIUM"
    return "LOW"


def _average_score(scores: Dict[str, float]) -> float:
    numeric = [float(value) for value in scores.values() if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else 0.0


@dataclass
class ResearchGoal:
    description: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    reference_arxiv_url: Optional[str] = None
    reference_paper_context: Dict[str, Any] = field(default_factory=dict)
    llm_model: Optional[str] = None
    critic_llm_model: Optional[str] = None
    num_hypotheses: Optional[int] = None
    generation_temperature: Optional[float] = None
    evolution_temperature: Optional[float] = None
    reflection_temperature: Optional[float] = None
    elo_k_factor: Optional[int] = None
    top_k_hypotheses: Optional[int] = None
    max_literature_results: Optional[int] = None
    enable_prior_art_check: Optional[bool] = None
    prior_art_queries_per_idea: Optional[int] = None
    prior_art_results_per_query: Optional[int] = None
    prior_art_embedding_candidates: Optional[int] = None
    prior_art_review_top_k: Optional[int] = None
    prior_art_similarity_threshold: Optional[float] = None
    prior_art_repair_attempts: Optional[int] = None
    ranking_matches_per_cycle: Optional[int] = None
    proximity_similarity_threshold: Optional[float] = None
    hypothesis_decay_fraction: Optional[float] = None
    enable_safety_review: Optional[bool] = None
    max_concurrency: Optional[int] = None

    def __post_init__(self) -> None:
        self.reference_arxiv_url = str(self.reference_arxiv_url or "").strip() or None
        if not isinstance(self.reference_paper_context, dict):
            self.reference_paper_context = {}
        step_temperatures = config.get("step_temperatures", {})
        self.llm_model = self.llm_model or _configured_llm_model("thinking", "mimo-v2.5-pro")
        self.critic_llm_model = self.critic_llm_model or _configured_llm_model("critic", self.llm_model)
        self.num_hypotheses = _bounded_int(
            self.num_hypotheses if self.num_hypotheses is not None else config.get("num_hypotheses", 4),
            default=4,
        )
        self.generation_temperature = _bounded_float(
            self.generation_temperature
            if self.generation_temperature is not None
            else step_temperatures.get("generation", 0.7),
            default=0.7,
            minimum=0.0,
            maximum=2.0,
        )
        self.evolution_temperature = _bounded_float(
            self.evolution_temperature
            if self.evolution_temperature is not None
            else step_temperatures.get("evolution", self.generation_temperature),
            default=self.generation_temperature,
            minimum=0.0,
            maximum=2.0,
        )
        self.reflection_temperature = _bounded_float(
            self.reflection_temperature
            if self.reflection_temperature is not None
            else step_temperatures.get("reflection", 0.3),
            default=0.3,
            minimum=0.0,
            maximum=2.0,
        )
        self.elo_k_factor = _bounded_int(
            self.elo_k_factor if self.elo_k_factor is not None else config.get("elo_k_factor", 32),
            default=32,
        )
        self.top_k_hypotheses = _bounded_int(
            self.top_k_hypotheses if self.top_k_hypotheses is not None else config.get("top_k_hypotheses", 3),
            default=3,
        )
        self.max_literature_results = _bounded_int(
            self.max_literature_results
            if self.max_literature_results is not None
            else config.get("max_literature_results", 5),
            default=5,
            minimum=0,
        )
        if self.enable_prior_art_check is None:
            self.enable_prior_art_check = bool(config.get("enable_prior_art_check", True))
        else:
            self.enable_prior_art_check = bool(self.enable_prior_art_check)
        self.prior_art_queries_per_idea = _bounded_int(
            self.prior_art_queries_per_idea
            if self.prior_art_queries_per_idea is not None
            else config.get("prior_art_queries_per_idea", 6),
            default=6,
        )
        self.prior_art_results_per_query = _bounded_int(
            self.prior_art_results_per_query
            if self.prior_art_results_per_query is not None
            else config.get("prior_art_results_per_query", 50),
            default=50,
        )
        self.prior_art_embedding_candidates = _bounded_int(
            self.prior_art_embedding_candidates
            if self.prior_art_embedding_candidates is not None
            else config.get("prior_art_embedding_candidates", 80),
            default=80,
        )
        self.prior_art_review_top_k = _bounded_int(
            self.prior_art_review_top_k
            if self.prior_art_review_top_k is not None
            else config.get("prior_art_review_top_k", 12),
            default=12,
        )
        self.prior_art_similarity_threshold = _bounded_float(
            self.prior_art_similarity_threshold
            if self.prior_art_similarity_threshold is not None
            else config.get("prior_art_similarity_threshold", 0.38),
            default=0.38,
            minimum=0.0,
            maximum=1.0,
        )
        self.prior_art_repair_attempts = _bounded_int(
            self.prior_art_repair_attempts
            if self.prior_art_repair_attempts is not None
            else config.get("prior_art_repair_attempts", 1),
            default=1,
            minimum=0,
        )
        self.ranking_matches_per_cycle = _bounded_int(
            self.ranking_matches_per_cycle
            if self.ranking_matches_per_cycle is not None
            else config.get("ranking_matches_per_cycle", 8),
            default=8,
        )
        self.proximity_similarity_threshold = _bounded_float(
            self.proximity_similarity_threshold
            if self.proximity_similarity_threshold is not None
            else config.get("proximity_similarity_threshold", 0.55),
            default=0.55,
            minimum=0.0,
            maximum=1.0,
        )
        self.hypothesis_decay_fraction = _bounded_float(
            self.hypothesis_decay_fraction
            if self.hypothesis_decay_fraction is not None
            else config.get("hypothesis_decay_fraction", 0.0),
            default=0.0,
            minimum=0.0,
            maximum=1.0,
        )
        if self.enable_safety_review is None:
            self.enable_safety_review = bool(config.get("enable_safety_review", True))
        else:
            self.enable_safety_review = bool(self.enable_safety_review)
        self.max_concurrency = _bounded_int(
            self.max_concurrency if self.max_concurrency is not None else config.get("max_concurrency", 8),
            default=8,
        )

    def signature(self) -> str:
        return "|".join(
            [
                self.description.strip(),
                str(self.reference_arxiv_url or ""),
                str(sorted(self.constraints.items())),
                str(self.llm_model),
                str(self.critic_llm_model),
                str(self.num_hypotheses),
                str(self.generation_temperature),
                str(self.evolution_temperature),
                str(self.reflection_temperature),
                str(self.elo_k_factor),
                str(self.top_k_hypotheses),
                str(self.max_literature_results),
                str(self.enable_prior_art_check),
                str(self.prior_art_queries_per_idea),
                str(self.prior_art_results_per_query),
                str(self.prior_art_embedding_candidates),
                str(self.prior_art_review_top_k),
                str(self.prior_art_similarity_threshold),
                str(self.prior_art_repair_attempts),
                str(self.ranking_matches_per_cycle),
                str(self.proximity_similarity_threshold),
                str(self.hypothesis_decay_fraction),
                str(self.enable_safety_review),
                str(self.max_concurrency),
            ]
        )

    def prompt_context(self) -> str:
        parts = [self.description.strip()]
        if self.reference_paper_context:
            compact_reference = {
                "arxiv_id": self.reference_paper_context.get("arxiv_id"),
                "arxiv_url": self.reference_paper_context.get("arxiv_url") or self.reference_arxiv_url,
                "title": self.reference_paper_context.get("title"),
                "concise_summary": self.reference_paper_context.get("concise_summary"),
                "core_problem": self.reference_paper_context.get("core_problem"),
                "core_mechanism": self.reference_paper_context.get("core_mechanism"),
                "key_results": self.reference_paper_context.get("key_results", [])[:5],
                "limitations": self.reference_paper_context.get("limitations", [])[:5],
                "reusable_insights_for_new_ideas": self.reference_paper_context.get(
                    "reusable_insights_for_new_ideas",
                    [],
                )[:8],
                "avoid_copying": self.reference_paper_context.get("avoid_copying", [])[:6],
            }
            parts.append("Required reference paper context:\n" + json.dumps(compact_reference, ensure_ascii=False, indent=2))
        elif self.reference_arxiv_url:
            parts.append(f"Required reference paper: {self.reference_arxiv_url}")
        return "\n\n".join(part for part in parts if part)

    def reference_seed_queries(self) -> List[str]:
        if not isinstance(self.reference_paper_context, dict):
            return []
        return _dedupe(self.reference_paper_context.get("seed_queries", []))


@dataclass
class ResearchPlan:
    objective: str
    domain: str = "general research"
    focus_areas: List[str] = field(default_factory=list)
    key_questions: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    evaluation_criteria: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_EVALUATION_CRITERIA))
    constraints: List[str] = field(default_factory=list)
    avoid: List[str] = field(default_factory=list)
    preferred_evidence: List[str] = field(default_factory=list)
    seed_queries: List[str] = field(default_factory=list)
    tool_preferences: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @classmethod
    def from_goal(cls, research_goal: ResearchGoal) -> "ResearchPlan":
        structured_constraints = research_goal.constraints if isinstance(research_goal.constraints, dict) else {}
        coverage_requirement = str(structured_constraints.get("coverage_requirement") or "").strip()
        direction_families = _dedupe(structured_constraints.get("direction_families") or [])
        hard_constraints = _dedupe(structured_constraints.get("hard_constraints") or [])
        avoid = _dedupe(structured_constraints.get("avoid") or [])
        success_metrics = _dedupe(structured_constraints.get("success_metrics") or [])

        constraint_values = [f"{key}: {value}" for key, value in structured_constraints.items() if key not in {
            "coverage_requirement",
            "direction_families",
            "hard_constraints",
            "avoid",
            "success_metrics",
        }]
        if coverage_requirement:
            constraint_values.insert(0, f"coverage_requirement: {coverage_requirement}")
        constraint_values.extend(_stringify_constraint_values("hard_constraint", hard_constraints))

        focus_areas = direction_families or [research_goal.description.strip()]
        success_criteria = [
            "Proposals should be aligned with the research objective.",
            "Proposals should contain a mechanistic explanation and a validation path.",
            "Proposals should be differentiable from generic incremental ideas.",
        ]
        if coverage_requirement:
            success_criteria.append(coverage_requirement)
        success_criteria.extend(success_metrics)

        key_questions = [f"What is the most promising research direction for: {research_goal.description.strip()}?"]
        if direction_families:
            key_questions.append("How can the search stay diverse across the requested mechanism families?")
        if hard_constraints:
            key_questions.append("Which directions best satisfy the explicit no-training and runtime-feasibility constraints?")

        preferred_evidence = [
            "recent literature grounding",
            "mechanistic reasoning",
            "clear experiments or observations",
        ]
        preferred_evidence.extend(success_metrics)

        seed_queries = [research_goal.description.strip()]
        seed_queries.extend(direction_families[:6])
        seed_queries.extend(research_goal.reference_seed_queries())

        notes = ["Fallback research plan generated locally because no structured plan was available."]
        if direction_families:
            notes.append("Constraint-aware fallback preserved the requested direction families for diversity tracking.")
        if research_goal.reference_arxiv_url:
            notes.append(f"Required reference paper: {research_goal.reference_arxiv_url}.")
        if research_goal.reference_paper_context:
            reference_title = str(research_goal.reference_paper_context.get("title") or "").strip()
            if reference_title:
                notes.append(f"Use the reference paper as inspiration, not as a template to copy: {reference_title}.")

        return cls(
            objective=research_goal.description.strip(),
            focus_areas=focus_areas,
            key_questions=key_questions,
            success_criteria=success_criteria,
            evaluation_criteria=dict(DEFAULT_EVALUATION_CRITERIA),
            constraints=constraint_values,
            avoid=avoid,
            preferred_evidence=_dedupe(preferred_evidence),
            seed_queries=_dedupe(seed_queries),
            tool_preferences=["arxiv_search"],
            notes=notes,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], research_goal: ResearchGoal) -> "ResearchPlan":
        fallback = cls.from_goal(research_goal)
        if not isinstance(data, dict):
            return fallback

        focus_areas = _dedupe(data.get("focus_areas") or [])
        key_questions = _dedupe(data.get("key_questions") or [])
        success_criteria = _dedupe(data.get("success_criteria") or [])
        constraints = _dedupe(data.get("constraints") or [])
        avoid = _dedupe(data.get("avoid") or [])
        preferred_evidence = _dedupe(data.get("preferred_evidence") or [])
        seed_queries = _dedupe(data.get("seed_queries") or [])
        tool_preferences = _dedupe(data.get("tool_preferences") or [])
        notes = _dedupe(data.get("notes") or [])

        return cls(
            objective=str(data.get("objective") or fallback.objective),
            domain=str(data.get("domain") or fallback.domain),
            focus_areas=_dedupe(focus_areas + fallback.focus_areas),
            key_questions=_dedupe(key_questions + fallback.key_questions),
            success_criteria=_dedupe(success_criteria + fallback.success_criteria),
            evaluation_criteria=_normalize_weights(data.get("evaluation_criteria")),
            constraints=_dedupe(constraints + fallback.constraints),
            avoid=_dedupe(avoid + fallback.avoid),
            preferred_evidence=_dedupe(preferred_evidence + fallback.preferred_evidence),
            seed_queries=_dedupe(seed_queries + fallback.seed_queries),
            tool_preferences=_dedupe(tool_preferences + fallback.tool_preferences),
            notes=_dedupe(notes + fallback.notes),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "domain": self.domain,
            "focus_areas": self.focus_areas,
            "key_questions": self.key_questions,
            "success_criteria": self.success_criteria,
            "evaluation_criteria": self.evaluation_criteria,
            "constraints": self.constraints,
            "avoid": self.avoid,
            "preferred_evidence": self.preferred_evidence,
            "seed_queries": self.seed_queries,
            "tool_preferences": self.tool_preferences,
            "notes": self.notes,
        }


@dataclass
class Hypothesis:
    hypothesis_id: str
    title: str
    text: str
    focus_area: str = ""
    primary_bottleneck: str = ""
    rationale: str = ""
    mechanism: str = ""
    problem_framing: str = ""
    central_insight: str = ""
    theoretical_story: str = ""
    why_not_simple_combination: str = ""
    generation_strategy: str = "generation"
    mutation_operator: str = ""
    evolution_delta: str = ""
    origin: str = "generation"
    parent_ids: List[str] = field(default_factory=list)
    predictions: List[str] = field(default_factory=list)
    key_assumptions: List[str] = field(default_factory=list)
    validation_experiments: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    supporting_observations: List[str] = field(default_factory=list)
    contradicting_observations: List[str] = field(default_factory=list)
    improvement_actions: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    novelty_review: Optional[str] = None
    feasibility_review: Optional[str] = None
    correctness_review: Optional[str] = None
    testability_review: Optional[str] = None
    review_verdict: Optional[str] = None
    review_summary: str = ""
    scores: Dict[str, float] = field(default_factory=dict)
    elo_score: float = 1200.0
    review_comments: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    literature_notes: List[Dict[str, Any]] = field(default_factory=list)
    prior_art_signature: Optional[str] = None
    prior_art_audit: Dict[str, Any] = field(default_factory=dict)
    prior_art_similar_papers: List[Dict[str, Any]] = field(default_factory=list)
    prior_art_repair_count: int = 0
    debate_history: List[Dict[str, Any]] = field(default_factory=list)
    review_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    codex_rerank_reviews: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    created_in_iteration: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hypothesis":
        if not isinstance(data, dict):
            data = {}

        scores: Dict[str, float] = {}
        raw_scores = data.get("scores")
        if isinstance(raw_scores, dict):
            for key, value in raw_scores.items():
                try:
                    scores[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        hypothesis_id = str(data.get("id") or data.get("hypothesis_id") or "restored_hypothesis").strip()
        try:
            elo_score = float(data.get("elo_score", 1200.0))
        except (TypeError, ValueError):
            elo_score = 1200.0
        try:
            prior_art_repair_count = int(data.get("prior_art_repair_count", 0))
        except (TypeError, ValueError):
            prior_art_repair_count = 0
        try:
            created_in_iteration = int(data.get("created_in_iteration", 0))
        except (TypeError, ValueError):
            created_in_iteration = 0

        return cls(
            hypothesis_id=hypothesis_id,
            title=str(data.get("title") or "Untitled hypothesis").strip(),
            text=str(data.get("text") or "").strip(),
            focus_area=str(data.get("focus_area") or "").strip(),
            primary_bottleneck=str(data.get("primary_bottleneck") or "").strip(),
            rationale=str(data.get("rationale") or "").strip(),
            mechanism=str(data.get("mechanism") or "").strip(),
            problem_framing=str(data.get("problem_framing") or "").strip(),
            central_insight=str(data.get("central_insight") or "").strip(),
            theoretical_story=str(data.get("theoretical_story") or "").strip(),
            why_not_simple_combination=str(data.get("why_not_simple_combination") or "").strip(),
            generation_strategy=str(data.get("generation_strategy") or "generation").strip(),
            mutation_operator=str(data.get("mutation_operator") or "").strip(),
            evolution_delta=str(data.get("evolution_delta") or "").strip(),
            origin=str(data.get("origin") or "generation").strip(),
            parent_ids=coerce_string_list(data.get("parent_ids", [])),
            predictions=coerce_string_list(data.get("predictions", [])),
            key_assumptions=coerce_string_list(data.get("key_assumptions", [])),
            validation_experiments=coerce_string_list(data.get("validation_experiments", [])),
            failure_modes=coerce_string_list(data.get("failure_modes", [])),
            supporting_observations=coerce_string_list(data.get("supporting_observations", [])),
            contradicting_observations=coerce_string_list(data.get("contradicting_observations", [])),
            improvement_actions=coerce_string_list(data.get("improvement_actions", [])),
            search_queries=coerce_string_list(data.get("search_queries", [])),
            novelty_review=data.get("novelty_review"),
            feasibility_review=data.get("feasibility_review"),
            correctness_review=data.get("correctness_review"),
            testability_review=data.get("testability_review"),
            review_verdict=data.get("review_verdict"),
            review_summary=str(data.get("review_summary") or ""),
            scores=scores,
            elo_score=elo_score,
            review_comments=coerce_string_list(data.get("review_comments", [])),
            references=coerce_string_list(data.get("references", [])),
            literature_notes=_dict_list(data.get("literature_notes")),
            prior_art_signature=data.get("prior_art_signature"),
            prior_art_audit=_mapping(data.get("prior_art_audit")),
            prior_art_similar_papers=_dict_list(data.get("prior_art_similar_papers")),
            prior_art_repair_count=prior_art_repair_count,
            debate_history=_dict_list(data.get("debate_history")),
            review_artifacts=_dict_list(data.get("review_artifacts")),
            codex_rerank_reviews=_dict_list(data.get("codex_rerank_reviews")),
            is_active=bool(data.get("is_active", True)),
            created_in_iteration=max(0, created_in_iteration),
        )

    def apply_review(self, review: Dict[str, Any], stage_label: str) -> None:
        if not isinstance(review, dict):
            return

        score_mapping = {
            "alignment": "alignment_score",
            "novelty": "novelty_score",
            "plausibility": "plausibility_score",
            "testability": "testability_score",
            "feasibility": "feasibility_score",
            "correctness": "correctness_score",
            "story_coherence": "story_coherence_score",
            "theoretical_depth": "theoretical_depth_score",
            "non_combination": "non_combination_score",
        }

        for key, source_key in score_mapping.items():
            value = _score_value(review.get(source_key))
            if value is not None:
                self.scores[key] = value

        self.novelty_review = _rating_bucket(self.scores.get("novelty"))
        self.feasibility_review = _rating_bucket(self.scores.get("feasibility") or self.scores.get("plausibility"))
        self.correctness_review = _rating_bucket(self.scores.get("correctness") or self.scores.get("alignment"))
        self.testability_review = _rating_bucket(self.scores.get("testability"))

        summary = str(review.get("short_summary") or review.get("summary") or "").strip()
        if summary:
            self.review_summary = summary

        verdict = str(review.get("verdict") or "").strip().lower()
        if bool(review.get("reject")) and not verdict:
            verdict = "reject"
        if verdict:
            self.review_verdict = verdict
            if verdict.startswith("reject"):
                self.is_active = False

        comments = coerce_string_list(review.get("strengths", []))
        comments.extend(coerce_string_list(review.get("weaknesses", [])))
        comments.extend(coerce_string_list(review.get("improvement_actions", [])))
        for free_text_key in ("reflection", "feasibility_rationale", "reference_connection"):
            free_text = str(review.get(free_text_key) or "").strip()
            if free_text:
                comments.append(f"{stage_label}: {free_text}")
        story_diagnosis = review.get("story_diagnosis") if isinstance(review.get("story_diagnosis"), dict) else {}
        combination_risk = str(story_diagnosis.get("combination_risk") or "").strip().lower()
        if combination_risk in {"medium", "high"}:
            comments.append(f"{stage_label}: {combination_risk} method-stacking risk.")
        if summary:
            comments.append(f"{stage_label}: {summary}")
        self.review_comments = _dedupe(self.review_comments + comments)

        self.key_assumptions = _dedupe(self.key_assumptions + coerce_string_list(review.get("critical_assumptions", [])))
        self.validation_experiments = _dedupe(
            self.validation_experiments + coerce_string_list(review.get("validation_experiments", []))
        )
        self.failure_modes = _dedupe(self.failure_modes + coerce_string_list(review.get("failure_modes", [])))
        self.supporting_observations = _dedupe(
            self.supporting_observations + coerce_string_list(review.get("supporting_observations", []))
        )
        self.contradicting_observations = _dedupe(
            self.contradicting_observations + coerce_string_list(review.get("contradicting_observations", []))
        )
        self.improvement_actions = _dedupe(
            self.improvement_actions
            + coerce_string_list(review.get("improvement_actions", []))
            + coerce_string_list(story_diagnosis.get("missing_theory", []))
            + coerce_string_list(story_diagnosis.get("repair_to_story", []))
        )
        self.references = _dedupe(self.references + coerce_string_list(review.get("references", [])))

    def overall_score(self) -> float:
        return _average_score(self.scores)

    def compact_summary(self) -> Dict[str, Any]:
        return {
            "id": self.hypothesis_id,
            "title": self.title,
            "focus_area": self.focus_area,
            "primary_bottleneck": self.primary_bottleneck,
            "problem_framing": self.problem_framing,
            "central_insight": self.central_insight,
            "theoretical_story": self.theoretical_story,
            "why_not_simple_combination": self.why_not_simple_combination,
            "summary": self.review_summary or self.text,
            "elo_score": round(self.elo_score, 2),
            "scores": self.scores,
            "strategy": self.generation_strategy,
            "mutation_operator": self.mutation_operator,
            "evolution_delta": self.evolution_delta,
            "parent_ids": self.parent_ids,
            "verdict": self.review_verdict,
            "latest_review": self.review_artifacts[-1] if self.review_artifacts else {},
            "latest_codex_rerank": self.codex_rerank_reviews[-1] if self.codex_rerank_reviews else {},
        }

    def comparison_signature(self) -> str:
        payload = {
            "title": self.title,
            "text": self.text,
            "focus_area": self.focus_area,
            "primary_bottleneck": self.primary_bottleneck,
            "rationale": self.rationale,
            "mechanism": self.mechanism,
            "problem_framing": self.problem_framing,
            "central_insight": self.central_insight,
            "theoretical_story": self.theoretical_story,
            "why_not_simple_combination": self.why_not_simple_combination,
            "generation_strategy": self.generation_strategy,
            "mutation_operator": self.mutation_operator,
            "evolution_delta": self.evolution_delta,
            "origin": self.origin,
            "parent_ids": self.parent_ids,
            "predictions": self.predictions,
            "key_assumptions": self.key_assumptions,
            "validation_experiments": self.validation_experiments,
            "failure_modes": self.failure_modes,
            "supporting_observations": self.supporting_observations,
            "contradicting_observations": self.contradicting_observations,
            "improvement_actions": self.improvement_actions,
            "review_verdict": self.review_verdict,
            "review_summary": self.review_summary,
            "scores": self.scores,
            "references": self.references,
            "review_artifacts": self.review_artifacts[-3:],
            "codex_rerank_reviews": self.codex_rerank_reviews[-3:],
            "is_active": self.is_active,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def idea_signature(self) -> str:
        payload = {
            "title": self.title,
            "text": self.text,
            "focus_area": self.focus_area,
            "primary_bottleneck": self.primary_bottleneck,
            "rationale": self.rationale,
            "mechanism": self.mechanism,
            "problem_framing": self.problem_framing,
            "central_insight": self.central_insight,
            "theoretical_story": self.theoretical_story,
            "why_not_simple_combination": self.why_not_simple_combination,
            "generation_strategy": self.generation_strategy,
            "mutation_operator": self.mutation_operator,
            "evolution_delta": self.evolution_delta,
            "origin": self.origin,
            "parent_ids": self.parent_ids,
            "predictions": self.predictions,
            "key_assumptions": self.key_assumptions,
            "validation_experiments": self.validation_experiments,
            "search_queries": self.search_queries,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.hypothesis_id,
            "title": self.title,
            "text": self.text,
            "focus_area": self.focus_area,
            "primary_bottleneck": self.primary_bottleneck,
            "rationale": self.rationale,
            "mechanism": self.mechanism,
            "problem_framing": self.problem_framing,
            "central_insight": self.central_insight,
            "theoretical_story": self.theoretical_story,
            "why_not_simple_combination": self.why_not_simple_combination,
            "generation_strategy": self.generation_strategy,
            "mutation_operator": self.mutation_operator,
            "evolution_delta": self.evolution_delta,
            "origin": self.origin,
            "parent_ids": self.parent_ids,
            "predictions": self.predictions,
            "key_assumptions": self.key_assumptions,
            "validation_experiments": self.validation_experiments,
            "failure_modes": self.failure_modes,
            "supporting_observations": self.supporting_observations,
            "contradicting_observations": self.contradicting_observations,
            "improvement_actions": self.improvement_actions,
            "search_queries": self.search_queries,
            "novelty_review": self.novelty_review,
            "feasibility_review": self.feasibility_review,
            "correctness_review": self.correctness_review,
            "testability_review": self.testability_review,
            "review_verdict": self.review_verdict,
            "review_summary": self.review_summary,
            "scores": self.scores,
            "elo_score": self.elo_score,
            "review_comments": self.review_comments,
            "references": self.references,
            "literature_notes": self.literature_notes,
            "prior_art_signature": self.prior_art_signature,
            "prior_art_audit": self.prior_art_audit,
            "prior_art_similar_papers": self.prior_art_similar_papers,
            "prior_art_repair_count": self.prior_art_repair_count,
            "debate_history": self.debate_history,
            "review_artifacts": self.review_artifacts,
            "codex_rerank_reviews": self.codex_rerank_reviews,
            "is_active": self.is_active,
            "created_in_iteration": self.created_in_iteration,
        }


@dataclass
class StepResult:
    name: str
    hypotheses: List[Hypothesis] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.data)
        payload["hypotheses"] = [hypothesis.to_dict() for hypothesis in self.hypotheses]
        if self.errors:
            payload["errors"] = self.errors
        if self.duration:
            payload["duration"] = self.duration
        return payload


@dataclass
class ContextMemory:
    hypotheses: Dict[str, Hypothesis] = field(default_factory=dict)
    tournament_results: List[Dict[str, Any]] = field(default_factory=list)
    codex_rerank_history: List[Dict[str, Any]] = field(default_factory=list)
    meta_review_feedback: List[Dict[str, Any]] = field(default_factory=list)
    research_overviews: List[Dict[str, Any]] = field(default_factory=list)
    cycle_history: List[Dict[str, Any]] = field(default_factory=list)
    reference_paper_context: Dict[str, Any] = field(default_factory=dict)
    research_plan: Optional[ResearchPlan] = None
    iteration_number: int = 0
    goal_signature: Optional[str] = None
    last_cycle_statistics: Dict[str, Any] = field(default_factory=dict)
    pairwise_decisions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], research_goal: ResearchGoal) -> "ContextMemory":
        context = cls()
        if not isinstance(data, dict):
            context.reset_for_goal(research_goal)
            return context

        raw_hypotheses = data.get("hypotheses", {})
        if isinstance(raw_hypotheses, dict):
            for key, value in raw_hypotheses.items():
                if not isinstance(value, dict):
                    continue
                restored = Hypothesis.from_dict({**value, "id": value.get("id") or key})
                context.add_hypothesis(restored)
        elif isinstance(raw_hypotheses, list):
            for value in raw_hypotheses:
                if isinstance(value, dict):
                    context.add_hypothesis(Hypothesis.from_dict(value))

        context.tournament_results = _dict_list(data.get("tournament_results"))
        context.codex_rerank_history = _dict_list(data.get("codex_rerank_history"))
        context.meta_review_feedback = _dict_list(data.get("meta_review_feedback"))
        context.research_overviews = _dict_list(data.get("research_overviews"))
        context.cycle_history = _dict_list(data.get("cycle_history"))
        context.reference_paper_context = _mapping(data.get("reference_paper_context")) or dict(
            research_goal.reference_paper_context
        )

        research_plan = data.get("research_plan")
        context.research_plan = ResearchPlan.from_dict(research_plan, research_goal) if isinstance(research_plan, dict) else None

        try:
            context.iteration_number = max(0, int(data.get("iteration_number", 0)))
        except (TypeError, ValueError):
            context.iteration_number = 0

        context.goal_signature = str(data.get("goal_signature") or research_goal.signature())
        context.last_cycle_statistics = _mapping(data.get("last_cycle_statistics"))

        raw_pairwise_decisions = data.get("pairwise_decisions")
        if isinstance(raw_pairwise_decisions, dict):
            context.pairwise_decisions = {
                str(key): value
                for key, value in raw_pairwise_decisions.items()
                if isinstance(value, dict)
            }

        return context

    def reset_for_goal(self, research_goal: ResearchGoal) -> None:
        self.hypotheses.clear()
        self.tournament_results.clear()
        self.codex_rerank_history.clear()
        self.meta_review_feedback.clear()
        self.research_overviews.clear()
        self.cycle_history.clear()
        self.reference_paper_context = dict(research_goal.reference_paper_context)
        self.research_plan = None
        self.iteration_number = 0
        self.goal_signature = research_goal.signature()
        self.last_cycle_statistics = {}
        self.pairwise_decisions.clear()

    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis

    def get_active_hypotheses(self) -> List[Hypothesis]:
        return [hypothesis for hypothesis in self.hypotheses.values() if hypothesis.is_active]

    def get_ranked_hypotheses(self) -> List[Hypothesis]:
        return sorted(self.hypotheses.values(), key=lambda item: item.elo_score, reverse=True)

    def latest_meta_review(self) -> Dict[str, Any]:
        return self.meta_review_feedback[-1] if self.meta_review_feedback else {}

    def latest_research_overview(self) -> Dict[str, Any]:
        return self.research_overviews[-1] if self.research_overviews else {}

    def summarize_hypotheses(self, limit: int = 8) -> List[Dict[str, Any]]:
        ranked = self.get_ranked_hypotheses()[:limit]
        return [hypothesis.compact_summary() for hypothesis in ranked]

    def compute_statistics(self) -> Dict[str, Any]:
        all_hypotheses = list(self.hypotheses.values())
        active_hypotheses = self.get_active_hypotheses()
        scores = [hypothesis.overall_score() for hypothesis in all_hypotheses if hypothesis.overall_score() > 0]
        strategies: Dict[str, int] = {}
        origins: Dict[str, int] = {}
        focus_counts: Dict[str, int] = {}
        for hypothesis in all_hypotheses:
            strategies[hypothesis.generation_strategy] = strategies.get(hypothesis.generation_strategy, 0) + 1
            origins[hypothesis.origin] = origins.get(hypothesis.origin, 0) + 1
            focus_label = (hypothesis.focus_area or hypothesis.title).strip()
            if focus_label:
                focus_counts[focus_label] = focus_counts.get(focus_label, 0) + 1

        evolved = [hypothesis for hypothesis in all_hypotheses if hypothesis.origin == "evolution"]
        with_valid_parents = [hypothesis for hypothesis in evolved if hypothesis.parent_ids]
        dominant_focus_area = ""
        dominant_focus_area_share = 0.0
        if focus_counts:
            dominant_focus_area, dominant_count = max(focus_counts.items(), key=lambda item: item[1])
            dominant_focus_area_share = round(dominant_count / len(all_hypotheses), 3)

        return {
            "total_hypotheses": len(all_hypotheses),
            "active_hypotheses": len(active_hypotheses),
            "rejected_hypotheses": len([hypothesis for hypothesis in all_hypotheses if not hypothesis.is_active]),
            "average_elo": round(
                sum(hypothesis.elo_score for hypothesis in all_hypotheses) / len(all_hypotheses), 2
            ) if all_hypotheses else 0.0,
            "average_review_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            "strategies": strategies,
            "origins": origins,
            "unique_focus_areas": len(focus_counts),
            "dominant_focus_area": dominant_focus_area,
            "dominant_focus_area_share": dominant_focus_area_share,
            "valid_evolution_lineages": len(with_valid_parents),
            "missing_evolution_lineages": max(0, len(evolved) - len(with_valid_parents)),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypotheses": {key: hypothesis.to_dict() for key, hypothesis in self.hypotheses.items()},
            "tournament_results": self.tournament_results,
            "codex_rerank_history": self.codex_rerank_history,
            "meta_review_feedback": self.meta_review_feedback,
            "research_overviews": self.research_overviews,
            "cycle_history": self.cycle_history,
            "reference_paper_context": self.reference_paper_context,
            "research_plan": self.research_plan.to_dict() if self.research_plan else None,
            "iteration_number": self.iteration_number,
            "goal_signature": self.goal_signature,
            "last_cycle_statistics": self.last_cycle_statistics,
            "pairwise_decisions": self.pairwise_decisions,
        }


class ResearchGoalRequest(BaseModel):
    description: str
    constraints: Dict[str, Any] = Field(default_factory=dict)
    reference_arxiv_url: str
    reference_paper_context: Dict[str, Any] = Field(default_factory=dict)
    llm_model: Optional[str] = None
    critic_llm_model: Optional[str] = None
    num_hypotheses: Optional[int] = None
    generation_temperature: Optional[float] = None
    evolution_temperature: Optional[float] = None
    reflection_temperature: Optional[float] = None
    elo_k_factor: Optional[int] = None
    top_k_hypotheses: Optional[int] = None
    max_literature_results: Optional[int] = None
    enable_prior_art_check: Optional[bool] = None
    prior_art_queries_per_idea: Optional[int] = None
    prior_art_results_per_query: Optional[int] = None
    prior_art_embedding_candidates: Optional[int] = None
    prior_art_review_top_k: Optional[int] = None
    prior_art_similarity_threshold: Optional[float] = None
    prior_art_repair_attempts: Optional[int] = None
    ranking_matches_per_cycle: Optional[int] = None
    proximity_similarity_threshold: Optional[float] = None
    hypothesis_decay_fraction: Optional[float] = None
    enable_safety_review: Optional[bool] = None
    max_concurrency: Optional[int] = None


class HypothesisResponse(BaseModel):
    id: str
    title: str
    text: str
    focus_area: str
    elo_score: float
    review_summary: str
    novelty_review: Optional[str]
    feasibility_review: Optional[str]
    correctness_review: Optional[str]
    testability_review: Optional[str]
    review_verdict: Optional[str]
    review_comments: List[str]
    references: List[str]
    is_active: bool


class OverviewResponse(BaseModel):
    iteration: int
    meta_review_critique: List[str]
    top_hypotheses: List[HypothesisResponse]
    suggested_next_steps: List[str]


class ArxivSearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10
    categories: Optional[List[str]] = None
    sort_by: Optional[str] = "relevance"
    days_back: Optional[int] = None


class ArxivPaper(BaseModel):
    arxiv_id: str
    entry_id: str
    title: str
    abstract: str
    authors: List[str]
    primary_category: str
    categories: List[str]
    published: Optional[str]
    updated: Optional[str]
    doi: Optional[str]
    pdf_url: str
    arxiv_url: str
    comment: Optional[str]
    journal_ref: Optional[str]
    source: str = "arxiv"


class ArxivSearchResponse(BaseModel):
    query: str
    total_results: int
    papers: List[ArxivPaper]
    search_time_ms: Optional[float]


class ArxivTrendsResponse(BaseModel):
    query: str
    total_papers: int
    date_range: str
    top_categories: List[tuple]
    top_authors: List[tuple]
    papers: List[ArxivPaper]
