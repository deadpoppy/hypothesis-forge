from __future__ import annotations

import math
import time
from typing import Callable, Dict, List

from .agents import (
    EvolutionAgent,
    GenerationAgent,
    IdeaLiteratureAgent,
    MetaReviewAgent,
    OpenCodeRerankingAgent,
    ProximityAgent,
    RankingAgent,
    ReflectionAgent,
    ResearchPlanAgent,
)
from .models import ContextMemory, ResearchGoal, StepResult
from .utils import logger


class SupervisorAgent:
    """Paper-aligned orchestrator for the AI co-scientist workflow."""

    def __init__(self) -> None:
        self.research_plan_agent = ResearchPlanAgent()
        self.generation_agent = GenerationAgent()
        self.idea_literature_agent = IdeaLiteratureAgent()
        self.reflection_agent = ReflectionAgent()
        self.proximity_agent = ProximityAgent()
        self.ranking_agent = RankingAgent()
        self.opencode_reranking_agent = OpenCodeRerankingAgent()
        self.evolution_agent = EvolutionAgent()
        self.meta_review_agent = MetaReviewAgent()

    def run_cycle(
        self,
        research_goal: ResearchGoal,
        context: ContextMemory,
        progress_callback: Callable[[str, Dict, ContextMemory], None] | None = None,
    ) -> Dict:
        cycle_start = time.time()
        iteration = context.iteration_number + 1
        logger.info("--- Starting cycle %d ---", iteration)

        if context.goal_signature != research_goal.signature():
            context.reset_for_goal(research_goal)
            iteration = 1
            logger.info("Goal signature changed. Context reset for new research goal.")

        cycle_details = {
            "iteration": iteration,
            "reference_paper": context.reference_paper_context or research_goal.reference_paper_context,
            "research_plan": context.research_plan.to_dict() if context.research_plan else {},
            "steps": {},
            "errors": [],
            "statistics_before": context.compute_statistics(),
        }

        def emit_progress(step_name: str) -> None:
            if progress_callback is not None:
                progress_callback(step_name, cycle_details, context)

        if context.research_plan is None:
            context.research_plan = self.research_plan_agent.create_plan(research_goal)
        cycle_details["research_plan"] = context.research_plan.to_dict()

        cycle_details["steps"]["research_plan"] = {
            "hypotheses": [],
            "plan": context.research_plan.to_dict(),
            "duration": 0.0,
        }
        emit_progress("research_plan")

        generation_result = self.generation_agent.generate_new_hypotheses(research_goal, context)
        self._add_step(cycle_details, generation_result)
        for hypothesis in generation_result.hypotheses:
            context.add_hypothesis(hypothesis)
        emit_progress("generation")

        idea_literature_result = self.idea_literature_agent.check_hypotheses(
            generation_result.hypotheses,
            research_goal,
            context,
            step_name="idea_literature",
        )
        self._add_step(cycle_details, idea_literature_result)
        emit_progress("idea_literature")

        generation_review_candidates = [hypothesis for hypothesis in generation_result.hypotheses if hypothesis.is_active]
        review_result = self.reflection_agent.review_hypotheses(
            generation_review_candidates,
            research_goal,
            context,
            step_name="review",
        )
        self._add_step(cycle_details, review_result)
        emit_progress("review")

        proximity_before_result = self.proximity_agent.build_proximity_graph(
            research_goal=research_goal,
            context=context,
            step_name="proximity_pre_ranking",
        )
        self._add_step(cycle_details, proximity_before_result)
        emit_progress("proximity_pre_ranking")

        ranking_result = self.ranking_agent.run_tournament(
            research_goal=research_goal,
            context=context,
            proximity_data=proximity_before_result.data,
            step_name="ranking",
        )
        self._add_step(cycle_details, ranking_result)
        emit_progress("ranking")

        opencode_reranking_result = self.opencode_reranking_agent.rerank_top_hypotheses(
            research_goal=research_goal,
            context=context,
            step_name="opencode_reranking",
        )
        self._add_step(cycle_details, opencode_reranking_result)
        emit_progress("opencode_reranking")

        evolution_result = self.evolution_agent.evolve_hypotheses(research_goal, context)
        self._add_step(cycle_details, evolution_result)
        for hypothesis in evolution_result.hypotheses:
            context.add_hypothesis(hypothesis)
        emit_progress("evolution")

        replacement_pruning_result = self._prune_replaced_hypotheses(research_goal, context, evolution_result.hypotheses)
        self._add_step(cycle_details, replacement_pruning_result)
        emit_progress("evolution_replacement_pruning")

        evolved_idea_literature = self.idea_literature_agent.check_hypotheses(
            evolution_result.hypotheses,
            research_goal,
            context,
            step_name="evolved_idea_literature",
        )
        self._add_step(cycle_details, evolved_idea_literature)
        emit_progress("evolved_idea_literature")

        evolution_review_candidates = [hypothesis for hypothesis in evolution_result.hypotheses if hypothesis.is_active]
        review_evolved = self.reflection_agent.review_hypotheses(
            evolution_review_candidates,
            research_goal,
            context,
            step_name="review_evolved",
        )
        self._add_step(cycle_details, review_evolved)
        emit_progress("review_evolved")

        proximity_final_result = self.proximity_agent.build_proximity_graph(
            research_goal=research_goal,
            context=context,
            step_name="proximity_final",
        )
        self._add_step(cycle_details, proximity_final_result)
        emit_progress("proximity_final")

        ranking_final_result = self.ranking_agent.run_tournament(
            research_goal=research_goal,
            context=context,
            proximity_data=proximity_final_result.data,
            step_name="ranking_final",
        )
        self._add_step(cycle_details, ranking_final_result)
        emit_progress("ranking_final")

        opencode_reranking_final = self.opencode_reranking_agent.rerank_top_hypotheses(
            research_goal=research_goal,
            context=context,
            step_name="opencode_reranking_final",
        )
        self._add_step(cycle_details, opencode_reranking_final)
        emit_progress("opencode_reranking_final")

        meta_review_result = self.meta_review_agent.summarize_and_feedback(
            research_goal=research_goal,
            context=context,
            proximity_data=proximity_final_result.data,
        )
        self._add_step(cycle_details, meta_review_result)
        cycle_details["meta_review"] = meta_review_result.data
        emit_progress("meta_review")

        decay_result = self._decay_low_elo_hypotheses(research_goal, context)
        self._add_step(cycle_details, decay_result)
        emit_progress("frontier_decay")

        context.iteration_number += 1
        context.last_cycle_statistics = context.compute_statistics()
        cycle_details["statistics_after"] = context.last_cycle_statistics
        cycle_details["cycle_duration"] = time.time() - cycle_start
        context.cycle_history.append(
            {
                "iteration": context.iteration_number,
                "top_hypotheses": [hypothesis.compact_summary() for hypothesis in context.get_ranked_hypotheses()[:5]],
                "meta_review": meta_review_result.data,
            }
        )

        logger.info("--- Cycle %d complete ---", context.iteration_number)
        emit_progress("cycle_complete")
        return cycle_details

    def _prune_replaced_hypotheses(
        self,
        research_goal: ResearchGoal,
        context: ContextMemory,
        new_hypotheses: List,
    ) -> StepResult:
        active_new_ids = {
            hypothesis.hypothesis_id
            for hypothesis in new_hypotheses
            if hypothesis.is_active
        }
        target_prune_count = len(active_new_ids)
        active_old = sorted(
            [
                hypothesis
                for hypothesis in context.get_active_hypotheses()
                if hypothesis.hypothesis_id not in active_new_ids
            ],
            key=lambda hypothesis: (hypothesis.elo_score, hypothesis.created_in_iteration, hypothesis.hypothesis_id),
        )
        prune_count = min(target_prune_count, len(active_old))
        pruned_hypotheses = active_old[:prune_count]

        for hypothesis in pruned_hypotheses:
            hypothesis.is_active = False
            hypothesis.review_verdict = "replaced_by_evolution"
            hypothesis.review_comments = list(
                dict.fromkeys(
                    hypothesis.review_comments
                    + [
                        (
                            "evolution_replacement_pruning: deactivated because each newly evolved idea "
                            "replaces one low-ranked active idea."
                        )
                    ]
                )
            )

        return StepResult(
            name="evolution_replacement_pruning",
            hypotheses=pruned_hypotheses,
            data={
                "new_active_evolved_count": target_prune_count,
                "requested_prune_count": target_prune_count,
                "pruned_count": len(pruned_hypotheses),
                "active_old_available": len(active_old),
                "pruned_hypotheses": [hypothesis.compact_summary() for hypothesis in pruned_hypotheses],
                "policy": "prune_one_lowest_ranked_old_active_hypothesis_per_new_evolved_hypothesis",
            },
        )

    def _decay_low_elo_hypotheses(self, research_goal: ResearchGoal, context: ContextMemory) -> StepResult:
        active_hypotheses = sorted(
            context.get_active_hypotheses(),
            key=lambda hypothesis: (hypothesis.elo_score, hypothesis.created_in_iteration, hypothesis.hypothesis_id),
        )
        fraction = max(0.0, min(1.0, float(research_goal.hypothesis_decay_fraction)))
        minimum_survivors = max(2, research_goal.top_k_hypotheses)
        pruned_count = 0
        if fraction > 0 and len(active_hypotheses) > minimum_survivors:
            requested_count = math.ceil(len(active_hypotheses) * fraction)
            pruned_count = min(requested_count, len(active_hypotheses) - minimum_survivors)

        pruned_hypotheses = active_hypotheses[:pruned_count]
        for hypothesis in pruned_hypotheses:
            hypothesis.is_active = False
            hypothesis.review_verdict = "decayed_low_elo"
            hypothesis.review_comments = list(
                dict.fromkeys(
                    hypothesis.review_comments
                    + [
                        (
                            "frontier_decay: deactivated at cycle end because it was in the "
                            f"lowest Elo {fraction:.2%} of the active frontier."
                        )
                    ]
                )
            )

        return StepResult(
            name="frontier_decay",
            hypotheses=pruned_hypotheses,
            data={
                "decay_fraction": fraction,
                "minimum_survivors": minimum_survivors,
                "active_before": len(active_hypotheses),
                "active_after": len(active_hypotheses) - len(pruned_hypotheses),
                "pruned_count": len(pruned_hypotheses),
                "pruned_hypotheses": [hypothesis.compact_summary() for hypothesis in pruned_hypotheses],
            },
        )

    def _add_step(self, cycle_details: Dict, step_result: StepResult, step_name: str | None = None) -> None:
        name = step_name or step_result.name
        cycle_details["steps"][name] = step_result.to_dict()
        cycle_details["errors"].extend(step_result.errors)
