import itertools
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.agents import (
    EvolutionAgent,
    GenerationAgent,
    MetaReviewAgent,
    OpenCodeRerankingAgent,
    PriorArtAgent,
    ProximityAgent,
    RankingAgent,
    ReflectionAgent,
    _extract_hypothesis_items,
    _evolution_source_budget_for_cycle,
    _evolution_temperature_for_cycle,
    _focus_area_query_variants,
    _generation_temperature_for_cycle,
    _literature_budget_for_step,
    _prioritize_diverse_hypotheses,
    _round_robin_query_bundle,
    _select_evolution_sources,
    _trajectory_state_snapshot,
)
from app.literature import LiteratureCache, LiteratureSearchService, dedupe_notes
import app.llm as llm_module
import app.reference as reference_module
from app.llm import call_llm
from app.models import ContextMemory, Hypothesis, ResearchGoal, ResearchPlan
from app.prompts import _serialize_literature, build_evolution_messages, build_generation_messages, build_meta_review_messages
from app.reference import ReferencePaperError, normalize_arxiv_reference
from app.reports import build_markdown_report
from app.tools.arxiv_search import ArxivSearchTool
from app.tools.semantic_scholar import SemanticScholarSearchTool
from app.trajectory import analyze_run_payload
from app.utils import coerce_string_list, embedding_slot, run_concurrently, similarity_score
from app.vector_index import PaperVectorIndex, paper_recall_text
from app.workflow import SupervisorAgent


def _load_cli_module():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("co_scientist_cli_under_test", repo_root / "app.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CoreBehaviorTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        sentence_model_patch = patch("app.utils.get_sentence_transformer_model", side_effect=RuntimeError("offline"))
        sentence_model_patch.start()
        self.addCleanup(sentence_model_patch.stop)

    def test_cli_missing_constraints_reports_clear_input_error(self):
        result = subprocess.run(
            [
                sys.executable,
                "app.py",
                "--goal",
                "goal",
                "--reference-arxiv",
                "https://arxiv.org/abs/2502.18864",
                "--constraints",
                "run_configs/does_not_exist.json",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            env={**os.environ, "LLM_API_KEY": "placeholder"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Input error: Constraints file not found", result.stdout)
        self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_cli_invalid_constraints_json_reports_clear_input_error(self):
        with TemporaryDirectory() as temp_dir:
            bad_path = Path(temp_dir) / "bad_constraints.json"
            bad_path.write_text("{not valid json", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "app.py",
                    "--goal",
                    "goal",
                    "--reference-arxiv",
                    "https://arxiv.org/abs/2502.18864",
                    "--constraints",
                    str(bad_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                env={**os.environ, "LLM_API_KEY": "placeholder"},
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Input error: Constraints file is not valid JSON", result.stdout)
        self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_cli_missing_api_key_fails_fast_with_clear_message(self):
        repo_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                """
logging_level: INFO
openai_api_key: null
openrouter_base_url: null
thinking_llm:
  providers:
    - model: test-model
critic_llm:
  providers:
    - model: test-model
""".strip(),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "app.py"),
                    "--goal",
                    "goal",
                    "--reference-arxiv",
                    "https://arxiv.org/abs/2502.18864",
                    "--cycles",
                    "1",
                ],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                env={key: value for key, value in os.environ.items() if key not in {"LLM_API_KEY", "OPENROUTER_API_KEY"}},
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Input error: LLM API key is not configured", result.stdout)
        self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_context_memory_round_trips_from_saved_payload(self):
        goal = ResearchGoal("resume goal", enable_safety_review=False)
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)
        context.iteration_number = 1
        context.last_cycle_statistics = {"total_hypotheses": 1}
        context.pairwise_decisions = {"H1::H2": {"winner_id": "H1"}}
        context.add_hypothesis(
            Hypothesis(
                "H1",
                "Restored Idea",
                "idea text",
                focus_area="runtime routing",
                scores={"novelty": 4.0},
                elo_score=1325.5,
                literature_notes=[{"title": "Paper"}],
                created_in_iteration=1,
            )
        )

        restored = ContextMemory.from_dict(context.to_dict(), goal)

        self.assertEqual(restored.goal_signature, goal.signature())
        self.assertEqual(restored.iteration_number, 1)
        self.assertEqual(restored.research_plan.objective, goal.description)
        self.assertEqual(restored.hypotheses["H1"].title, "Restored Idea")
        self.assertEqual(restored.hypotheses["H1"].scores["novelty"], 4.0)
        self.assertEqual(restored.pairwise_decisions["H1::H2"]["winner_id"], "H1")

    def test_context_memory_migrates_legacy_codex_rerank_fields(self):
        goal = ResearchGoal("resume goal", enable_safety_review=False)
        payload = {
            "hypotheses": {
                "H1": {
                    "id": "H1",
                    "title": "Legacy Idea",
                    "text": "idea text",
                    "codex_rerank_reviews": [{"rank": 1, "summary": "legacy review"}],
                }
            },
            "codex_rerank_history": [{"status": "applied", "ranking": [{"id": "H1", "rank": 1}]}],
        }

        restored = ContextMemory.from_dict(payload, goal)

        self.assertEqual(restored.hypotheses["H1"].opencode_rerank_reviews[0]["rank"], 1)
        self.assertEqual(restored.opencode_rerank_history[0]["status"], "applied")
        self.assertIn("opencode_rerank_reviews", restored.to_dict()["hypotheses"]["H1"])
        self.assertIn("opencode_rerank_history", restored.to_dict())

    def test_resume_state_loads_matching_checkpoint_only(self):
        cli = _load_cli_module()
        goal = ResearchGoal("resume goal", num_hypotheses=2, enable_safety_review=False)
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)
        context.iteration_number = 1
        context.add_hypothesis(Hypothesis("H1", "Saved Idea", "idea text", created_in_iteration=1))

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            payload = {
                "metadata": {"cycle_count": 2, "goal_signature": goal.signature()},
                "cycles": [
                    {"iteration": 1, "steps": {}, "errors": [], "statistics_after": {}, "cycle_duration": 0.1},
                    {"iteration": 2, "steps": {"generation": {"hypotheses": []}}, "errors": []},
                ],
                "final_context": context.to_dict(),
            }
            (output_dir / "checkpoint_latest.json").write_text(json.dumps(payload), encoding="utf-8")

            state = cli._load_resume_state(str(output_dir), goal)
            mismatch = cli._load_resume_state(str(output_dir), ResearchGoal("different goal", num_hypotheses=2, enable_safety_review=False))

        self.assertIsNotNone(state)
        self.assertEqual(len(state["cycles"]), 1)
        self.assertEqual(state["partial_cycle_count"], 1)
        self.assertEqual(state["context"].hypotheses["H1"].title, "Saved Idea")
        self.assertIsNone(mismatch)

    def test_run_cycles_auto_resumes_to_requested_total(self):
        cli = _load_cli_module()

        class FakeSupervisor:
            def __init__(self):
                self.calls = 0

            def run_cycle(self, research_goal, context, progress_callback=None):
                self.calls += 1
                iteration = context.iteration_number + 1
                hypothesis = Hypothesis(f"H{iteration}", f"Idea {iteration}", "idea text", created_in_iteration=iteration)
                context.add_hypothesis(hypothesis)
                context.iteration_number = iteration
                context.last_cycle_statistics = context.compute_statistics()
                cycle = {
                    "iteration": iteration,
                    "research_plan": context.research_plan.to_dict() if context.research_plan else {},
                    "steps": {"generation": {"hypotheses": [hypothesis.to_dict()]}},
                    "errors": [],
                    "statistics_after": context.last_cycle_statistics,
                    "cycle_duration": 0.1,
                }
                if progress_callback is not None:
                    progress_callback("cycle_complete", cycle, context)
                return cycle

        with TemporaryDirectory() as temp_dir:
            args = cli.build_parser().parse_args(
                [
                    "--goal",
                    "resume goal",
                    "--reference-arxiv",
                    "https://arxiv.org/abs/2502.18864",
                    "--cycles",
                    "2",
                    "--output-dir",
                    temp_dir,
                    "--num-hypotheses",
                    "2",
                    "--disable-safety-review",
                ]
            )
            goal = cli._build_research_goal(args, {})
            context = ContextMemory()
            context.reset_for_goal(goal)
            context.research_plan = ResearchPlan.from_goal(goal)
            context.iteration_number = 1
            context.add_hypothesis(Hypothesis("H1", "Saved Idea", "idea text", created_in_iteration=1))
            existing_cycle = {
                "iteration": 1,
                "research_plan": context.research_plan.to_dict(),
                "steps": {},
                "errors": [],
                "statistics_after": context.compute_statistics(),
                "cycle_duration": 0.1,
            }
            (Path(temp_dir) / "checkpoint_latest.json").write_text(
                json.dumps(
                    {
                        "metadata": {"cycle_count": 1, "goal_signature": goal.signature()},
                        "cycles": [existing_cycle],
                        "final_context": context.to_dict(),
                    }
                ),
                encoding="utf-8",
            )
            fake_supervisor = FakeSupervisor()

            with patch.object(cli, "_require_llm_api_key"), patch.object(
                cli,
                "prepare_reference_paper",
                return_value={"arxiv_id": "2502.18864", "arxiv_url": "https://arxiv.org/abs/2502.18864"},
            ), patch.object(cli, "SupervisorAgent", return_value=fake_supervisor):
                written = cli.run_cycles(args)

            report = json.loads(Path(written["json"]).read_text(encoding="utf-8"))

        self.assertEqual(fake_supervisor.calls, 1)
        self.assertEqual([cycle["iteration"] for cycle in report["cycles"]], [1, 2])
        self.assertIn("H1", report["final_context"]["hypotheses"])
        self.assertIn("H2", report["final_context"]["hypotheses"])

    def test_string_list_coercion_does_not_split_strings(self):
        self.assertEqual(coerce_string_list("one complete item"), ["one complete item"])
        self.assertEqual(coerce_string_list(["A", "a", "B"]), ["A", "B"])

    def test_research_plan_from_dict_accepts_string_fields(self):
        plan = ResearchPlan.from_dict({"focus_areas": "single focus"}, ResearchGoal("goal"))
        self.assertEqual(plan.focus_areas[0], "single focus")

    def test_fallback_research_plan_preserves_direction_family_constraints(self):
        goal = ResearchGoal(
            "goal",
            constraints={
                "coverage_requirement": "Cover multiple mechanism families.",
                "direction_families": ["KV cache", "Speculative decoding"],
                "hard_constraints": ["No retraining."],
                "avoid": ["No duplicates."],
                "success_metrics": ["Latency reduction"],
            },
        )

        plan = ResearchPlan.from_goal(goal)

        self.assertEqual(plan.focus_areas[:2], ["KV cache", "Speculative decoding"])
        self.assertIn("Cover multiple mechanism families.", plan.success_criteria)
        self.assertIn("Latency reduction", plan.success_criteria)
        self.assertIn("hard_constraint: No retraining.", plan.constraints)
        self.assertIn("No duplicates.", plan.avoid)

    def test_research_plan_from_dict_merges_llm_output_with_fallback_constraints(self):
        goal = ResearchGoal(
            "goal",
            constraints={
                "direction_families": ["KV cache", "Speculative decoding"],
                "avoid": ["No duplicates."],
            },
        )

        plan = ResearchPlan.from_dict(
            {
                "focus_areas": ["Kernel fusion"],
                "avoid": ["Avoid training"],
                "seed_queries": ["kernel fusion inference"],
            },
            goal,
        )

        self.assertIn("Kernel fusion", plan.focus_areas)
        self.assertIn("KV cache", plan.focus_areas)
        self.assertIn("Speculative decoding", plan.focus_areas)
        self.assertIn("Avoid training", plan.avoid)
        self.assertIn("No duplicates.", plan.avoid)

    def test_research_goal_supports_separate_evolution_temperature(self):
        goal = ResearchGoal("goal", generation_temperature=0.9, evolution_temperature=0.65)
        self.assertEqual(goal.generation_temperature, 0.9)
        self.assertEqual(goal.evolution_temperature, 0.65)

    def test_research_goal_signature_includes_reference_arxiv(self):
        first = ResearchGoal("goal", reference_arxiv_url="https://arxiv.org/abs/2502.18864")
        second = ResearchGoal("goal", reference_arxiv_url="https://arxiv.org/abs/2501.00001")

        self.assertNotEqual(first.signature(), second.signature())

    def test_normalize_arxiv_reference_accepts_url_and_id(self):
        arxiv_id, url = normalize_arxiv_reference("https://arxiv.org/pdf/2502.18864v1")

        self.assertEqual(arxiv_id, "2502.18864v1")
        self.assertEqual(url, "https://arxiv.org/abs/2502.18864v1")

    def test_reference_markdown_conversion_uses_installed_arxiv2md_package(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            Path(command[-1]).write_text("# Converted\n\nBody from installed package", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with patch.object(reference_module.subprocess, "run", side_effect=fake_run):
            markdown = reference_module._convert_arxiv_to_markdown(
                "https://arxiv.org/abs/2502.18864",
                "2502.18864",
            )

        self.assertIn("# Converted", markdown)
        self.assertIn("Body from installed package", markdown)
        self.assertEqual(calls[0][0], sys.executable)
        self.assertEqual(calls[0][1], "-c")
        self.assertIn("from arxiv2md import ingest_paper_sync", calls[0][2])

    def test_reference_markdown_import_error_mentions_installed_arxiv2md(self):
        command = [sys.executable, "-c", "", "https://arxiv.org/abs/2502.18864", "/tmp/missing.md"]
        completed = subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="arxiv2md is not importable in this Python environment.",
        )

        with patch.object(reference_module.subprocess, "run", return_value=completed):
            with self.assertRaises(ReferencePaperError) as ctx:
                reference_module._convert_arxiv_to_markdown(
                    "https://arxiv.org/abs/2502.18864",
                    "2502.18864",
                )

        self.assertIn("installed arxiv2md", str(ctx.exception))
        self.assertIn("not importable", str(ctx.exception))

    def test_generation_temperature_cap_stays_below_overheated_range(self):
        goal = ResearchGoal("goal", generation_temperature=0.88)
        context = ContextMemory()
        temperature = _generation_temperature_for_cycle(
            goal,
            context,
            {"uncovered_focus_areas": ["new focus"], "overexplored_areas": []},
        )

        self.assertEqual(temperature, 0.92)

    def test_trajectory_state_snapshot_exposes_stagnation_signals(self):
        goal = ResearchGoal("goal", constraints={"direction_families": ["KV cache", "Speculative decoding", "Kernel fusion"]})
        plan = ResearchPlan.from_goal(goal)
        context = ContextMemory()
        context.research_plan = plan
        context.add_hypothesis(Hypothesis("H1", "KV A", "kv reuse", focus_area="KV cache", origin="generation", elo_score=1350, is_active=True))
        context.add_hypothesis(Hypothesis("H2", "KV B", "kv eviction", focus_area="KV cache", origin="generation", elo_score=1320, is_active=True))
        context.add_hypothesis(Hypothesis("H3", "KV C", "kv sharing", focus_area="KV cache", origin="evolution", elo_score=1300, is_active=True))
        context.add_hypothesis(Hypothesis("H4", "Spec A", "spec decode", focus_area="Speculative decoding", origin="evolution", parent_ids=[], elo_score=1280, is_active=True))
        context.cycle_history = [
            {"top_hypotheses": [{"title": "KV A"}]},
            {"top_hypotheses": [{"title": "KV A"}]},
        ]

        coverage = {"uncovered_focus_areas": ["Kernel fusion"], "overexplored_areas": ["KV cache"]}
        snapshot = _trajectory_state_snapshot(plan, context, coverage)

        self.assertIn("coverage_gap", snapshot["stagnation_signals"])
        self.assertIn("frontier_clustered", snapshot["stagnation_signals"])
        self.assertIn("lineage_loss", snapshot["stagnation_signals"])
        self.assertIn("Kernel fusion", snapshot["frontier_focus_gaps"])

    def test_evolution_temperature_adapts_to_trajectory_state(self):
        goal = ResearchGoal("goal", evolution_temperature=0.68)
        temperature, adjustments = _evolution_temperature_for_cycle(
            goal,
            {
                "stagnation_signals": ["coverage_gap", "frontier_clustered", "top1_stable", "lineage_loss"],
                "lineage_coverage": 0.5,
            },
        )

        self.assertAlmostEqual(temperature, 0.72, places=6)
        self.assertIn("coverage_gap:+0.03", adjustments)
        self.assertIn("lineage_loss:-0.04", adjustments)

    def test_evolution_source_budget_expands_under_stagnation(self):
        goal = ResearchGoal("goal", top_k_hypotheses=4)
        budget, adjustments = _evolution_source_budget_for_cycle(
            goal,
            ranked_count=10,
            trajectory_state={"stagnation_signals": ["coverage_gap", "frontier_clustered", "lineage_loss"]},
        )

        self.assertEqual(budget, 6)
        self.assertIn("coverage_gap:+1", adjustments)

    def test_review_scores_are_clamped_without_safety_dimension(self):
        hypothesis = Hypothesis("H1", "Title", "Hypothesis text")
        hypothesis.apply_review({"novelty_score": "9", "verdict": "revise"}, "unit")

        self.assertEqual(hypothesis.scores["novelty"], 5.0)
        self.assertEqual(hypothesis.review_verdict, "revise")
        self.assertTrue(hypothesis.is_active)

    def test_explicit_reject_verdict_deactivates_hypothesis(self):
        hypothesis = Hypothesis("H1", "Title", "Hypothesis text")
        hypothesis.apply_review({"verdict": "reject"}, "unit")

        self.assertEqual(hypothesis.review_verdict, "reject")
        self.assertFalse(hypothesis.is_active)

    def test_proximity_graph_keeps_isolated_nodes(self):
        context = ContextMemory()
        context.add_hypothesis(Hypothesis("H1", "A", "alpha"))
        context.add_hypothesis(Hypothesis("H2", "B", "beta"))

        with patch("app.agents.similarity_score", return_value=0.1):
            result = ProximityAgent().build_proximity_graph(
                ResearchGoal("goal", proximity_similarity_threshold=0.5),
                context,
                "proximity",
            )

        self.assertEqual(set(result.data["adjacency_graph"]), {"H1", "H2"})
        self.assertEqual({node["id"] for node in result.data["nodes"]}, {"H1", "H2"})
        self.assertEqual(sorted(result.data["clusters"]), [["H1"], ["H2"]])

    def test_ranking_pair_selection_prioritizes_new_challenger_against_top(self):
        hypotheses = [
            Hypothesis("H1", "A", "alpha", created_in_iteration=1, elo_score=1300),
            Hypothesis("H2", "B", "beta", created_in_iteration=1, elo_score=1250),
            Hypothesis("H3", "C", "gamma", created_in_iteration=2, elo_score=1200),
        ]
        adjacency = {
            "H1": [{"other_id": "H2", "similarity": 0.2}, {"other_id": "H3", "similarity": 0.1}],
            "H2": [{"other_id": "H1", "similarity": 0.2}, {"other_id": "H3", "similarity": 0.1}],
            "H3": [{"other_id": "H1", "similarity": 0.1}, {"other_id": "H2", "similarity": 0.1}],
        }

        context = ContextMemory()
        for hypothesis in hypotheses:
            context.add_hypothesis(hypothesis)
        pairs = RankingAgent()._select_pairs(hypotheses, adjacency, max_matches=1, similarity_threshold=0.55, context=context)

        self.assertEqual({pairs[0][0].hypothesis_id, pairs[0][1].hypothesis_id}, {"H1", "H3"})

    def test_ranking_pair_selection_round_robins_fresh_evolved_challengers(self):
        hypotheses = [
            Hypothesis("H1", "Top", "alpha", created_in_iteration=1, elo_score=1500),
            Hypothesis("H2", "Parent A", "beta", created_in_iteration=1, elo_score=1400),
            Hypothesis("H3", "Parent B", "gamma", created_in_iteration=1, elo_score=1300),
            Hypothesis("E1", "Mutant A", "delta", origin="evolution", parent_ids=["H2"], created_in_iteration=2),
            Hypothesis("E2", "Mutant B", "epsilon", origin="evolution", parent_ids=["H3"], created_in_iteration=2),
            Hypothesis("E3", "Mutant C", "zeta", origin="evolution", created_in_iteration=2),
        ]
        context = ContextMemory()
        for hypothesis in hypotheses:
            context.add_hypothesis(hypothesis)

        pairs = RankingAgent()._select_pairs(hypotheses, {}, max_matches=3, similarity_threshold=0.55, context=context)
        pair_sets = [{pair[0].hypothesis_id, pair[1].hypothesis_id} for pair in pairs]
        fresh_ids_seen = {hypothesis_id for pair in pair_sets for hypothesis_id in pair if hypothesis_id.startswith("E")}

        self.assertEqual(fresh_ids_seen, {"E1", "E2", "E3"})
        self.assertIn({"E1", "H2"}, pair_sets)
        self.assertIn({"E2", "H3"}, pair_sets)
        self.assertIn({"E3", "H1"}, pair_sets)

    def test_evolution_source_selection_keeps_diverse_challenger(self):
        ranked = [
            Hypothesis("H1", "KV Cache A", "kv cache sharing", focus_area="KV cache sharing", elo_score=1330, created_in_iteration=1),
            Hypothesis("H2", "KV Cache B", "kv cache reuse", focus_area="KV cache sharing", elo_score=1320, created_in_iteration=2),
            Hypothesis("H3", "Spec Draft", "speculative decoding with shifts", focus_area="Speculative decoding", elo_score=1280, created_in_iteration=3),
            Hypothesis("H4", "Kernel Fusion", "tensor core fusion path", focus_area="Kernel fusion", elo_score=1240, created_in_iteration=4),
        ]

        selected = _select_evolution_sources(ranked, source_budget=3)

        self.assertEqual(selected[0].hypothesis_id, "H1")
        self.assertIn("H3", {hypothesis.hypothesis_id for hypothesis in selected})

    def test_evolution_source_selection_prefers_frontier_gap_challenger(self):
        ranked = [
            Hypothesis("H1", "KV Cache A", "kv cache sharing", focus_area="KV cache sharing", elo_score=1360, created_in_iteration=1),
            Hypothesis("H2", "KV Cache B", "kv cache reuse", focus_area="KV cache sharing", elo_score=1340, created_in_iteration=2),
            Hypothesis("H3", "Spec Draft", "speculative decoding with shifts", focus_area="Speculative decoding", elo_score=1270, created_in_iteration=3),
            Hypothesis("H4", "Runtime Router", "route requests across decode paths", focus_area="Runtime routing", elo_score=1240, created_in_iteration=4),
        ]

        selected = _select_evolution_sources(
            ranked,
            source_budget=3,
            preferred_focus_areas=["Runtime routing"],
            penalized_focus_areas=["KV cache sharing"],
        )

        self.assertIn("H4", {hypothesis.hypothesis_id for hypothesis in selected})

    def test_portfolio_prioritization_prefers_gaps_and_penalizes_clusters(self):
        hypotheses = [
            Hypothesis("H1", "KV Cache A", "kv sharing", focus_area="KV cache sharing", primary_bottleneck="cache pressure"),
            Hypothesis("H2", "KV Cache B", "kv eviction", focus_area="KV cache sharing", primary_bottleneck="memory bandwidth"),
            Hypothesis("H3", "Runtime Router", "route requests", focus_area="Runtime routing", primary_bottleneck="latency"),
            Hypothesis("H4", "Kernel Fusion", "fuse kernels", focus_area="Kernel fusion", primary_bottleneck="compute"),
        ]

        selected = _prioritize_diverse_hypotheses(
            hypotheses,
            target_count=3,
            preferred_focus_areas=["Runtime routing"],
            penalized_focus_areas=["KV cache sharing"],
        )

        selected_ids = [hypothesis.hypothesis_id for hypothesis in selected]
        self.assertEqual(selected_ids[0], "H3")
        self.assertIn("H4", selected_ids)
        self.assertLess(selected_ids.count("H1") + selected_ids.count("H2"), 2)

    def test_literature_budget_trims_mature_clustered_cycle(self):
        goal = ResearchGoal("goal", max_literature_results=5)
        query_budget, max_results, adjustments = _literature_budget_for_step(
            goal,
            trajectory_state={
                "uncovered_focus_areas": [],
                "frontier_focus_gaps": [],
                "overexplored_areas": ["KV cache", "Speculative decoding"],
            },
            step_name="generation",
            iteration_number=3,
        )

        self.assertEqual(query_budget, 2)
        self.assertEqual(max_results, 3)
        self.assertIn("mature_frontier_trim", adjustments)

    def test_focus_area_query_variants_add_search_friendly_runtime_routing_terms(self):
        variants = _focus_area_query_variants("runtime routing or input-adaptive inference control")

        self.assertIn("runtime adaptive inference routing llm", variants)
        self.assertIn("input adaptive inference control transformer serving", variants)

    def test_generation_query_builder_expands_focus_area_variants(self):
        goal = ResearchGoal("goal")
        plan = ResearchPlan.from_goal(goal)
        agent = GenerationAgent()
        hypothesis = Hypothesis(
            "H1",
            "Adaptive Router",
            "route requests dynamically",
            focus_area="runtime routing or input-adaptive inference control",
        )

        queries = agent._build_queries(goal, plan, hypothesis)

        self.assertIn("Adaptive Router", queries)
        self.assertIn("runtime adaptive inference routing llm", queries)

    def test_round_robin_query_bundle_spreads_across_focus_areas_first(self):
        queries = _round_robin_query_bundle(
            [
                ["focus_a_raw", "focus_a_alt"],
                ["focus_b_raw", "focus_b_alt"],
                ["focus_c_raw", "focus_c_alt"],
            ],
            limit=3,
        )

        self.assertEqual(queries, ["focus_a_raw", "focus_b_raw", "focus_c_raw"])

    def test_ranking_reuses_cached_pairwise_decision_without_recounting_elo(self):
        goal = ResearchGoal("goal", ranking_matches_per_cycle=1, max_literature_results=0)
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)
        winner = Hypothesis("H1", "A", "alpha")
        loser = Hypothesis("H2", "B", "beta")
        context.add_hypothesis(winner)
        context.add_hypothesis(loser)
        adjacency = {
            "H1": [{"other_id": "H2", "similarity": 0.8}],
            "H2": [{"other_id": "H1", "similarity": 0.8}],
        }

        with patch(
            "app.agents.call_json",
            return_value={
                "winner_id": "H1",
                "loser_id": "H2",
                "comparison_summary": "H1 wins",
                "winner_reason": "stronger",
                "confidence": 0.7,
            },
        ) as mocked_call:
            ranker = RankingAgent()
            first = ranker.run_tournament(goal, context, {"adjacency_graph": adjacency}, "ranking")
            first_winner_elo = winner.elo_score
            first_loser_elo = loser.elo_score
            second = ranker.run_tournament(goal, context, {"adjacency_graph": adjacency}, "ranking_final")

        self.assertEqual(len(first.data["matches"]), 1)
        self.assertFalse(first.data["matches"][0]["cached"])
        self.assertEqual(len(second.data["matches"]), 1)
        self.assertTrue(second.data["matches"][0]["cached"])
        self.assertEqual(second.data["matches"][0]["winner"], "H1")
        self.assertEqual(second.data["matches"][0]["loser"], "H2")
        self.assertFalse(second.data["matches"][0]["counted_for_elo"])
        self.assertEqual(second.data["matches"][0]["adjudication_source"], "cache_replay")
        self.assertEqual(winner.elo_score, first_winner_elo)
        self.assertEqual(loser.elo_score, first_loser_elo)
        self.assertEqual(second.data["elo_counted_matches"], 0)
        self.assertEqual(second.data["skipped_cached_pairs"], 1)
        self.assertEqual(mocked_call.call_count, 1)
        self.assertEqual(len(context.tournament_results), 1)

    def test_cached_pairwise_decisions_do_not_consume_new_match_budget(self):
        goal = ResearchGoal("goal", ranking_matches_per_cycle=1, max_literature_results=0)
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)
        context.add_hypothesis(Hypothesis("H1", "A", "alpha", elo_score=1300))
        context.add_hypothesis(Hypothesis("H2", "B", "beta", elo_score=1200))
        adjacency = {
            "H1": [{"other_id": "H2", "similarity": 0.8}],
            "H2": [{"other_id": "H1", "similarity": 0.8}],
        }

        call_payloads = [
            {
                "winner_id": "H1",
                "loser_id": "H2",
                "comparison_summary": "H1 beats H2",
                "winner_reason": "stronger",
                "confidence": 0.7,
            },
            {
                "winner_id": "H1",
                "loser_id": "H3",
                "comparison_summary": "H1 beats H3",
                "winner_reason": "still stronger",
                "confidence": 0.7,
            },
        ]

        with patch("app.agents.call_json", side_effect=call_payloads) as mocked_call:
            ranker = RankingAgent()
            ranker.run_tournament(goal, context, {"adjacency_graph": adjacency}, "ranking")
            context.add_hypothesis(Hypothesis("H3", "C", "gamma", elo_score=1250))
            expanded_adjacency = {
                "H1": [{"other_id": "H2", "similarity": 0.8}, {"other_id": "H3", "similarity": 0.7}],
                "H2": [{"other_id": "H1", "similarity": 0.8}, {"other_id": "H3", "similarity": 0.2}],
                "H3": [{"other_id": "H1", "similarity": 0.7}, {"other_id": "H2", "similarity": 0.2}],
            }
            second = ranker.run_tournament(goal, context, {"adjacency_graph": expanded_adjacency}, "ranking_final")

        self.assertEqual(mocked_call.call_count, 2)
        self.assertEqual(len(second.data["matches"]), 2)
        self.assertEqual(sum(1 for match in second.data["matches"] if match["cached"]), 1)
        self.assertEqual(sum(1 for match in second.data["matches"] if not match["cached"]), 1)
        self.assertEqual(second.data["cached_pairs_considered"], 1)
        self.assertEqual(second.data["new_pairs_considered"], 1)
        self.assertEqual(second.data["elo_counted_matches"], 1)
        self.assertEqual(sum(1 for match in second.data["matches"] if match["counted_for_elo"]), 1)
        self.assertEqual(second.data["skipped_cached_pairs"], 1)
        self.assertEqual(second.data["match_budget"], 1)

    def test_cached_replays_do_not_take_full_debate_slot_from_fresh_challenger(self):
        goal = ResearchGoal("goal", ranking_matches_per_cycle=1, max_literature_results=0)
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)
        context.add_hypothesis(Hypothesis("H1", "Top", "alpha", elo_score=1400, created_in_iteration=1))
        context.add_hypothesis(Hypothesis("H2", "Old", "beta", elo_score=1300, created_in_iteration=1))

        call_payloads = [
            {
                "winner_id": "H1",
                "loser_id": "H2",
                "comparison_summary": "H1 beats H2",
                "winner_reason": "stronger",
                "confidence": 0.7,
            },
            {
                "winner_id": "E1",
                "loser_id": "H1",
                "comparison_summary": "E1 challenges the top idea",
                "winner_reason": "more novel mutation",
                "confidence": 0.7,
            },
        ]

        with patch("app.agents.call_json", side_effect=call_payloads):
            ranker = RankingAgent()
            ranker.run_tournament(goal, context, {"adjacency_graph": {}}, "ranking")
            context.add_hypothesis(
                Hypothesis("E1", "Mutant", "gamma", origin="evolution", parent_ids=["H1"], created_in_iteration=2)
            )
            second = ranker.run_tournament(goal, context, {"adjacency_graph": {}}, "ranking_final")

        self.assertEqual(len(second.data["matches"]), 2)
        self.assertFalse(second.data["matches"][0]["cached"])
        self.assertEqual(second.data["matches"][0]["mode"], "full_debate")
        self.assertEqual(second.data["matches"][0]["scheduled_because"], "fresh_challenger_vs_frontier")
        self.assertEqual({second.data["matches"][0]["winner"], second.data["matches"][0]["loser"]}, {"E1", "H1"})
        self.assertTrue(second.data["matches"][0]["counted_for_elo"])
        self.assertTrue(second.data["matches"][1]["cached"])
        self.assertFalse(second.data["matches"][1]["counted_for_elo"])
        self.assertEqual(second.data["fresh_new_pairs_considered"], 1)

    def test_cached_top_pair_reaudit_uses_distinct_prompt_and_updates_cache(self):
        goal = ResearchGoal("goal", ranking_matches_per_cycle=1, max_literature_results=0, max_concurrency=1)
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)
        h1 = Hypothesis("H1", "Top", "alpha", elo_score=1300, created_in_iteration=1)
        h2 = Hypothesis("H2", "Challenger", "beta", elo_score=1290, created_in_iteration=1)
        context.add_hypothesis(h1)
        context.add_hypothesis(h2)

        call_payloads = [
            {
                "winner_id": "H1",
                "loser_id": "H2",
                "comparison_summary": "Initial decision",
                "winner_reason": "H1 looked stronger",
                "confidence": 0.8,
            },
            {
                "winner_id": "H2",
                "loser_id": "H1",
                "comparison_summary": "Re-audit flips the result",
                "winner_reason": "H2 was underweighted",
                "confidence": 0.75,
            },
        ]

        with patch("app.agents.call_json", side_effect=call_payloads) as mocked_call:
            ranker = RankingAgent()
            ranker.run_tournament(goal, context, {"adjacency_graph": {}}, "ranking")
            context.iteration_number = 1
            second = ranker.run_tournament(goal, context, {"adjacency_graph": {}}, "ranking")

        self.assertEqual(mocked_call.call_count, 2)
        reaudit_prompt = mocked_call.call_args_list[1].args[0][1]["content"]
        self.assertIn("Re-audit this cached tournament decision", reaudit_prompt)
        self.assertEqual(second.data["matches"][0]["adjudication_source"], "cache_reaudit")
        self.assertTrue(second.data["matches"][0]["cached"])
        self.assertTrue(second.data["matches"][0]["counted_for_elo"])
        self.assertEqual(second.data["reaudited_pairs_considered"], 1)
        self.assertEqual(context.pairwise_decisions["H1::H2"]["winner"], "H2")
        self.assertEqual(len(context.tournament_results), 2)

    def test_reflection_and_ranking_use_critic_profile(self):
        goal = ResearchGoal(
            "goal",
            max_literature_results=0,
            ranking_matches_per_cycle=1,
            max_concurrency=1,
            critic_llm_model="critic-model",
        )
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)
        h1 = Hypothesis("H1", "Idea A", "alpha", elo_score=1300)
        h2 = Hypothesis("H2", "Idea B", "beta", elo_score=1200)
        context.add_hypothesis(h1)
        context.add_hypothesis(h2)

        with patch(
            "app.agents.call_json",
            return_value={
                "alignment_score": 4,
                "novelty_score": 4,
                "plausibility_score": 4,
                "testability_score": 4,
                "verdict": "advance",
                "short_summary": "Critic review.",
            },
        ) as mocked_review:
            ReflectionAgent().review_hypotheses([h1], goal, context)

        self.assertEqual(mocked_review.call_args.kwargs["profile"], "critic")
        self.assertEqual(mocked_review.call_args.kwargs["model"], "critic-model")

        with patch(
            "app.agents.call_json",
            return_value={
                "winner_id": "H1",
                "loser_id": "H2",
                "comparison_summary": "H1 wins",
                "winner_reason": "stronger",
                "confidence": 0.7,
            },
        ) as mocked_ranking:
            RankingAgent().run_tournament(goal, context, {"adjacency_graph": {}}, "ranking")

        self.assertEqual(mocked_ranking.call_args.kwargs["profile"], "critic")
        self.assertEqual(mocked_ranking.call_args.kwargs["model"], "critic-model")

    def test_random_cached_reaudit_selection_is_stable_and_capped(self):
        goal = ResearchGoal("goal", ranking_matches_per_cycle=10, max_literature_results=0)
        context = ContextMemory()
        context.reset_for_goal(goal)
        ranker = RankingAgent()
        hypotheses = [Hypothesis(f"H{i}", f"Idea {i}", f"text {i}", created_in_iteration=1) for i in range(6)]
        for hypothesis in hypotheses:
            context.add_hypothesis(hypothesis)
        for first, second in itertools.combinations(hypotheses, 2):
            context.pairwise_decisions[ranker._pair_key(first, second)] = {
                "state_signature": ranker._pair_state_signature(first, second),
                "hypothesis_ids": sorted([first.hypothesis_id, second.hypothesis_id]),
                "payload": {
                    "winner_id": first.hypothesis_id,
                    "loser_id": second.hypothesis_id,
                    "comparison_summary": "cached",
                    "winner_reason": "cached",
                    "confidence": 0.8,
                },
                "winner": first.hypothesis_id,
                "loser": second.hypothesis_id,
                "confidence": 0.8,
                "created_iteration": 1,
            }
        pairs = [(first, second, 0.0) for first, second in itertools.combinations(hypotheses, 2)]
        context.iteration_number = 1

        first_selection = ranker._select_reaudit_keys(pairs, context, goal, "ranking", top_ids=set(), fresh_ids=set())
        second_selection = ranker._select_reaudit_keys(pairs, context, goal, "ranking", top_ids=set(), fresh_ids=set())

        self.assertEqual(first_selection, second_selection)
        self.assertLessEqual(len(first_selection), 1)

    def test_ranking_final_budget_does_not_contract_when_frontier_is_large_and_cached(self):
        goal = ResearchGoal("goal", ranking_matches_per_cycle=10)
        hypotheses = [
            Hypothesis(f"H{i}", f"Idea {i}", f"text {i}", elo_score=1400 - i * 5)
            for i in range(17)
        ]

        budget = RankingAgent()._match_budget(
            research_goal=goal,
            active_hypotheses=hypotheses,
            skipped_cached_pairs=23,
            step_name="ranking_final",
        )

        self.assertEqual(budget, 10)

    def test_evolution_infers_parent_ids_when_model_omits_them(self):
        goal = ResearchGoal("goal", top_k_hypotheses=2, max_literature_results=0, evolution_temperature=0.6)
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)
        context.add_hypothesis(
            Hypothesis("H1", "KV Sharing", "share kv cache across nearby tokens", focus_area="KV cache sharing", elo_score=1320)
        )
        context.add_hypothesis(
            Hypothesis("H2", "Spec Draft", "draft model switches on entropy spikes", focus_area="Speculative decoding", elo_score=1290)
        )
        context.add_hypothesis(
            Hypothesis("H3", "Kernel Fusion", "fuse entropy kernels into attention path", focus_area="Kernel fusion", elo_score=1260)
        )

        with patch.object(EvolutionAgent, "_search_literature", return_value=[]), patch(
            "app.agents.call_json",
            return_value={
                "hypotheses": [
                    {
                        "title": "Adaptive KV Sharing",
                        "focus_area": "KV cache sharing",
                        "core_hypothesis": "Reuse cache segments dynamically.",
                        "mechanism": "Entropy gates cache reuse.",
                        "novelty_rationale": "Adds runtime gating.",
                        "generation_strategy": "grounded_enhancement",
                        "predictions": ["Latency drops."],
                        "test_plan": ["Measure token latency."],
                    }
                ]
            },
        ):
            result = EvolutionAgent().evolve_hypotheses(goal, context)

        self.assertEqual(len(result.hypotheses), 1)
        self.assertGreaterEqual(len(result.hypotheses[0].parent_ids), 1)
        self.assertIn(result.hypotheses[0].parent_ids[0], {"H1", "H2", "H3"})

    def test_generation_refills_when_model_returns_single_object(self):
        goal = ResearchGoal(
            "goal",
            num_hypotheses=4,
            max_literature_results=0,
            constraints={"direction_families": ["KV cache", "Speculative decoding", "Kernel fusion", "Scheduling"]},
        )
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)

        with patch.object(GenerationAgent, "_search_literature", return_value=[]), patch(
            "app.agents.call_json",
            side_effect=[
                {
                    "title": "Cross-Layer KV Recycling",
                    "focus_area": "KV cache",
                    "primary_bottleneck": "cache pressure",
                    "core_hypothesis": "Reuse KV segments across nearby tokens.",
                    "mechanism": "Self-supervised similarity gates reuse.",
                    "novelty_rationale": "Adds runtime similarity control.",
                    "generation_strategy": "literature_grounding",
                    "predictions": ["Latency drops."],
                    "test_plan": ["Measure decode latency."],
                },
                {
                    "hypotheses": [
                        {
                            "title": "Speculative Prefix Reuse",
                            "focus_area": "Speculative decoding",
                            "primary_bottleneck": "latency",
                            "core_hypothesis": "Reuse validated draft prefixes across requests.",
                            "mechanism": "Confidence-conditioned prefix cache.",
                            "novelty_rationale": "Targets cross-request reuse.",
                            "generation_strategy": "expansion",
                            "predictions": ["Higher acceptance rate."],
                            "test_plan": ["Benchmark draft acceptance."],
                        },
                        {
                            "title": "Kernel-Resident Verification",
                            "focus_area": "Kernel fusion",
                            "primary_bottleneck": "compute",
                            "core_hypothesis": "Fuse verification into the attention kernel.",
                            "mechanism": "On-chip validator avoids launch overhead.",
                            "novelty_rationale": "Moves verification into the hot path.",
                            "generation_strategy": "decomposition",
                            "predictions": ["Kernel launch count drops."],
                            "test_plan": ["Profile fused kernels."],
                        },
                        {
                            "title": "Queue-Aware Token Scheduling",
                            "focus_area": "Scheduling",
                            "primary_bottleneck": "latency",
                            "core_hypothesis": "Schedule token groups by contention signature.",
                            "mechanism": "Runtime queue partitioning reduces stalls.",
                            "novelty_rationale": "Adds contention-aware scheduling.",
                            "generation_strategy": "debate",
                            "predictions": ["Tail latency improves."],
                            "test_plan": ["Replay bursty traces."],
                        },
                    ]
                },
            ],
        ) as mocked_call:
            result = GenerationAgent().generate_new_hypotheses(goal, context)

        self.assertEqual(len(result.hypotheses), 4)
        self.assertEqual(result.data["requested_hypotheses"], 4)
        self.assertEqual(result.data["accepted_hypotheses"], 4)
        self.assertEqual(result.data["refill_attempts"], 1)
        self.assertEqual(result.data["refill_added"], 3)
        self.assertEqual(mocked_call.call_count, 2)

    def test_evolution_refills_when_initial_mutation_batch_underfills(self):
        goal = ResearchGoal("goal", top_k_hypotheses=3, max_literature_results=0, evolution_temperature=0.66)
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)
        context.add_hypothesis(Hypothesis("H1", "KV Sharing", "share kv cache across nearby tokens", focus_area="KV cache sharing", elo_score=1330))
        context.add_hypothesis(Hypothesis("H2", "Spec Draft", "draft model switches on entropy spikes", focus_area="Speculative decoding", elo_score=1300))
        context.add_hypothesis(Hypothesis("H3", "Kernel Fusion", "fuse entropy kernels into attention path", focus_area="Kernel fusion", elo_score=1270))
        context.add_hypothesis(Hypothesis("H4", "Scheduler", "schedule by queue depth", focus_area="Scheduling", elo_score=1240))

        with patch.object(EvolutionAgent, "_search_literature", return_value=[]), patch(
            "app.agents.call_json",
            side_effect=[
                {
                    "title": "Adaptive KV Sharing",
                    "focus_area": "KV cache sharing",
                    "primary_bottleneck": "cache pressure",
                    "core_hypothesis": "Reuse cache segments dynamically.",
                    "mechanism": "Entropy gates cache reuse.",
                    "novelty_rationale": "Adds runtime gating.",
                    "generation_strategy": "grounded_enhancement",
                    "mutation_operator": "grounded_enhancement",
                    "parent_ids": ["H1"],
                    "delta_from_parents": "Adds adaptive gating.",
                    "diversity_reason": "Keeps KV line but changes controller.",
                    "predictions": ["Latency drops."],
                    "test_plan": ["Measure token latency."],
                },
                {
                    "hypotheses": [
                        {
                            "title": "Speculative Queue Bypass",
                            "focus_area": "Speculative decoding",
                            "primary_bottleneck": "latency",
                            "core_hypothesis": "Bypass stalled drafts with queue-state triggers.",
                            "mechanism": "Queue-aware draft fallback.",
                            "novelty_rationale": "Combines drafting with scheduling signals.",
                            "generation_strategy": "combination",
                            "mutation_operator": "combination",
                            "parent_ids": ["H2", "H4"],
                            "delta_from_parents": "Injects scheduler signals into draft acceptance.",
                            "diversity_reason": "Bridges two focus areas.",
                            "predictions": ["Tail latency improves."],
                            "test_plan": ["Replay bursty request traces."],
                        },
                        {
                            "title": "Kernel-Assisted Draft Verification",
                            "focus_area": "Kernel fusion",
                            "primary_bottleneck": "compute",
                            "core_hypothesis": "Verify draft branches inside fused kernels.",
                            "mechanism": "On-chip verifier reduces relaunch overhead.",
                            "novelty_rationale": "Turns verification into a fused operation.",
                            "generation_strategy": "inspiration",
                            "mutation_operator": "inspiration",
                            "parent_ids": ["H2", "H3"],
                            "delta_from_parents": "Moves verification into fused kernels.",
                            "diversity_reason": "Expands mutation into a new mechanism family.",
                            "predictions": ["Kernel launches drop."],
                            "test_plan": ["Profile GPU launches."],
                        },
                    ]
                },
            ],
        ) as mocked_call:
            result = EvolutionAgent().evolve_hypotheses(goal, context)

        self.assertEqual(len(result.hypotheses), 3)
        self.assertEqual(result.data["requested_hypotheses"], 3)
        self.assertEqual(result.data["accepted_hypotheses"], 3)
        self.assertEqual(result.data["refill_attempts"], 1)
        self.assertEqual(result.data["refill_added"], 2)
        self.assertEqual(mocked_call.call_count, 2)

    def test_evolution_prompt_requires_ranking_aware_parent_improvement(self):
        messages = build_evolution_messages(
            research_goal="goal",
            research_plan={},
            top_hypotheses=[{"id": "H1", "title": "Parent"}],
            literature_notes=[],
            meta_feedback={},
            research_overview={},
            trajectory_state={},
            num_hypotheses=2,
            uncovered_focus_areas=[],
            overexplored_areas=[],
        )
        prompt = messages[-1]["content"]

        self.assertIn("credible improvement over the source hypotheses", prompt)
        self.assertIn("rank above at least one of its provided source hypotheses", prompt)
        self.assertIn("not just a safe refinement", prompt)
        self.assertIn("problem framing", prompt)
        self.assertIn("theoretical story", prompt)
        self.assertIn("simple method combination", prompt)
        self.assertNotIn("why_better_than_parents", prompt)

    def test_generation_prompt_requires_story_skeleton(self):
        messages = build_generation_messages(
            research_goal="goal",
            research_plan={},
            context_hypotheses=[],
            literature_notes=[],
            meta_feedback={},
            research_overview={},
            trajectory_state={},
            num_hypotheses=2,
            uncovered_focus_areas=[],
            overexplored_areas=[],
        )
        prompt = messages[-1]["content"]

        self.assertIn("central insight", prompt)
        self.assertIn("theoretical story", prompt)
        self.assertIn("simple combination", prompt)

    def test_meta_review_prompt_requests_story_guidance(self):
        messages = build_meta_review_messages(
            research_goal="goal",
            research_plan={},
            ranked_hypotheses=[],
            recent_reviews=[],
            tournament_history=[],
            proximity_summary={},
            trajectory_state={},
            literature_notes=[],
        )
        prompt = messages[-1]["content"]

        self.assertIn("story_guidance", prompt)
        self.assertIn("method-stacking patterns", prompt)
        self.assertIn("frontier_storyline", prompt)

    def test_unified_review_reuses_seeded_and_parent_literature_for_evolved_hypotheses(self):
        goal = ResearchGoal("goal", max_literature_results=3, reflection_temperature=0.3)
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)

        parent = Hypothesis(
            "H1",
            "KV Sharing",
            "share kv cache across nearby tokens",
            focus_area="KV cache sharing",
            literature_notes=[
                {"source": "arxiv", "arxiv_id": "2501.00001v1", "title": "Parent Paper A"},
                {"source": "arxiv", "arxiv_id": "2501.00002v1", "title": "Parent Paper B"},
            ],
        )
        child = Hypothesis(
            "E1",
            "Adaptive KV Sharing",
            "adapt kv sharing to context drift",
            focus_area="KV cache sharing",
            origin="evolution",
            parent_ids=["H1"],
            literature_notes=[
                {"source": "arxiv", "arxiv_id": "2501.00003v1", "title": "Child Paper"},
            ],
        )
        context.add_hypothesis(parent)
        context.add_hypothesis(child)

        with patch.object(ReflectionAgent, "_search_literature", return_value=[]) as mocked_search, patch(
            "app.agents.call_json",
            return_value={
                "alignment_score": 4,
                "novelty_score": 4,
                "plausibility_score": 4,
                "feasibility_score": 4,
                "testability_score": 4,
                "verdict": "advance",
                "short_summary": "Grounded review.",
            },
        ):
            result = ReflectionAgent().review_hypotheses([child], goal, context)

        self.assertEqual(mocked_search.call_count, 0)
        self.assertEqual(len(result.data["literature_by_hypothesis"]["E1"]), 3)
        self.assertEqual(len(child.literature_notes), 3)

    def test_unified_review_fills_missing_literature_from_local_embedding_recall(self):
        goal = ResearchGoal("goal", max_literature_results=2, reflection_temperature=0.3)
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)
        hypothesis = Hypothesis("H1", "Adaptive Router", "route requests by cache pressure")
        local_paper = {
            "source": "arxiv",
            "arxiv_id": "2501.00004v1",
            "title": "Cache Pressure Routing",
            "abstract": "Routing requests by cache pressure.",
        }

        class FakeVectorIndex:
            def search(self, _query_text, notes, top_k):
                return [{"note": notes[0], "semantic_score": 0.88, "vector_rank": 1, "vector_index_backend": "fake"}][:top_k]

        agent = ReflectionAgent()
        with patch.object(agent.literature_service, "local_corpus", return_value=[local_paper]), patch.object(
            agent,
            "_search_literature",
            return_value=[],
        ) as mocked_search, patch("app.agents.get_prior_art_vector_index", return_value=FakeVectorIndex()), patch(
            "app.agents.call_json",
            return_value={
                "alignment_score": 4,
                "novelty_score": 4,
                "plausibility_score": 4,
                "feasibility_score": 4,
                "testability_score": 4,
                "verdict": "advance",
                "short_summary": "Local-library grounded review.",
            },
        ):
            result = agent.review_hypotheses([hypothesis], goal, context)

        notes = result.data["literature_by_hypothesis"]["H1"]
        self.assertEqual(mocked_search.call_count, 0)
        self.assertEqual(notes[0]["title"], "Cache Pressure Routing")
        self.assertEqual(notes[0]["selection_reason"], "review_local_embedding")
        self.assertEqual(hypothesis.literature_notes[0]["semantic_score"], 0.88)

    def test_meta_review_reuses_ranked_hypothesis_literature_before_searching(self):
        goal = ResearchGoal("goal", max_literature_results=3, reflection_temperature=0.3)
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)
        hypothesis = Hypothesis(
            "H1",
            "Top Idea",
            "Idea text",
            elo_score=1300,
            literature_notes=[
                {"source": "arxiv", "arxiv_id": "2501.00001v1", "title": "Paper A"},
                {"source": "arxiv", "arxiv_id": "2501.00002v1", "title": "Paper B"},
                {"source": "arxiv", "arxiv_id": "2501.00003v1", "title": "Paper C"},
            ],
        )
        context.add_hypothesis(hypothesis)

        with patch.object(MetaReviewAgent, "_search_literature", return_value=[]) as mocked_search, patch(
            "app.agents.call_json",
            return_value={
                "meta_review_critique": ["Keep going"],
                "generation_guidance": ["Broaden"],
                "reflection_guidance": ["Ground more"],
                "ranking_guidance": ["Compare sharper"],
                "research_overview": {"summary": "Overview"},
                "expert_profiles": ["systems"],
            },
        ):
            result = MetaReviewAgent().summarize_and_feedback(goal, context, {"clusters": [], "duplicate_candidates": []})

        self.assertEqual(mocked_search.call_count, 0)
        self.assertEqual(result.data["literature_reuse_count"], 3)

    def test_markdown_report_surfaces_story_fields(self):
        report = build_markdown_report(
            [
                {
                    "iteration": 1,
                    "cycle_duration": 0.2,
                    "statistics_before": {},
                    "statistics_after": {},
                    "research_plan": {
                        "objective": "Goal",
                        "domain": "Domain",
                        "focus_areas": ["Area"],
                        "success_criteria": ["Need story"],
                    },
                    "steps": {
                        "ranking_final": {
                            "hypotheses": [
                                {
                                    "id": "H1",
                                    "title": "Story Idea",
                                    "elo_score": 1300,
                                    "review_verdict": "advance",
                                    "focus_area": "Area",
                                    "primary_bottleneck": "latency",
                                    "problem_framing": "A hidden tension exists",
                                    "central_insight": "One mechanism dominates",
                                    "theoretical_story": "The story explains the gain",
                                    "why_not_simple_combination": "It is one causal chain",
                                    "text": "Text",
                                    "scores": {},
                                }
                            ]
                        }
                    },
                    "meta_review": {
                        "meta_review_critique": ["Critique"],
                        "story_guidance": {
                            "frontier_story": "Tell a cleaner story",
                            "method_stacking_patterns": ["Avoid stacks"],
                            "theory_gaps": ["Need sharper theory"],
                        },
                        "research_overview": {
                            "summary": "Overview",
                            "frontier_storyline": "One storyline",
                            "anti_combination_guidance": ["Do not stack methods"],
                            "theory_gaps": ["Need a core claim"],
                            "suggested_next_steps": [],
                            "suggested_experiments": [],
                        },
                    },
                }
            ]
        )

        self.assertIn("Problem framing:", report)
        self.assertIn("Frontier story:", report)
        self.assertIn("Frontier storyline:", report)

    def test_prior_art_check_skips_unchanged_idea_signature(self):
        goal = ResearchGoal("goal", enable_prior_art_check=True)
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)
        hypothesis = Hypothesis("H1", "Cached Idea", "Idea text")
        hypothesis.prior_art_signature = hypothesis.idea_signature()
        hypothesis.prior_art_audit = {"novelty_risk": "low", "decision": "keep"}
        agent = PriorArtAgent()

        with patch.object(
            agent.literature_service,
            "prefetch_corpus",
            return_value={},
        ) as mocked_prefetch:
            result = agent.check_hypotheses([hypothesis], goal, context)

        self.assertEqual(result.data["checked_count"], 0)
        self.assertEqual(result.data["skipped_count"], 1)
        self.assertEqual(result.data["audits"]["H1"]["status"], "skipped_unchanged")
        self.assertEqual(mocked_prefetch.call_count, 0)

    def test_prior_art_prefetches_one_complete_query_for_all_active_ideas(self):
        goal = ResearchGoal(
            "goal",
            enable_prior_art_check=True,
            prior_art_embedding_candidates=1,
            prior_art_review_top_k=1,
            prior_art_similarity_threshold=0.5,
            max_concurrency=1,
        )
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)
        first = Hypothesis("H1", "Cache Router", "Route requests by cache pressure.", search_queries=["cache routing"])
        second = Hypothesis("H2", "Spec Verifier", "Verify drafts with fused kernels.", search_queries=["draft verification"])
        agent = PriorArtAgent()
        paper = {"source": "arxiv", "arxiv_id": "2501.00001v1", "title": "Local Paper", "abstract": "background"}

        class FakeVectorIndex:
            def search(self, _query_text, notes, top_k):
                return [{"note": notes[0], "semantic_score": 0.1, "vector_rank": 1, "vector_index_backend": "fake"}][:top_k]

        batch_term_payload = {
            "ideas": [
                {
                    "idea_id": "H1",
                    "idea_title": "Cache Router",
                    "search_terms": [
                        "route requests by cache pressure",
                        "transformer serving",
                        "KV cache",
                    ],
                    "broader_terms": ["LLM inference"],
                    "rationale": "broad serving vocabulary",
                },
                {
                    "idea_id": "H2",
                    "idea_title": "Spec Verifier",
                    "search_terms": [
                        "verify drafts with fused kernels",
                        "speculative decoding",
                        "kernel fusion",
                    ],
                    "broader_terms": ["LLM acceleration"],
                    "rationale": "broad decoding vocabulary",
                },
            ],
            "shared_terms": ["large language model inference", "transformer inference"],
            "shared_broader_terms": ["efficient inference"],
            "notes": ["batch terms"],
        }
        audit_payload = {
            "novelty_risk": "low",
            "decision": "keep",
            "short_summary": "Batch search was grounded in the local corpus.",
            "overlap_assessment": [],
            "gap_analysis": [],
            "repair_strategy": "",
            "revised_hypothesis": {},
        }
        captured_queries = {}

        def fake_prefetch(queries, results_per_query, max_total=None):
            captured_queries["queries"] = list(queries)
            captured_queries["results_per_query"] = results_per_query
            return {
                "queries": list(queries),
                "query_count": len(queries),
                "results_per_query": results_per_query,
                "fetched_count": 1,
                "local_corpus_size": 1,
                "source_counts": {"arxiv": 1},
                "query_source_counts": {},
                "cache_counts": {},
                "sample_titles": ["Local Paper"],
            }

        with patch(
            "app.agents.call_json",
            side_effect=[batch_term_payload, audit_payload],
        ) as mocked_call, patch.object(
            agent.literature_service,
            "prefetch_corpus",
            side_effect=fake_prefetch,
        ) as mocked_prefetch, patch.object(
            agent.literature_service,
            "local_corpus",
            return_value=[paper],
        ) as mocked_local, patch("app.agents.get_prior_art_vector_index", return_value=FakeVectorIndex()), patch(
            "app.agents.lexical_similarity_score",
            return_value=0.0,
        ):
            result = agent.check_hypotheses([first, second], goal, context)

        self.assertEqual(mocked_call.call_count, 1)
        self.assertEqual(mocked_prefetch.call_count, 1)
        self.assertGreaterEqual(captured_queries["results_per_query"], 1000)
        self.assertEqual(mocked_local.call_count, 1)
        self.assertEqual(len(captured_queries["queries"]), 1)
        combined_query = captured_queries["queries"][0]
        self.assertIn(" OR ", combined_query)
        self.assertIn("transformer serving", combined_query)
        self.assertIn("KV cache", combined_query)
        self.assertIn("LLM inference", combined_query)
        self.assertIn("speculative decoding", combined_query)
        self.assertIn("kernel fusion", combined_query)
        self.assertIn("large language model inference", combined_query)
        self.assertNotIn("route requests by cache pressure", combined_query)
        self.assertNotIn("verify drafts with fused kernels", combined_query)
        self.assertEqual(result.data["checked_count"], 2)
        self.assertEqual(result.data["query_count"], 1)
        self.assertEqual(result.data["batch_prefetch"]["query_count"], 1)
        self.assertEqual(result.data["batch_prefetch"]["query"], combined_query)
        self.assertEqual(result.data["retrieval_mode"], "batch_prefetch_local_vector_index_embedding")

    def test_paper_vector_index_persists_and_ranks_by_embedding(self):
        class FakeEmbeddingModel:
            def encode(self, texts, **_kwargs):
                vectors = []
                for text in texts:
                    normalized = text.casefold()
                    if "beta" in normalized:
                        vectors.append([1.0, 0.0])
                    elif "alpha" in normalized:
                        vectors.append([0.0, 1.0])
                    else:
                        vectors.append([0.7, 0.3])
                return vectors

        notes = [
            {"source": "arxiv", "arxiv_id": "1", "title": "Alpha Compression"},
            {"source": "arxiv", "arxiv_id": "2", "title": "Beta Retrieval"},
        ]
        with TemporaryDirectory() as temp_dir, patch(
            "app.vector_index.get_sentence_transformer_model",
            return_value=FakeEmbeddingModel(),
        ):
            index_path = Path(temp_dir) / "prior_art_index.npz"
            index = PaperVectorIndex(
                index_path=index_path,
                model_name="fake-model",
                backend="numpy",
                query_prefix="",
                document_prefix="",
            )
            results = index.search("beta query", notes, top_k=1)
            index_exists = index_path.exists()

        self.assertEqual(results[0]["note"]["title"], "Beta Retrieval")
        self.assertEqual(results[0]["vector_index_backend"], "numpy")
        self.assertTrue(index_exists)

    def test_paper_vector_index_builds_once_under_concurrent_searches(self):
        class FakeEmbeddingModel:
            def __init__(self):
                self.calls = []

            def encode(self, texts, **_kwargs):
                values = list(texts)
                self.calls.append(values)
                return [
                    [1.0, 0.0] if "beta" in text.casefold() else [0.0, 1.0]
                    for text in values
                ]

        notes = [
            {"source": "arxiv", "arxiv_id": "1", "title": "Alpha Compression"},
            {"source": "arxiv", "arxiv_id": "2", "title": "Beta Retrieval"},
        ]
        model = FakeEmbeddingModel()
        with TemporaryDirectory() as temp_dir, patch(
            "app.vector_index.get_sentence_transformer_model",
            return_value=model,
        ):
            index = PaperVectorIndex(
                index_path=Path(temp_dir) / "prior_art_index.npz",
                model_name="fake-model",
                backend="numpy",
                query_prefix="",
                document_prefix="",
            )
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(lambda _: index.search("beta query", notes, top_k=1), range(4)))

        corpus_builds = [call for call in model.calls if len(call) == len(notes)]
        self.assertEqual(len(corpus_builds), 1)
        self.assertTrue(all(result[0]["note"]["title"] == "Beta Retrieval" for result in results))

    def test_paper_vector_index_incrementally_encodes_only_new_or_changed_papers(self):
        class FakeEmbeddingModel:
            def __init__(self):
                self.calls = []

            def encode(self, texts, **_kwargs):
                values = list(texts)
                self.calls.append(values)
                vectors = []
                for text in values:
                    normalized = text.casefold()
                    if "gamma" in normalized:
                        vectors.append([1.0, 0.0, 0.0])
                    elif "revised" in normalized:
                        vectors.append([0.0, 1.0, 0.0])
                    else:
                        vectors.append([0.0, 0.0, 1.0])
                return vectors

        initial_notes = [
            {"source": "arxiv", "arxiv_id": "1", "title": "Alpha", "abstract": "Original"},
            {"source": "arxiv", "arxiv_id": "2", "title": "Beta", "abstract": "Original"},
        ]
        appended_notes = initial_notes + [
            {"source": "arxiv", "arxiv_id": "3", "title": "Gamma", "abstract": "New paper"},
        ]
        changed_notes = [
            {"source": "arxiv", "arxiv_id": "1", "title": "Alpha", "abstract": "Revised abstract"},
            appended_notes[1],
            appended_notes[2],
        ]
        final_notes = [changed_notes[0], changed_notes[2]]
        model = FakeEmbeddingModel()

        with TemporaryDirectory() as temp_dir, patch(
            "app.vector_index.get_sentence_transformer_model",
            return_value=model,
        ):
            index_path = Path(temp_dir) / "prior_art_index.npz"
            for query, notes in [
                ("first query", initial_notes),
                ("second query", appended_notes),
                ("third query", changed_notes),
                ("fourth query", final_notes),
            ]:
                index = PaperVectorIndex(
                    index_path=index_path,
                    model_name="fake-model",
                    backend="numpy",
                    query_prefix="",
                    document_prefix="",
                )
                index.search(query, notes, top_k=1)

        document_batches = [
            call
            for call in model.calls
            if call and all(text.startswith("Title:") for text in call)
        ]
        self.assertEqual([len(batch) for batch in document_batches], [2, 1, 1])
        self.assertIn("Title: Gamma", document_batches[1][0])
        self.assertIn("Revised abstract", document_batches[2][0])

    def test_embedding_concurrency_is_separate_from_general_concurrency(self):
        active = 0
        max_active = 0
        lock = threading.Lock()

        def worker(_item):
            nonlocal active, max_active
            with embedding_slot():
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.01)
                with lock:
                    active -= 1
            return "done"

        with patch.dict("app.utils.config", {"max_concurrency": 16, "embedding_concurrency": 1}, clear=False):
            results = run_concurrently(list(range(8)), worker)

        self.assertEqual(results, ["done"] * 8)
        self.assertEqual(max_active, 1)

    def test_online_literature_query_bundles_limit_or_terms(self):
        query = (
            '"one two" OR "three four" OR "five six" OR "seven eight" OR '
            '"nine ten" OR "eleven twelve" OR "thirteen fourteen" OR '
            '"fifteen sixteen" OR "seventeen eighteen" OR "nineteen twenty"'
        )
        with patch.dict("app.agents.config", {"literature_or_terms_per_query": 4}, clear=False):
            bundled = GenerationAgent()._bundle_online_literature_queries([query])

        self.assertEqual([item.count(" OR ") + 1 for item in bundled], [4, 4, 2])

    def test_prior_art_ranking_uses_vector_index_without_lexical_prefilter(self):
        class FakeVectorIndex:
            def __init__(self):
                self.seen_corpus_sizes = []

            def search(self, _query_text, notes, top_k):
                self.seen_corpus_sizes.append(len(notes))
                return [
                    {
                        "note": notes[1],
                        "semantic_score": 0.91,
                        "vector_rank": 1,
                        "vector_index_backend": "fake",
                    }
                ][:top_k]

        goal = ResearchGoal("goal", prior_art_embedding_candidates=1, prior_art_review_top_k=1)
        hypothesis = Hypothesis("H1", "Target Idea", "Find semantic target paper.")
        lexical_bait = {"source": "arxiv", "arxiv_id": "1", "title": "Target Idea lexical bait"}
        semantic_target = {"source": "arxiv", "arxiv_id": "2", "title": "Semantic Match"}
        fake_index = FakeVectorIndex()

        with patch("app.agents.get_prior_art_vector_index", return_value=fake_index), patch(
            "app.agents.lexical_similarity_score",
            side_effect=lambda _left, right: 1.0 if "lexical bait" in right else 0.0,
        ):
            ranked = PriorArtAgent()._rank_prior_art(hypothesis, [lexical_bait, semantic_target], goal)

        self.assertEqual(fake_index.seen_corpus_sizes, [2])
        self.assertEqual(ranked[0]["title"], "Semantic Match")
        self.assertEqual(ranked[0]["recall_score"], 0.91)

    def test_prior_art_embedding_text_uses_title_abstract_only(self):
        hypothesis = Hypothesis(
            "H1",
            "Compact Query",
            "Core abstract sentence about the proposed method.",
            mechanism="This mechanism should not be duplicated when text exists.",
            rationale="This rationale should not be duplicated when text exists.",
            predictions=["Prediction text should stay out of embedding queries."],
            validation_experiments=["Experiment text should stay out of embedding queries."],
        )

        idea_text = PriorArtAgent()._idea_recall_text(hypothesis)
        paper_text = paper_recall_text(
            {
                "title": "Compact Paper",
                "abstract": "Paper abstract sentence.",
                "citation": "Citation should stay out.",
                "venue": "Venue should stay out.",
            }
        )

        self.assertIn("Title: Compact Query", idea_text)
        self.assertIn("Abstract: Core abstract sentence", idea_text)
        self.assertNotIn("Mechanism:", idea_text)
        self.assertNotIn("Novelty rationale:", idea_text)
        self.assertNotIn("Prediction text", idea_text)
        self.assertNotIn("Experiment text", idea_text)
        self.assertNotIn("This mechanism should not be duplicated", idea_text)
        self.assertIn("Title: Compact Paper", paper_text)
        self.assertIn("Abstract: Paper abstract sentence.", paper_text)
        self.assertNotIn("Citation should stay out", paper_text)
        self.assertNotIn("Venue should stay out", paper_text)

    def test_prior_art_check_repairs_duplicate_and_rechecks_new_signature(self):
        goal = ResearchGoal(
            "goal",
            llm_model="thinking-model",
            critic_llm_model="critic-model",
            enable_prior_art_check=True,
            prior_art_embedding_candidates=1,
            prior_art_review_top_k=1,
            prior_art_similarity_threshold=0.5,
            prior_art_repair_attempts=1,
        )
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)
        hypothesis = Hypothesis("H1", "Duplicate Idea", "Reuse exact mechanism from prior work.")
        agent = PriorArtAgent()
        paper = {
            "source": "arxiv",
            "arxiv_id": "2501.00001v1",
            "title": "Prior Duplicate",
            "summary": "Prior work uses the same mechanism.",
            "arxiv_url": "https://arxiv.org/abs/2501.00001v1",
        }
        audit_payload = {
            "novelty_risk": "high",
            "decision": "repair",
            "short_summary": "The original mechanism is already covered.",
            "overlap_assessment": [
                {
                    "paper_title": "Prior Duplicate",
                    "problem_overlap": "high",
                    "mechanism_overlap": "high",
                    "claim_overlap": "high",
                    "duplicate_level": "strong",
                    "rationale": "Same problem, mechanism, and novelty claim.",
                }
            ],
            "gap_analysis": ["Prior work lacks queue-aware gating."],
            "repair_strategy": "Move the claim to queue-aware gating.",
            "revised_hypothesis": {
                "title": "Queue-Aware Gap Idea",
                "focus_area": "runtime routing",
                "primary_bottleneck": "latency",
                "core_hypothesis": "Use queue-aware gating to avoid the prior duplicate mechanism.",
                "mechanism": "A queue-state controller chooses when reuse is safe.",
                "novelty_rationale": "Targets a scheduling gap not covered by the prior work.",
                "predictions": ["Tail latency improves."],
                "test_plan": ["Replay bursty traces."],
                "search_queries": ["queue-aware gating transformer serving"],
            },
        }
        vector_responses = [
            [{"note": paper, "semantic_score": 0.9, "vector_rank": 1, "vector_index_backend": "fake"}],
            [{"note": paper, "semantic_score": 0.1, "vector_rank": 1, "vector_index_backend": "fake"}],
        ]

        class FakeVectorIndex:
            def search(self, _query_text, _notes, top_k):
                return vector_responses.pop(0)

        with patch.object(
            agent,
            "_prefetch_literature_for_hypotheses",
            return_value={
                "queries": ["duplicate query"],
                "results_per_query": 1,
                "query_count": 1,
                "fetched_count": 1,
                "local_corpus_size": 1,
            },
        ), patch.object(
            agent.literature_service,
            "local_corpus",
            return_value=[paper],
        ) as mocked_local, patch("app.agents.get_prior_art_vector_index", return_value=FakeVectorIndex()), patch(
            "app.agents.lexical_similarity_score",
            return_value=0.0,
        ), patch("app.agents.call_json", return_value=audit_payload) as mocked_call:
            result = agent.check_hypotheses([hypothesis], goal, context)

        self.assertEqual(result.data["repaired_count"], 1)
        self.assertEqual(result.data["rejected_count"], 0)
        self.assertTrue(hypothesis.is_active)
        self.assertEqual(hypothesis.title, "Queue-Aware Gap Idea")
        self.assertEqual(hypothesis.prior_art_repair_count, 1)
        self.assertEqual(hypothesis.prior_art_signature, hypothesis.idea_signature())
        self.assertEqual(hypothesis.prior_art_audit["novelty_risk"], "low")
        self.assertEqual(mocked_local.call_count, 1)
        self.assertEqual(mocked_call.call_count, 1)
        self.assertEqual(mocked_call.call_args.kwargs["profile"], "thinking")
        self.assertEqual(mocked_call.call_args.kwargs["model"], "thinking-model")

    def test_semantic_scholar_enabled_without_api_key(self):
        service = LiteratureSearchService()
        service.sources = {"semantic_scholar"}
        service.semantic_scholar_tool.api_key = None

        self.assertTrue(service._semantic_scholar_enabled())

    def test_literature_search_selects_cited_recent_and_random_papers(self):
        with TemporaryDirectory() as temp_dir:
            service = LiteratureSearchService()
            service.cache = LiteratureCache(str(Path(temp_dir) / "literature_cache.json"))
            service.sources = set()
            notes = [
                {
                    "source": "semantic_scholar",
                    "semantic_scholar_id": f"S2-{index}",
                    "title": f"Paper {index}",
                    "published": published,
                    "citation_count": citation_count,
                }
                for index, (published, citation_count) in enumerate(
                    [
                        ("2024-01-01", 5),
                        ("2023-01-01", 50),
                        ("2026-02-01", 7),
                        ("2022-01-01", 40),
                        ("2026-05-01", 1),
                        ("2021-01-01", 3),
                        ("2025-01-01", 2),
                        ("2020-01-01", 100),
                        ("2026-03-01", 20),
                        ("2024-06-01", 30),
                    ]
                )
            ]
            service.cache.set("arxiv", "trajectory representation", notes)

            with patch.object(service, "_search_live_source_batch", return_value=([], {}, {})) as mocked_batch:
                sampled = service.search(["trajectory representation"], max_results=5, sample_seed="cycle-2")

        titles = [note["title"] for note in sampled]
        reasons = {note["title"]: note.get("selection_reason") for note in sampled}
        self.assertEqual(mocked_batch.call_count, 1)
        self.assertEqual(mocked_batch.call_args.args[0], ["trajectory representation"])
        self.assertEqual(len(sampled), 5)
        self.assertIn("Paper 7", titles)
        self.assertIn("Paper 1", titles)
        self.assertIn("Paper 4", titles)
        self.assertIn("Paper 8", titles)
        self.assertEqual(reasons["Paper 7"], "high_relevance_high_citation")
        self.assertEqual(reasons["Paper 4"], "high_relevance_recent")
        self.assertIn("exploratory_random", reasons.values())

    def test_online_literature_search_bundles_plain_queries_with_or(self):
        agent = GenerationAgent()

        with patch.dict("app.agents.config", {"literature_or_terms_per_query": 3}, clear=False), patch.object(
            agent.literature_service,
            "search",
            return_value=[],
        ) as mocked_search:
            agent._search_literature(
                [
                    "end-to-end model",
                    "auto driving model",
                    "behavior cloning",
                    "fused kernel",
                    "draft validation",
                ],
                max_results=5,
                sample_seed="cycle-2",
            )

        bundled_queries = mocked_search.call_args.args[0]
        self.assertEqual(mocked_search.call_args.kwargs["max_results"], 5)
        self.assertEqual(mocked_search.call_args.kwargs["sample_seed"], "cycle-2")
        self.assertEqual(len(bundled_queries), 2)
        self.assertTrue(all(" OR " in query for query in bundled_queries))
        self.assertTrue(any("end-to-end model" in query and "auto driving model" in query for query in bundled_queries))
        self.assertTrue(any("fused kernel" in query and "draft validation" in query for query in bundled_queries))

    def test_literature_prompt_serializes_all_supplied_notes(self):
        notes = [
            {"title": f"Paper {index}", "summary": "s", "citation": "c", "arxiv_url": "u"}
            for index in range(12)
        ]

        payload = _serialize_literature(notes)

        self.assertIn("Paper 11", payload)

    def test_literature_prompt_serializes_temporal_and_selection_metadata(self):
        notes = [
            {
                "title": "Frontier Paper",
                "summary": "s",
                "citation": "c",
                "source": "semantic_scholar",
                "sources": ["semantic_scholar"],
                "published": "2026-05-01",
                "year": 2026,
                "venue": "TestConf",
                "citation_count": 42,
                "selection_reason": "high_relevance_recent",
                "arxiv_url": "u",
            }
        ]

        payload = _serialize_literature(notes)

        self.assertIn("2026-05-01", payload)
        self.assertIn("citation_count", payload)
        self.assertIn("high_relevance_recent", payload)

    def test_trajectory_analysis_counts_focus_coverage(self):
        payload = {
            "cycles": [
                {
                    "steps": {
                        "ranking_final": {
                            "hypotheses": [
                                {"id": "H1", "title": "KV Idea", "focus_area": "KV cache", "elo_score": 1300},
                                {"id": "H2", "title": "Spec Idea", "focus_area": "Speculative decoding", "elo_score": 1260},
                            ]
                        },
                        "generation": {"hypotheses": [{}, {}], "requested_hypotheses": 4, "raw_payload_keys": [["title"]]},
                        "evolution": {"hypotheses": [{}], "requested_hypotheses": 2, "raw_payload_keys": [["hypotheses"]]},
                    },
                    "errors": [],
                },
                {
                    "steps": {
                        "ranking_final": {
                            "hypotheses": [
                                {"id": "H3", "title": "Kernel Idea", "focus_area": "Kernel fusion", "elo_score": 1320},
                                {"id": "H1", "title": "KV Idea", "focus_area": "KV cache", "elo_score": 1290},
                            ]
                        },
                        "generation": {"hypotheses": [{}, {}, {}], "requested_hypotheses": 4, "raw_payload_keys": [["hypotheses"]]},
                        "evolution": {"hypotheses": [{}, {}], "requested_hypotheses": 2, "raw_payload_keys": [["title"]]},
                    },
                    "errors": [],
                },
            ],
            "final_context": {
                "research_plan": {
                    "focus_areas": ["KV cache", "Speculative decoding", "Kernel fusion"],
                },
                "hypotheses": {
                    "H1": {"id": "H1", "title": "KV Idea", "focus_area": "KV cache", "origin": "generation", "elo_score": 1290, "is_active": True, "created_in_iteration": 1},
                    "H2": {"id": "H2", "title": "Spec Idea", "focus_area": "Speculative decoding", "origin": "evolution", "parent_ids": ["G1"], "elo_score": 1260, "is_active": True, "created_in_iteration": 1},
                    "H3": {"id": "H3", "title": "Kernel Idea", "focus_area": "Kernel fusion", "origin": "evolution", "parent_ids": ["G2"], "elo_score": 1320, "is_active": True, "created_in_iteration": 2},
                },
            },
        }

        report = analyze_run_payload(payload, label="synthetic")

        self.assertEqual(report["active_focus_coverage"], 1.0)
        self.assertEqual(report["top10_focus_coverage"], 1.0)
        self.assertEqual(report["lineage_coverage"], 1.0)
        self.assertEqual(report["generation_shortfall_cycles"], 2)
        self.assertEqual(report["evolution_shortfall_cycles"], 1)
        self.assertEqual(report["generation_schema_violation_cycles"], 1)
        self.assertEqual(report["evolution_schema_violation_cycles"], 1)
        self.assertAlmostEqual(report["avg_generation_fill_ratio"], 0.625)
        self.assertGreater(report["trajectory_score"], 0.0)

    def test_llm_call_falls_back_to_secondary_model(self):
        class FakeMessage:
            def __init__(self, content):
                self.content = content

        class FakeChoice:
            def __init__(self, content):
                self.message = FakeMessage(content)

        class FakeCompletion:
            def __init__(self, content):
                self.choices = [FakeChoice(content)]

        class FakeChatCompletions:
            def __init__(self):
                self.calls = []

            def create(self, model, messages, temperature):
                self.calls.append(model)
                if model == "primary-model":
                    raise RuntimeError("502 Bad Gateway")
                return FakeCompletion("{\"ok\":true}")

        class FakeClient:
            def __init__(self):
                self.chat = type("Chat", (), {"completions": FakeChatCompletions()})()

        fake_client = FakeClient()

        with patch(
            "app.llm._candidate_providers",
            return_value=[
                llm_module.LLMProviderCandidate(model="primary-model", api_key="primary-key", base_url="http://primary.test/v1"),
                llm_module.LLMProviderCandidate(model="fallback-model", api_key="fallback-key", base_url="http://fallback.test/v1"),
            ],
        ), patch("app.llm._get_client", return_value=fake_client):
            text = call_llm("Return ok", temperature=0.1)

        self.assertEqual(text, "{\"ok\":true}")
        self.assertEqual(fake_client.chat.completions.calls, ["primary-model", "fallback-model"])

    def test_llm_call_retries_until_success_by_default(self):
        class FakeMessage:
            def __init__(self, content):
                self.content = content

        class FakeChoice:
            def __init__(self, content):
                self.message = FakeMessage(content)

        class FakeCompletion:
            def __init__(self, content):
                self.choices = [FakeChoice(content)]

        class FakeChatCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, model, messages, temperature):
                self.calls += 1
                if self.calls < 4:
                    raise RuntimeError("502 Bad Gateway")
                return FakeCompletion("{\"ok\":true}")

        class FakeClient:
            def __init__(self):
                self.chat = type("Chat", (), {"completions": FakeChatCompletions()})()

        fake_client = FakeClient()

        with patch.dict(
            llm_module.config,
            {
                "llm_retry_until_success": True,
                "initial_retry_delay": 0,
                "max_retry_delay_seconds": 0,
            },
        ), patch(
            "app.llm._candidate_providers",
            return_value=[llm_module.LLMProviderCandidate(model="primary-model", api_key="test-key", base_url="http://provider.test/v1")],
        ), patch("app.llm._get_client", return_value=fake_client):
            text = call_llm("Return ok", temperature=0.1)

        self.assertEqual(text, "{\"ok\":true}")
        self.assertEqual(fake_client.chat.completions.calls, 4)

    def test_llm_critic_profile_uses_separate_config(self):
        class FakeMessage:
            def __init__(self, content):
                self.content = content

        class FakeChoice:
            def __init__(self, content):
                self.message = FakeMessage(content)

        class FakeCompletion:
            def __init__(self, content):
                self.choices = [FakeChoice(content)]

        class FakeChatCompletions:
            def __init__(self):
                self.calls = []

            def create(self, model, messages, temperature):
                self.calls.append(
                    {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                    }
                )
                return FakeCompletion("{\"ok\":true}")

        class FakeOpenAI:
            instances = []

            def __init__(self, base_url, api_key, max_retries):
                self.base_url = base_url
                self.api_key = api_key
                self.max_retries = max_retries
                self.chat = type("Chat", (), {"completions": FakeChatCompletions()})()
                self.instances.append(self)

        llm_module._clients.clear()
        try:
            with patch.dict(os.environ, {}, clear=True), patch.dict(
                llm_module.config,
                {
                    "thinking_llm": {
                        "api_key": "thinking-key",
                        "base_url": "http://thinking.test/v1",
                        "model": "thinking-model",
                    },
                    "critic_llm": {
                        "api_key": "critic-key",
                        "base_url": "http://critic.test/v1",
                        "model": "critic-model",
                    },
                    "max_retries": 1,
                    "initial_retry_delay": 0,
                },
                clear=True,
            ), patch("app.llm.OpenAI", FakeOpenAI):
                text = llm_module.call_llm("Return ok", temperature=0.1, profile="critic")
        finally:
            llm_module._clients.clear()

        self.assertEqual(text, "{\"ok\":true}")
        self.assertEqual(FakeOpenAI.instances[0].api_key, "critic-key")
        self.assertEqual(FakeOpenAI.instances[0].base_url, "http://critic.test/v1")
        self.assertEqual(FakeOpenAI.instances[0].chat.completions.calls[0]["model"], "critic-model")

    def test_llm_critic_profile_falls_back_to_secondary_provider(self):
        class FakeMessage:
            def __init__(self, content):
                self.content = content

        class FakeChoice:
            def __init__(self, content):
                self.message = FakeMessage(content)

        class FakeCompletion:
            def __init__(self, content):
                self.choices = [FakeChoice(content)]

        class FakeChatCompletions:
            def __init__(self, owner):
                self.owner = owner

            def create(self, model, messages, temperature):
                self.owner.calls.append((self.owner.base_url, self.owner.api_key, model))
                if self.owner.base_url == "http://critic-a.test/v1":
                    raise RuntimeError("provider busy")
                return FakeCompletion("{\"ok\":true}")

        class FakeOpenAI:
            instances = []

            def __init__(self, base_url, api_key, max_retries):
                self.base_url = base_url
                self.api_key = api_key
                self.max_retries = max_retries
                self.calls = []
                self.chat = type("Chat", (), {"completions": FakeChatCompletions(self)})()
                self.instances.append(self)

        llm_module._clients.clear()
        try:
            with patch.dict(os.environ, {}, clear=True), patch.dict(
                llm_module.config,
                {
                    "critic_llm": {
                        "providers": [
                            {
                                "api_key": "critic-key-a",
                                "base_url": "http://critic-a.test/v1",
                                "model": "critic-model-a",
                            },
                            {
                                "api_key": "critic-key-b",
                                "base_url": "http://critic-b.test/v1",
                                "model": "critic-model-b",
                            },
                        ]
                    },
                    "max_retries": 1,
                    "initial_retry_delay": 0,
                    "llm_retry_until_success": False,
                },
                clear=True,
            ), patch("app.llm.OpenAI", FakeOpenAI):
                text = llm_module.call_llm("Return ok", temperature=0.1, profile="critic")
        finally:
            llm_module._clients.clear()

        self.assertEqual(text, "{\"ok\":true}")
        self.assertEqual(len(FakeOpenAI.instances), 2)
        self.assertEqual(FakeOpenAI.instances[0].calls, [("http://critic-a.test/v1", "critic-key-a", "critic-model-a")])
        self.assertEqual(FakeOpenAI.instances[1].calls, [("http://critic-b.test/v1", "critic-key-b", "critic-model-b")])

    def test_similarity_score_falls_back_when_sentence_transformer_unavailable(self):
        with patch("app.utils.get_sentence_transformer_model", side_effect=RuntimeError("offline")):
            score = similarity_score("kv cache reuse for transformers", "transformers kv cache reuse")

        self.assertGreater(score, 0.0)

    def test_supervisor_progress_callback_is_invoked(self):
        goal = ResearchGoal("goal", enable_safety_review=False)
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)
        observed_steps = []

        supervisor = SupervisorAgent()
        supervisor.generation_agent.generate_new_hypotheses = lambda *_: type("R", (), {"name": "generation", "hypotheses": [], "data": {}, "errors": [], "duration": 0.0, "to_dict": lambda self: {"hypotheses": []}})()
        supervisor.reflection_agent.review_hypotheses = lambda *args, **kwargs: type("R", (), {"name": kwargs.get("step_name", "review"), "hypotheses": [], "data": {"reviewed_count": 0}, "errors": [], "duration": 0.0, "to_dict": lambda self: {"hypotheses": [], "reviewed_count": 0}})()
        supervisor.proximity_agent.build_proximity_graph = lambda *args, **kwargs: type("R", (), {"name": kwargs.get("step_name", "proximity"), "hypotheses": [], "data": {"clusters": [], "duplicate_candidates": [], "adjacency_graph": {}}, "errors": [], "duration": 0.0, "to_dict": lambda self: {"hypotheses": [], "clusters": [], "duplicate_candidates": [], "adjacency_graph": {}}})()
        supervisor.ranking_agent.run_tournament = lambda *args, **kwargs: type("R", (), {"name": kwargs.get("step_name", "ranking"), "hypotheses": [], "data": {"matches": [], "pairs_considered": 0, "skipped_cached_pairs": 0}, "errors": [], "duration": 0.0, "to_dict": lambda self: {"hypotheses": [], "matches": [], "pairs_considered": 0, "skipped_cached_pairs": 0}})()
        supervisor.opencode_reranking_agent.rerank_top_hypotheses = lambda *args, **kwargs: type("R", (), {"name": kwargs.get("step_name", "opencode_reranking"), "hypotheses": [], "data": {"status": "skipped"}, "errors": [], "duration": 0.0, "to_dict": lambda self: {"hypotheses": [], "status": "skipped"}})()
        supervisor.evolution_agent.evolve_hypotheses = lambda *args, **kwargs: type("R", (), {"name": "evolution", "hypotheses": [], "data": {}, "errors": [], "duration": 0.0, "to_dict": lambda self: {"hypotheses": []}})()
        supervisor.meta_review_agent.summarize_and_feedback = lambda *args, **kwargs: type("R", (), {"name": "meta_review", "hypotheses": [], "data": {"meta_review_critique": [], "research_overview": {}}, "errors": [], "duration": 0.0, "to_dict": lambda self: {"hypotheses": [], "meta_review_critique": [], "research_overview": {}}})()

        supervisor.run_cycle(goal, context, progress_callback=lambda step, *_: observed_steps.append(step))

        self.assertIn("generation", observed_steps)
        self.assertIn("review", observed_steps)
        self.assertIn("opencode_reranking", observed_steps)
        self.assertIn("evolution_replacement_pruning", observed_steps)
        self.assertIn("ranking_final", observed_steps)
        self.assertIn("opencode_reranking_final", observed_steps)
        self.assertIn("cycle_complete", observed_steps)

    def test_frontier_decay_prunes_lowest_elo_active_hypotheses(self):
        goal = ResearchGoal("goal", top_k_hypotheses=3, hypothesis_decay_fraction=0.25, enable_safety_review=False)
        context = ContextMemory()
        context.reset_for_goal(goal)
        for index in range(8):
            context.add_hypothesis(
                Hypothesis(f"H{index}", f"Idea {index}", "text", elo_score=1000 + index * 10)
            )

        result = SupervisorAgent()._decay_low_elo_hypotheses(goal, context)

        self.assertEqual(result.data["pruned_count"], 2)
        self.assertFalse(context.hypotheses["H0"].is_active)
        self.assertFalse(context.hypotheses["H1"].is_active)
        self.assertEqual(context.hypotheses["H0"].review_verdict, "decayed_low_elo")

    def test_evolution_replacement_pruning_removes_one_old_idea_per_new_idea(self):
        goal = ResearchGoal("goal", enable_safety_review=False)
        context = ContextMemory()
        context.reset_for_goal(goal)
        old_hypotheses = [
            Hypothesis("H1", "Old Low", "text", elo_score=1000),
            Hypothesis("H2", "Old Mid", "text", elo_score=1010),
            Hypothesis("H3", "Old High", "text", elo_score=1300),
        ]
        new_hypotheses = [
            Hypothesis("E1", "New A", "text", origin="evolution", elo_score=1200),
            Hypothesis("E2", "New B", "text", origin="evolution", elo_score=1200),
        ]
        for hypothesis in old_hypotheses + new_hypotheses:
            context.add_hypothesis(hypothesis)

        result = SupervisorAgent()._prune_replaced_hypotheses(goal, context, new_hypotheses)

        self.assertEqual(result.data["pruned_count"], 2)
        self.assertFalse(context.hypotheses["H1"].is_active)
        self.assertFalse(context.hypotheses["H2"].is_active)
        self.assertTrue(context.hypotheses["H3"].is_active)
        self.assertTrue(context.hypotheses["E1"].is_active)
        self.assertEqual(context.hypotheses["H1"].review_verdict, "replaced_by_evolution")

    def test_opencode_reranking_applies_returned_top4_order_and_records_evaluations(self):
        goal = ResearchGoal(
            "goal",
            reference_arxiv_url="https://arxiv.org/abs/2502.18864",
            reference_paper_context={"title": "Reference", "arxiv_url": "https://arxiv.org/abs/2502.18864"},
            enable_safety_review=False,
        )
        context = ContextMemory()
        context.reset_for_goal(goal)
        context.research_plan = ResearchPlan.from_goal(goal)
        for index, hypothesis_id in enumerate(["H1", "H2", "H3", "H4"], start=0):
            context.add_hypothesis(
                Hypothesis(hypothesis_id, f"Idea {hypothesis_id}", "text", elo_score=1300 - index * 10)
            )
        payload = {
            "ranking": [
                {"id": "H3", "rank": 1, "overall_score": 4.7, "summary": "Best differentiated idea."},
                {"id": "H1", "rank": 2, "overall_score": 4.2},
                {"id": "H2", "rank": 3, "overall_score": 3.8},
                {"id": "H4", "rank": 4, "overall_score": 3.0},
            ],
            "evaluations": {
                "H3": {
                    "summary": "Strong but needs evidence.",
                    "weaknesses": ["Needs a clearer baseline."],
                    "novelty_risks": ["Adjacent to known routing work."],
                    "feasibility_risks": ["May need costly experiments."],
                    "recommended_action": "keep",
                }
            },
            "overall_rationale": "H3 has the strongest gap.",
        }

        with patch(
            "app.agents.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["opencode"],
                returncode=0,
                stdout=json.dumps(payload),
                stderr="",
            ),
        ) as mocked_run:
            result = OpenCodeRerankingAgent().rerank_top_hypotheses(goal, context)

        command = mocked_run.call_args.args[0]
        self.assertEqual(command[:3], ["opencode", "run", "--dangerously-skip-permissions"])
        self.assertEqual(len(command), 4)
        self.assertIn("Rerank the current top-four research ideas.", command[-1])
        self.assertEqual(result.data["status"], "applied")
        self.assertEqual(result.data["command"], "opencode run --dangerously-skip-permissions")
        self.assertEqual(context.get_ranked_hypotheses()[0].hypothesis_id, "H3")
        self.assertEqual(context.hypotheses["H3"].elo_score, 1300)
        self.assertEqual(context.hypotheses["H3"].opencode_rerank_reviews[-1]["rank"], 1)
        self.assertIn("Needs a clearer baseline.", context.hypotheses["H3"].review_comments)
        self.assertEqual(context.opencode_rerank_history[-1]["overall_rationale"], "H3 has the strongest gap.")

    def test_frontier_decay_preserves_minimum_active_pool(self):
        goal = ResearchGoal("goal", top_k_hypotheses=4, hypothesis_decay_fraction=0.75, enable_safety_review=False)
        context = ContextMemory()
        context.reset_for_goal(goal)
        for index in range(5):
            context.add_hypothesis(
                Hypothesis(f"H{index}", f"Idea {index}", "text", elo_score=1000 + index * 10)
            )

        result = SupervisorAgent()._decay_low_elo_hypotheses(goal, context)

        self.assertEqual(result.data["pruned_count"], 1)
        self.assertEqual(result.data["active_after"], 4)
        self.assertEqual(len(context.get_active_hypotheses()), 4)

    def test_literature_dedupe_merges_same_paper_from_multiple_sources(self):
        notes = dedupe_notes(
            [
                {
                    "source": "semantic_scholar",
                    "semantic_scholar_id": "S2-1",
                    "title": "Shared Paper",
                    "doi": "10.123/shared",
                    "authors": ["Ada"],
                    "sources": ["semantic_scholar"],
                },
                {
                    "source": "arxiv",
                    "title": "Shared Paper",
                    "doi": "10.123/shared",
                    "arxiv_id": "2501.00001v2",
                    "authors": ["Ada", "Turing"],
                    "sources": ["arxiv"],
                },
            ]
        )

        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["sources"], ["arxiv", "semantic_scholar"])
        self.assertEqual(notes[0]["authors"], ["Ada", "Turing"])
        self.assertEqual(notes[0]["arxiv_id"], "2501.00001v2")

    def test_literature_service_queries_both_sources_and_cache(self):
        class FakeSemanticScholar:
            calls = 0

            def search_papers(self, query, max_results=10):
                self.calls += 1
                return [
                    {
                        "semantic_scholar_id": "S2-1",
                        "title": "Shared Paper",
                        "abstract": "Semantic Scholar abstract",
                        "authors": ["Ada"],
                        "year": 2025,
                        "venue": "TestConf",
                        "doi": "10.123/shared",
                        "arxiv_id": "2501.00001",
                        "url": "https://semanticscholar.org/paper/S2-1",
                        "citation_count": 17,
                    }
                ]

        class FakeArxiv:
            calls = 0

            def search_papers(self, query, max_results=None, categories=None, sort_by="relevance"):
                self.calls += 1
                return [
                    {
                        "title": "Shared Paper",
                        "abstract": "arXiv abstract",
                        "authors": ["Ada", "Turing"],
                        "published": "2025-01-01T00:00:00",
                        "journal_ref": "",
                        "doi": "10.123/shared",
                        "arxiv_id": "2501.00001v1",
                        "arxiv_url": "https://arxiv.org/abs/2501.00001",
                        "pdf_url": "https://arxiv.org/pdf/2501.00001",
                    }
                ]

        with TemporaryDirectory() as temp_dir:
            service = LiteratureSearchService()
            service.cache = LiteratureCache(f"{temp_dir}/literature_cache.json")
            service.sources = {"semantic_scholar", "arxiv"}
            service.semantic_scholar_tool = FakeSemanticScholar()
            service.arxiv_tool = FakeArxiv()

            first = service.search(["neural reasoning"], max_results=5)
            second = service.search(["reasoning neural"], max_results=5)

        self.assertEqual(service.semantic_scholar_tool.calls, 2)
        self.assertEqual(service.arxiv_tool.calls, 2)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["sources"], ["arxiv", "semantic_scholar"])
        self.assertEqual(first[0]["citation_count"], 17)
        self.assertGreaterEqual(len(second), 1)

    def test_prefetch_expands_partial_exact_query_cache_to_requested_size(self):
        query = '"end-to-end model" OR "autonomous driving model"'
        cached_paper = {
            "source": "arxiv",
            "arxiv_id": "2501.00001v1",
            "title": "Cached Paper",
            "abstract": "cached",
        }

        class FakeArxiv:
            def __init__(self):
                self.calls = 0

            def search_papers(self, query, max_results=None, categories=None, sort_by="relevance"):
                self.calls += 1
                self.query = query
                self.max_results = max_results
                return [
                    {
                        "title": "New Paper A",
                        "abstract": "new a",
                        "authors": [],
                        "published": "2026-01-01T00:00:00",
                        "arxiv_id": "2601.00002v1",
                        "arxiv_url": "https://arxiv.org/abs/2601.00002",
                        "pdf_url": "https://arxiv.org/pdf/2601.00002",
                    },
                    {
                        "title": "New Paper B",
                        "abstract": "new b",
                        "authors": [],
                        "published": "2026-01-02T00:00:00",
                        "arxiv_id": "2601.00003v1",
                        "arxiv_url": "https://arxiv.org/abs/2601.00003",
                        "pdf_url": "https://arxiv.org/pdf/2601.00003",
                    },
                    {
                        "title": "New Paper C",
                        "abstract": "new c",
                        "authors": [],
                        "published": "2026-01-03T00:00:00",
                        "arxiv_id": "2601.00004v1",
                        "arxiv_url": "https://arxiv.org/abs/2601.00004",
                        "pdf_url": "https://arxiv.org/pdf/2601.00004",
                    },
                ]

        with TemporaryDirectory() as temp_dir:
            service = LiteratureSearchService()
            service.cache = LiteratureCache(f"{temp_dir}/literature_cache.json")
            service.sources = {"arxiv"}
            service.arxiv_tool = FakeArxiv()
            service.cache.set("arxiv", query, [cached_paper])

            result = service.prefetch_corpus([query], results_per_query=3)
            expanded_cache = service.cache.get("arxiv", query)

        self.assertEqual(service.arxiv_tool.calls, 1)
        self.assertEqual(service.arxiv_tool.query, query)
        self.assertEqual(service.arxiv_tool.max_results, 3)
        self.assertEqual(len(expanded_cache), 4)
        self.assertEqual(result["query_count"], 1)
        self.assertEqual(result["fetched_count"], 4)

    def test_extract_hypothesis_items_accepts_common_top_level_list_keys(self):
        payload = {
            "list": [
                {"title": "Idea A", "core_hypothesis": "A"},
                {"title": "Idea B", "core_hypothesis": "B"},
            ]
        }

        items = _extract_hypothesis_items(payload)

        self.assertEqual([item["title"] for item in items], ["Idea A", "Idea B"])

    def test_extract_hypothesis_items_falls_back_to_single_list_value(self):
        payload = {
            "coverage_report": {"covered": ["x"]},
            "results": [
                {"title": "Idea A", "core_hypothesis": "A"},
            ],
        }

        items = _extract_hypothesis_items(payload)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "Idea A")

    def test_literature_service_uses_semantic_scholar_without_api_key(self):
        class FakeSemanticScholar:
            api_key = None

            def __init__(self):
                self.calls = 0

            def search_papers(self, query, max_results=10):
                self.calls += 1
                return [
                    {
                        "semantic_scholar_id": "S2-1",
                        "title": "Only Semantic Scholar Paper",
                        "abstract": "Semantic Scholar abstract",
                        "authors": ["Grace"],
                        "year": 2025,
                        "venue": "Test Venue",
                        "published": "2025-01-01",
                        "doi": "10.0000/example",
                        "arxiv_id": "",
                        "url": "https://semanticscholar.org/paper/S2-1",
                        "pdf_url": "",
                        "citation_count": 23,
                    }
                ]

        class FakeArxiv:
            def __init__(self):
                self.calls = 0

            def search_papers(self, query, max_results=None, categories=None, sort_by="relevance"):
                self.calls += 1
                return [
                    {
                        "title": "Only arXiv Paper",
                        "abstract": "arXiv abstract",
                        "authors": ["Ada"],
                        "published": "2025-01-01T00:00:00",
                        "journal_ref": "",
                        "doi": None,
                        "arxiv_id": "2501.00001v1",
                        "arxiv_url": "https://arxiv.org/abs/2501.00001",
                        "pdf_url": "https://arxiv.org/pdf/2501.00001",
                    }
                ]

        with TemporaryDirectory() as temp_dir:
            service = LiteratureSearchService()
            service.cache = LiteratureCache(f"{temp_dir}/literature_cache.json")
            service.sources = {"semantic_scholar", "arxiv"}
            service.semantic_scholar_tool = FakeSemanticScholar()
            service.arxiv_tool = FakeArxiv()

            notes = service.search(["neural reasoning"], max_results=5)

        self.assertEqual(service.semantic_scholar_tool.calls, 1)
        self.assertEqual(service.arxiv_tool.calls, 1)
        self.assertEqual(len(notes), 2)
        self.assertIn(23, [note.get("citation_count") for note in notes])

    def test_arxiv_429_retries_once_before_skipping_live_search(self):
        with TemporaryDirectory() as temp_dir, patch.dict(
            "app.tools.arxiv_search.config",
            {
                "arxiv_state_path": f"{temp_dir}/arxiv_state.json",
                "arxiv_rate_limit_max_retries": 1,
                "arxiv_rate_limit_retry_after_seconds": 0,
            },
            clear=False,
        ):
            tool = ArxivSearchTool(max_results=2)

            with patch.object(tool.client, "results", side_effect=RuntimeError("HTTP 429 Too Many Requests")) as mocked_results, patch(
                "app.tools.arxiv_search.time.sleep"
            ) as mocked_sleep:
                notes = tool.search_papers("kv cache")

            self.assertEqual(notes, [])
            self.assertGreater(tool._disabled_until, 0.0)
            self.assertEqual(tool._consecutive_failures, 1)
            self.assertEqual(mocked_results.call_count, 2)
            mocked_sleep.assert_called_once_with(0.0)

    def test_arxiv_query_translates_generic_or_phrases_to_field_syntax(self):
        tool = ArxivSearchTool(max_results=2)
        query = (
            '"LoRA vision-language-action robot manipulation" OR '
            '"in-context learning diffusion policy robot"'
        )

        translated = tool._arxiv_query(query)

        self.assertIn("all:LoRA AND all:vision AND all:language AND all:action", translated)
        self.assertIn("all:context AND all:learning AND all:diffusion", translated)
        self.assertNotIn('"', translated)
        self.assertNotIn("vision-language-action", translated)
        self.assertEqual(tool._arxiv_query("au:Geoffrey Hinton"), "au:Geoffrey Hinton")

    def test_semantic_scholar_query_removes_exact_phrase_punctuation(self):
        tool = SemanticScholarSearchTool()

        translated = tool._semantic_scholar_query(
            '"LoRA vision-language-action robot manipulation" OR "diffusion-policy robot"'
        )

        self.assertEqual(
            translated,
            "LoRA vision language action robot manipulation | diffusion policy robot",
        )

    def test_arxiv_429_retry_can_recover_live_search(self):
        with TemporaryDirectory() as temp_dir, patch.dict(
            "app.tools.arxiv_search.config",
            {
                "arxiv_state_path": f"{temp_dir}/arxiv_state.json",
                "arxiv_rate_limit_max_retries": 1,
                "arxiv_rate_limit_retry_after_seconds": 0,
            },
            clear=False,
        ):
            tool = ArxivSearchTool(max_results=2)
            formatted_paper = {"source": "arxiv", "title": "Recovered Paper", "arxiv_id": "2501.00001v1"}

            with patch.object(
                tool.client,
                "results",
                side_effect=[RuntimeError("HTTP 429 Too Many Requests"), [object()]],
            ) as mocked_results, patch.object(tool, "_format_paper", return_value=formatted_paper), patch(
                "app.tools.arxiv_search.time.sleep"
            ) as mocked_sleep:
                notes = tool.search_papers("kv cache")

            self.assertEqual(notes, [formatted_paper])
            self.assertEqual(tool._disabled_until, 0.0)
            self.assertEqual(tool._consecutive_failures, 0)
            self.assertEqual(mocked_results.call_count, 2)
            mocked_sleep.assert_called_once_with(0.0)

    def test_semantic_scholar_429_retries_once_before_skipping(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return b'{"data":[{"paperId":"S2-1","title":"Recovered Paper"}]}'

        rate_limit_error = urllib.error.HTTPError(
            url="https://api.semanticscholar.org/graph/v1/paper/search",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=None,
        )

        with patch.dict(
            "app.tools.semantic_scholar.config",
            {
                "semantic_scholar_max_retries": 1,
                "semantic_scholar_retry_after_seconds": 0,
                "semantic_scholar_min_delay_seconds": 0,
            },
            clear=False,
        ):
            tool = SemanticScholarSearchTool()

            with patch(
                "app.tools.semantic_scholar.urllib.request.urlopen",
                side_effect=[rate_limit_error, FakeResponse()],
            ) as mocked_urlopen, patch("app.tools.semantic_scholar.time.sleep") as mocked_sleep:
                notes = tool.search_papers("kv cache", max_results=1)

        self.assertEqual(notes[0]["semantic_scholar_id"], "S2-1")
        self.assertEqual(mocked_urlopen.call_count, 2)
        mocked_sleep.assert_called_once_with(0.0)

    def test_arxiv_cooldown_persists_across_tool_instances(self):
        with TemporaryDirectory() as temp_dir, patch.dict(
            "app.tools.arxiv_search.config",
            {
                "arxiv_state_path": f"{temp_dir}/arxiv_state.json",
                "arxiv_rate_limit_max_retries": 1,
                "arxiv_rate_limit_retry_after_seconds": 0,
            },
            clear=False,
        ):
            first = ArxivSearchTool(max_results=2)
            with patch.object(first.client, "results", side_effect=RuntimeError("HTTP 429 Too Many Requests")), patch(
                "app.tools.arxiv_search.time.sleep"
            ):
                first.search_papers("kv cache")

            state_path = Path(temp_dir) / "arxiv_state.json"
            self.assertTrue(state_path.exists())

            second = ArxivSearchTool(max_results=2)
            with patch.object(second.client, "results") as mocked_results:
                notes = second.search_papers("speculative decoding")

        self.assertEqual(notes, [])
        self.assertGreater(second._disabled_until, 0.0)
        mocked_results.assert_not_called()


if __name__ == "__main__":
    unittest.main()
