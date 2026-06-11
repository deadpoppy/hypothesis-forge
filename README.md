# Hypothesis Forge

![Hypothesis Forge workflow](docs/assets/hypothesis-forge-overview.svg)

Hypothesis Forge is a CLI-first research ideation engine inspired by the AI co-scientist workflow in [arXiv:2502.18864](https://arxiv.org/pdf/2502.18864). It turns a research goal plus a required reference arXiv paper into a structured plan, generates and evolves hypotheses over multiple cycles, grounds ideas with literature search, audits prior art, ranks candidates, and writes reviewable Markdown/JSON reports.

The project is intentionally lightweight: it is built for local experimentation, reproducible idea sweeps, and quick inspection of how a pool of research hypotheses changes over time.

For agents: copy `config.example.yaml` to `config.yaml`, set `thinking_llm` and `critic_llm`, then run `python app.py --goal "YOUR_RESEARCH_GOAL" --reference-arxiv "https://arxiv.org/abs/2502.18864" --output-dir results/YOUR_RUN`.

## What It Does

- Builds a structured research plan from a plain-language goal and one required arXiv reference paper.
- Generates fresh hypotheses and evolves high-scoring candidates across cycles.
- Converts and summarizes the reference paper with the installed `arxiv2md` package before ideation.
- Retrieves literature from Semantic Scholar and arXiv for grounding.
- Runs larger-pool prior-art checks to catch duplicate or weakly differentiated ideas.
- Reviews hypotheses with one consolidated critic pass and reranks the current top four ideas with `opencode run --dangerously-skip-permissions`.
- Tracks trajectory quality, diversity, stability, and failure modes.
- Saves timestamped reports plus live checkpoints for long runs.

## Repository Layout

```text
.
├── app.py                    # CLI entrypoint
├── app/                      # Workflow, agents, prompts, models, tools
├── config.example.yaml       # Public configuration template
├── docs/assets/              # README and documentation assets
├── run_configs/              # Optional constraint/profile examples
├── tests/                    # Unit tests
├── trajectory_sweep.py       # Optional sweep runner for trajectory/debug profiles
├── Makefile
└── README.md
```

Local-only files such as `config.yaml`, `.cache/`, virtual environments, and `results/` are ignored by git.

## Installation

Use Python 3.10+.

```bash
git clone https://github.com/deadpoppy/hypothesis-forge.git
cd hypothesis-forge
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The install includes `sentence-transformers` and `torch` because prior-art recall and hypothesis proximity scoring use semantic embedding similarity for the full workflow. `scikit-learn` and `numpy` are not listed directly because the repository code does not import them.

The top-four reranking stage also requires the `opencode` CLI to be installed and available on `PATH`.

## Configuration

Create a private local config from the public template:

```bash
cp config.example.yaml config.yaml
```

Only two LLM profiles are required:

```yaml
thinking_llm:
  providers:
    - api_key: null
      base_url: "https://api.openai.com/v1"
      model: "gpt-4.1"

critic_llm:
  providers:
    - api_key: null
      base_url: "https://api.openai.com/v1"
      model: "gpt-4.1"
    - api_key: null
      base_url: "https://openrouter.ai/api/v1"
      model: "openai/gpt-4.1"
```

`thinking_llm` handles planning, hypothesis generation, evolution, literature query planning, and prior-art repair. `critic_llm` handles safety checks, reflection, pairwise ranking, and meta-review. Each profile accepts one or more provider entries, and calls automatically move to the next configured entry if the current provider errors or is busy.

You can put keys in `config.yaml`, but environment variables are safer:

```bash
export THINKING_LLM_API_KEY="your_thinking_key"
export CRITIC_LLM_API_KEY="your_critic_key"
```

For single-key setups, `LLM_API_KEY` is also accepted as a shared fallback. Do not commit `config.yaml`.

LLM calls retry until they succeed by default, which is useful when the provider is temporarily congested:

```yaml
llm_retry_until_success: true
initial_retry_delay: 2
max_retry_delay_seconds: 60
```

Set `llm_retry_until_success: false` only if you want `max_retries` to cap failed requests.

Semantic similarity is enabled by default:

```yaml
sentence_transformer_enabled: true
sentence_transformer_model: "BAAI/bge-small-en-v1.5"
sentence_transformer_local_files_only: false
prior_art_include_cache_corpus: true
cycle_literature_terms_per_idea: 6
cycle_literature_shared_terms: 12
cycle_literature_results_per_query: 1000
literature_or_terms_per_query: 4
prior_art_embedding_candidates: 200
prior_art_vector_index_path: ".cache/prior_art_vector_index.npz"
prior_art_vector_index_backend: "auto"
prior_art_query_prefix: ""
```

Prior-art recall now plans broad field/category search terms for the whole cycle's idea set, deduplicates all terms into one complete OR query, sends that query once to each paper source with a default limit of 1000 papers per source, writes every returned paper into the cache, and then reuses the local literature cache as the embedding corpus for per-idea checks. Other online literature grounding also bundles plain candidate terms before calling the paper sources. Query and paper embeddings use a compact `Title:` + `Abstract:` text format. The `auto` backend uses FAISS when it is installed and falls back to chunked numpy search otherwise. Keep `sentence_transformer_local_files_only: true` only after the configured embedding model is already cached locally.

## Quick Start

With the default run settings in `config.yaml`, a normal run needs a goal, reference arXiv paper, and output directory:

```bash
python app.py \
  --goal "Generate strong research ideas for compressed trajectory representations in multimodal embodied agents, focusing on separating noise from meaningful structure in action/state trajectories and using compression to improve planning, generalization, and policy quality." \
  --reference-arxiv "https://arxiv.org/abs/2502.18864" \
  --output-dir results/trajectory_compression_c6650
```

The default template currently uses:

```yaml
cycles: 5
num_hypotheses: 5
top_k_hypotheses: 3
ranking_matches_per_cycle: 1000
max_literature_results: 6
max_concurrency: 16
enable_opencode_reranking: true
opencode_rerank_timeout_seconds: 900
```

You can override any of these from the CLI when needed:

```bash
python app.py \
  --goal "Study robust evaluation methods for AI scientific hypothesis generation" \
  --reference-arxiv "https://arxiv.org/abs/2502.18864" \
  --cycles 1 \
  --num-hypotheses 2 \
  --ranking-matches-per-cycle 3 \
  --output-dir results/smoke_test
```

The Makefile wrapper is useful for short goals:

```bash
make run GOAL="Study robust evaluation methods for AI scientific hypothesis generation" REFERENCE_ARXIV="https://arxiv.org/abs/2502.18864" OUTPUT_DIR=results/smoke_test
```

Runs resume automatically by default. If the same `--goal` and workflow settings are used with the same `--output-dir`, the CLI loads the newest matching `checkpoint_latest.json` or timestamped report and continues until the requested `--cycles` total is reached. Use `--no-resume` when you intentionally want to ignore existing artifacts in that directory.

## Outputs

Each run writes:

- `co_scientist_run_YYYY-MM-DD_HH-MM-SS.json`
- `co_scientist_run_YYYY-MM-DD_HH-MM-SS.md`
- `checkpoint_latest.json`
- `checkpoint_latest.md`

The Markdown report is usually the easiest artifact to inspect first. The JSON report contains the full cycle history, per-step diagnostics, and final context memory.

## Analysis Tools

The `app/trajectory.py`, `app/evolution_analysis.py`, `trajectory_sweep.py`, and `run_configs/trajectory_profiles.json` files are optional diagnostics for understanding how hypothesis pools evolve. They are useful when you want to compare exploration settings, check lineage stability, or debug why a run converged too quickly.

Trajectory analysis:

```bash
python -m app.trajectory results/
```

Save trajectory analysis artifacts:

```bash
python -m app.trajectory results/ \
  --markdown-out results/trajectory_summary.md \
  --json-out results/trajectory_summary.json
```

Deep evolution analysis:

```bash
python -m app.evolution_analysis results/
```

Trajectory sweep command generator:

```bash
python trajectory_sweep.py \
  --goal "YOUR_RESEARCH_GOAL" \
  --constraints run_configs/llm_inference_10_direction_constraints.json
```

## Tests

```bash
make test
```

Equivalent:

```bash
python -m unittest discover -s tests
```

## Operational Notes

- The required reference paper is converted through the installed `arxiv2md` package, then summarized by the thinking LLM before the workflow starts.
- Top-four reranking calls OpenCode as `opencode run --dangerously-skip-permissions "<prompt>"`; set `enable_opencode_reranking: false` only for smoke tests without the OpenCode CLI.
- arXiv `429` responses trigger a cooldown saved at `arxiv_state_path`.
- Reference paper conversion uses the `arxiv2md` package installed in the same
  Python environment that runs Hypothesis Forge. If conversion fails with an
  import error, install or expose that package in the active environment.
- Semantic Scholar is optional. Without an API key, the workflow uses unauthenticated rate limits or continues with the remaining sources.
- `sentence_transformer_local_files_only: true` requires the configured embedding model to already exist locally; otherwise vector recall falls back to direct lexical/embedding safeguards.
- Set `max_literature_results: 0` for LLM-only smoke tests without live literature grounding.

## Scope

Hypothesis Forge is a research ideation and review assistant, not an autonomous scientific authority. Generated ideas should be treated as candidates for expert review, implementation, and empirical validation.
