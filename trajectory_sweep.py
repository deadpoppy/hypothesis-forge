from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def _build_command(
    goal: str,
    constraints: str,
    cycles: int,
    output_root: str,
    profile: Dict[str, Any],
) -> List[str]:
    profile_name = str(profile["name"])
    profile_output = str(Path(output_root) / profile_name)
    command = [
        "python",
        "app.py",
        "--goal",
        goal,
        "--constraints",
        constraints,
        "--cycles",
        str(cycles),
        "--output-dir",
        profile_output,
    ]

    for flag, key in (
        ("--generation-temperature", "generation_temperature"),
        ("--evolution-temperature", "evolution_temperature"),
        ("--reflection-temperature", "reflection_temperature"),
        ("--top-k-hypotheses", "top_k_hypotheses"),
        ("--ranking-matches-per-cycle", "ranking_matches_per_cycle"),
        ("--deep-review-top-k", "deep_review_top_k"),
        ("--max-literature-results", "max_literature_results"),
        ("--max-concurrency", "max_concurrency"),
    ):
        if key in profile:
            command.extend([flag, str(profile[key])])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run a trajectory sweep over multiple parameter profiles.")
    parser.add_argument("--goal", required=True, help="Research goal.")
    parser.add_argument("--constraints", required=True, help="Constraint JSON file.")
    parser.add_argument("--profiles", default="run_configs/trajectory_profiles.json", help="Profile JSON file.")
    parser.add_argument("--cycles", type=int, default=10, help="Cycles per run.")
    parser.add_argument("--output-root", default="results/trajectory_sweep", help="Output root directory.")
    parser.add_argument("--execute", action="store_true", help="Actually execute the sweep instead of printing commands.")
    args = parser.parse_args()

    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    commands = [
        _build_command(
            goal=args.goal,
            constraints=args.constraints,
            cycles=args.cycles,
            output_root=args.output_root,
            profile=profile,
        )
        for profile in profiles
    ]

    for profile, command in zip(profiles, commands):
        print(f"# {profile['name']}: {profile.get('description', '')}")
        print(shlex.join(command))
        if args.execute:
            subprocess.run(command, check=False)


if __name__ == "__main__":
    main()
