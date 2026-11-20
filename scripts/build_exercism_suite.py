"""Build 10 pinned Exercism Python benchmark repos and TaskManifests (no .meta leakage)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXERCISES: tuple[tuple[str, str], ...] = (
    ("two-fer", "easy"),
    ("raindrops", "easy"),
    ("hamming", "easy"),
    ("word-count", "medium"),
    ("anagram", "medium"),
    ("grade-school", "medium"),
    ("rotational-cipher", "medium"),
    ("kindergarten-garden", "hard"),
    ("clock", "hard"),
    ("tournament", "hard"),
)


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _find_solution_and_test(exercise_dir: Path) -> tuple[str, str]:
    cfg_path = exercise_dir / ".meta" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    files = cfg.get("files", {})
    solution = files["solution"][0]
    test_file = files["test"][0]
    return str(solution), str(test_file)


def _build_issue_text(
    slug: str, solution_file: str, test_file: str, exercise_dir: Path
) -> str:
    parts: list[str] = [
        f"Implement the Python exercise '{slug}' in `{solution_file}` so that all tests in `{test_file}` pass.",
    ]
    intro = exercise_dir / ".docs" / "introduction.md"
    instr = exercise_dir / ".docs" / "instructions.md"
    append = exercise_dir / ".docs" / "instructions.append.md"
    for doc in (intro, instr, append):
        if doc.is_file():
            text = doc.read_text(encoding="utf-8").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def build_suite(
    *,
    exercism_src: Path,
    repos_root: Path,
    manifests_dir: Path,
    python_exe: str,
) -> list[dict[str, object]]:
    upstream_sha = _git(exercism_src, "rev-parse", "HEAD")
    repos_root.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

    for slug, difficulty in EXERCISES:
        src_dir = exercism_src / "exercises" / "practice" / slug
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Missing Exercism exercise: {src_dir}")
        solution_file, test_file = _find_solution_and_test(src_dir)
        example_path = src_dir / ".meta" / "example.py"
        if not example_path.is_file():
            raise FileNotFoundError(f"Missing .meta/example.py for {slug}")

        repo_dir = repos_root / slug
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        repo_dir.mkdir(parents=True)

        # Copy only public exercise files (strictly excluding .meta/)
        shutil.copy2(src_dir / solution_file, repo_dir / solution_file)
        shutil.copy2(src_dir / test_file, repo_dir / test_file)
        (repo_dir / ".gitignore").write_text(
            "__pycache__/\n*.pyc\n*.pyo\n.pytest_cache/\n.aider*\n",
            encoding="utf-8",
        )
        docs_dir = src_dir / ".docs"
        if docs_dir.is_dir():
            shutil.copytree(docs_dir, repo_dir / ".docs")
        assert not (repo_dir / ".meta").exists(), ".meta must never exist in task repo"

        _git(repo_dir, "init", "-q")
        _git(repo_dir, "config", "user.email", "benchmark@local")
        _git(repo_dir, "config", "user.name", "Benchmark Builder")
        _git(repo_dir, "add", ".")
        _git(repo_dir, "commit", "-q", "-m", f"Initial Exercism stub for {slug}")
        base_commit = _git(repo_dir, "rev-parse", "HEAD")

        # Sanity check 1: stub must fail pytest
        stub_proc = subprocess.run(
            [python_exe, "-m", "pytest", "-q", test_file],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if stub_proc.returncode == 0:
            raise RuntimeError(f"Stub for {slug} unexpectedly passed pytest!")

        # Sanity check 2: official .meta/example.py must pass pytest in a temp copy
        with tempfile.TemporaryDirectory(prefix=f"verify_{slug}_") as tmp:
            check_dir = Path(tmp) / slug
            shutil.copytree(repo_dir, check_dir)
            shutil.copy2(example_path, check_dir / solution_file)
            ref_proc = subprocess.run(
                [python_exe, "-m", "pytest", "-q", test_file],
                cwd=check_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if ref_proc.returncode != 0:
                raise RuntimeError(
                    f"Reference solution for {slug} failed pytest:\n{ref_proc.stdout}\n{ref_proc.stderr}"
                )

        issue_text = _build_issue_text(slug, solution_file, test_file, src_dir)
        manifest = {
            "task_id": f"exercism__{slug}",
            "repo": str(repo_dir.resolve()),
            "base_commit": base_commit,
            "issue_text": issue_text,
            "oracle_tests": [test_file],
            "oracle_tier": "local_full",
            "agent_test_targets": [test_file],
            "metadata": {
                "source": "exercism/python",
                "upstream_commit": upstream_sha,
                "slug": slug,
                "difficulty": difficulty,
                "solution_file": solution_file,
                "test_file": test_file,
            },
        }
        out_path = manifests_dir / f"{slug}.json"
        out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        summary.append(
            {
                "task_id": manifest["task_id"],
                "difficulty": difficulty,
                "solution_file": solution_file,
                "test_file": test_file,
                "base_commit": base_commit[:10],
            }
        )
        print(
            f"[OK] {manifest['task_id']} ({difficulty}): stub fails, reference passes",
            flush=True,
        )

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pinned Exercism Python suite.")
    parser.add_argument(
        "--exercism-src", type=Path, default=Path(".runs/exercism_src/python")
    )
    parser.add_argument("--repos-root", type=Path, default=Path(".runs/exercism_repos"))
    parser.add_argument(
        "--manifests-dir", type=Path, default=Path("benchmarks/exercism")
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    build_suite(
        exercism_src=args.exercism_src,
        repos_root=args.repos_root,
        manifests_dir=args.manifests_dir,
        python_exe=args.python,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
