import subprocess
from pathlib import Path
import yaml


def get_head_commit(repo_path: str) -> str:
    """Gets the exact 40-character SHA-256 commit hash of a repository."""
    res = subprocess.run(
        ["git", "-C", repo_path, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


# 1. Target repository path (absolute or relative)
target_repo = str(Path("tests/fixtures/sample_repo").resolve())
commit_hash = get_head_commit(".")  # Uses current repo commit hash

# 2. Define your 5 benchmark items for this repository
benchmark_items = [
    {
        "id": "sample-001",
        "repo": "tests/fixtures/sample_repo",
        "commit": commit_hash,
        "question": "Where is the login authentication function defined?",
        "question_type": "lookup",
        "gold_files": ["tests/fixtures/sample_repo/login.py"],
        "gold_ranges": [
            {
                "file": "tests/fixtures/sample_repo/login.py",
                "start": 1,
                "end": 10,
            }
        ],
        "reference_answer": "Defined in login.py around lines 1-10.",
    },
    {
        "id": "sample-002",
        "repo": "tests/fixtures/sample_repo",
        "commit": commit_hash,
        "question": "How is a token generated during the authentication flow?",
        "question_type": "lookup",
        "gold_files": ["tests/fixtures/sample_repo/token.py"],
        "gold_ranges": [
            {
                "file": "tests/fixtures/sample_repo/token.py",
                "start": 1,
                "end": 8,
            }
        ],
        "reference_answer": "Defined in token.py.",
    },
    {
        "id": "sample-003",
        "repo": "tests/fixtures/sample_repo",
        "commit": commit_hash,
        "question": "How do login and token creation work together?",
        "question_type": "multi-file",
        "gold_files": [
            "tests/fixtures/sample_repo/login.py",
            "tests/fixtures/sample_repo/token.py",
        ],
        "gold_ranges": [
            {
                "file": "tests/fixtures/sample_repo/login.py",
                "start": 1,
                "end": 10,
            },
            {
                "file": "tests/fixtures/sample_repo/token.py",
                "start": 1,
                "end": 8,
            },
        ],
        "reference_answer": "login.py verifies user details and token.py generates the token string.",
    },
    {
        "id": "sample-004",
        "repo": "tests/fixtures/sample_repo",
        "commit": commit_hash,
        "question": "What configuration settings exist in the application?",
        "question_type": "conceptual",
        "gold_files": ["tests/fixtures/sample_repo/config.py"],
        "gold_ranges": [
            {
                "file": "tests/fixtures/sample_repo/config.py",
                "start": 1,
                "end": 5,
            }
        ],
        "reference_answer": "Defined in config.py.",
    },
    {
        "id": "sample-005",
        "repo": "tests/fixtures/sample_repo",
        "commit": commit_hash,
        "question": "How is Kubernetes pod autoscaling configured?",
        "question_type": "unanswerable",
        "gold_files": [],
        "gold_ranges": [],
        "reference_answer": "not enough information",
    },
]

# 3. Ensure the benchmark folder exists and save output
output_dir = Path("src/evaluation/benchmarks")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "sample_repo.yaml"

with open(output_file, "w") as f:
    yaml.dump(benchmark_items, f, sort_keys=False)

print(
    f"Successfully generated benchmark YAML at: {output_file.get_abs_path() if hasattr(output_file, 'get_abs_path') else output_file.resolve()}"
)