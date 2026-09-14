import json
import subprocess
import sys
from pathlib import Path

import wandb


ROOT = Path(__file__).resolve().parent


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    repo_url = git_output("remote", "get-url", "origin")
    branch = git_output("branch", "--show-current")
    commit_sha = git_output("rev-parse", "HEAD")
    test_command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]

    test_result = subprocess.run(test_command, cwd=ROOT, text=True, capture_output=True)
    test_status = "passed" if test_result.returncode == 0 else "failed"

    run = wandb.init(
        project="shakespeare-char-reproducibility",
        group="submission-reproducibility",
        name="task-reproducibility-link-git-wandb",
        config={
            "repository_url": repo_url,
            "github_url": repo_url.removesuffix(".git"),
            "branch": branch,
            "commit_sha": commit_sha,
            "readme_path": "README.md",
            "submission_manifest_path": "submission_manifest.json",
            "experiment_summary_path": "reports/wandb/experiment_summary.md",
            "reproducibility_report_path": "reports/reproducibility_run.json",
            "test_command": " ".join(test_command),
        },
    )

    run.summary.update(
        {
            "repository": repo_url,
            "commit_sha": commit_sha,
            "readme_path": "README.md",
            "submission_manifest_path": "submission_manifest.json",
            "experiment_summary_path": "reports/wandb/experiment_summary.md",
            "reproducibility_report_path": "reports/reproducibility_run.json",
            "test_status": test_status,
            "test_returncode": test_result.returncode,
        }
    )

    output_path = ROOT / "reports" / "reproducibility_run.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "run_name": run.name,
                "wandb_run_url": run.url,
                "repository": {
                    "url": repo_url,
                    "web_url": repo_url.removesuffix(".git"),
                    "branch": branch,
                    "commit_sha": commit_sha,
                },
                "paths": {
                    "readme": "README.md",
                    "submission_manifest": "submission_manifest.json",
                    "experiment_summary": "reports/wandb/experiment_summary.md",
                    "reproducibility_report": "reports/reproducibility_run.json",
                    "tests": "tests/test_reproducibility.py",
                },
                "execution": {
                    "test_status": test_status,
                    "test_returncode": test_result.returncode,
                    "stdout_tail": test_result.stdout[-4000:],
                    "stderr_tail": test_result.stderr[-4000:],
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    run.finish()
    return test_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
