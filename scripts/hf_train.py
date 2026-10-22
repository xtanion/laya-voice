# /// script
# requires-python = ">=3.12"
# dependencies = ["laya>=0.4.12", "huggingface_hub>=1.0"]
# ///
"""Fine-tune Laya on Hugging Face Jobs and push the checkpoint to a private model repo.

    hf jobs uv run --flavor a10g-small --timeout 2h --secrets HF_TOKEN \\
        -v ./data/train:/data scripts/hf_train.py --repo USER/laya-voice-v1

Arguments after --repo/--data are passed to laya-train (defaults: --shuffle-options
--loss soft-ce). --dry-run checks the data without loading weights or pushing.
"""
import argparse
import os
import subprocess
import sys
import tempfile

DEFAULT_TRAIN_ARGS = ["--shuffle-options", "--loss", "soft-ce"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", help="model repo to push to (required unless --dry-run)")
    p.add_argument("--data", default="/data/v1.jsonl")
    p.add_argument("--dry-run", action="store_true")
    args, train_args = p.parse_known_args()
    if not args.dry_run and not args.repo:
        p.error("--repo is required")

    out = tempfile.mkdtemp(prefix="laya-voice-")
    cmd = ["laya-train", "--data", args.data, *(train_args or DEFAULT_TRAIN_ARGS)]
    cmd += ["--dry-run"] if args.dry_run else ["--out", out]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, "USE_TF": "0"})
    if args.dry_run:
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, private=True, exist_ok=True)
    commit = api.upload_folder(repo_id=args.repo, folder_path=out,
                               commit_message=f"laya-train {' '.join(cmd[1:])} (job {os.environ.get('JOB_ID', 'local')})")
    print(f"pushed {commit.oid[:12]} to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    sys.exit(main())
