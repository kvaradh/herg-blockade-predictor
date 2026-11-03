"""Create the Hugging Face Space on free CPU hardware and upload the app.

The website forces new Spaces onto ZeroGPU, which refuses to run an app with no
@spaces.GPU function - and this assignment requires CPU anyway. Downgrading an
existing ZeroGPU Space needs a PRO subscription, but pinning cpu-basic at
CREATION time is allowed, which is what this does.

Run `.venv/bin/hf auth login` first (paste a WRITE token from
https://huggingface.co/settings/tokens). The token is stored by the CLI; this
script never prints it.

    python deploy_space.py <your-hf-username> [space-name]
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).parent
FILES = HERE / "part2"


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python deploy_space.py <your-hf-username> [space-name]")
    user = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "herg-blockade-predictor"
    repo_id = f"{user}/{name}"

    api = HfApi()
    whoami = api.whoami()                       # fails clearly if not logged in
    print(f"authenticated as {whoami['name']}")

    print(f"creating {repo_id} on cpu-basic ...")
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio",
                    space_hardware="cpu-basic", private=False, exist_ok=True)

    print(f"uploading {len(list(FILES.iterdir()))} files from {FILES} ...")
    api.upload_folder(folder_path=str(FILES), repo_id=repo_id, repo_type="space",
                      commit_message="hERG blockade predictor")

    runtime = api.get_space_runtime(repo_id)
    print(f"\nhardware : {runtime.hardware}")
    print(f"stage    : {runtime.stage}")
    print(f"\n  https://huggingface.co/spaces/{repo_id}\n")
    print("Building takes 2-3 minutes. If hardware is not cpu-basic, stop and say so.")


if __name__ == "__main__":
    main()
