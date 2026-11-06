#!/usr/bin/env bash
# Upload the model card and the released adapters to the Hugging Face Hub.
#   hf auth login                (or export HF_TOKEN=...)
#   bash huggingface/upload.sh <user-or-org>/CRePE
set -euo pipefail
REPO="${1:?usage: upload.sh <hf-repo-id>}"
cd "$(dirname "$0")/.."
sha256sum -c huggingface/SHA256SUMS
python - "$REPO" <<'PY'
import sys
from huggingface_hub import create_repo, upload_file
repo = sys.argv[1]
create_repo(repo, repo_type="model", exist_ok=True)
for src, dst in [("huggingface/README.md", "README.md"),
                 ("checkpoints/crepe.safetensors", "crepe.safetensors"),
                 ("checkpoints/crepe_mix.safetensors", "crepe_mix.safetensors")]:
    upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, repo_type="model")
    print("uploaded", dst)
PY
