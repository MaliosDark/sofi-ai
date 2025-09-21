#!/usr/bin/env python3

from huggingface_hub import HfApi
import os

# Initialize API with token
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("Please set HF_TOKEN environment variable")
api = HfApi(token=hf_token)

# Repository details
repo_id = "MaliosDark/SOFIA-v2-agi"
readme_path = "/home/nexland/sofi-labs/SOFIA-v2-lora/README.md"

print(f"Updating README for {repo_id}...")

try:
    # Upload just the README
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update README with correct repository URLs and latest documentation"
    )

    print("✅ Successfully updated README on HuggingFace!")
    print(f"📍 Model available at: https://huggingface.co/{repo_id}")

except Exception as e:
    print(f"❌ Error updating README: {e}")
