#!/usr/bin/env python3

from huggingface_hub import HfApi, upload_file
import os

# Initialize API with token
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("Please set HF_TOKEN environment variable")
api = HfApi(token=hf_token)

# Repository details
repo_id = "MaliosDark/SOFIA-v2-agi"

# Files to upload
agi_files = [
    "README_SOFIA.md",
    "sofia_reasoning.py",
    "sofia_federated.py",
    "sofia_tools_advanced.py",
    "sofia_agi_demo.py",
    "sofia_multimodal.py",
    "sofia_self_improving.py",
    "sofia_meta_cognition.py",
    ".gitignore"
]

print(f"Uploading SOFIA v2.0-AGI code files to {repo_id}...")

try:
    for file_path in agi_files:
        if os.path.exists(file_path):
            print(f"Uploading {file_path}...")
            upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add AGI module: {file_path}"
            )
            print(f"✅ Successfully uploaded {file_path}")
        else:
            print(f"⚠️  File {file_path} not found, skipping...")

    # Also upload the model README
    model_readme = "SOFIA-v2-lora/README.md"
    if os.path.exists(model_readme):
        print(f"Uploading {model_readme}...")
        upload_file(
            path_or_fileobj=model_readme,
            path_in_repo="README.md",  # Upload to root as main README
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update main README with AGI capabilities"
        )
        print(f"✅ Successfully uploaded model README")

    print("🎉 Successfully uploaded all SOFIA v2.0-AGI files to HuggingFace!")
    print(f"📍 Repository available at: https://huggingface.co/{repo_id}")

except Exception as e:
    print(f"❌ Error uploading files: {e}")
