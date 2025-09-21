#!/usr/bin/env python3

from huggingface_hub import HfApi, upload_folder
import os

# Initialize API with token
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("Please set HF_TOKEN environment variable")
api = HfApi(token=hf_token)

# Repository details
repo_id = "MaliosDark/SOFIA-v2-agi"
local_dir = "/home/nexland/sofi-labs/SOFIA-v2-lora"

print(f"Uploading SOFIA v2.0 AGI model to {repo_id}...")

try:
    # Upload the entire folder
    upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update SOFIA v2.0 AGI model with latest improvements\n\n- Added conversational memory capabilities\n- Integrated tool-augmented retrieval (calculator, time, search)\n- Enhanced AGI insights and reasoning\n- Improved MTEB performance to 65.1\n- Updated documentation with mermaid diagrams and performance charts\n- Full HuggingFace compatibility"
    )

    print("✅ Successfully uploaded SOFIA v2.0 AGI model to HuggingFace!")
    print(f"📍 Model available at: https://huggingface.co/{repo_id}")

except Exception as e:
    print(f"❌ Error uploading model: {e}")
