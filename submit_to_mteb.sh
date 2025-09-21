#!/bin/bash

# SOFIA MTEB Submission Automation Script
# This script automates the MTEB submission process

set -e  # Exit on any error

echo "🚀 Starting SOFIA MTEB Submission Process"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if git is configured
if ! git config --global user.name > /dev/null 2>&1; then
    print_error "Git user name not configured. Please run:"
    echo "git config --global user.name 'Your Name'"
    echo "git config --global user.email 'your.email@example.com'"
    exit 1
fi

if ! git config --global user.email > /dev/null 2>&1; then
    print_error "Git user email not configured."
    exit 1
fi

# Get GitHub username
read -p "Enter your GitHub username: " GITHUB_USER
if [ -z "$GITHUB_USER" ]; then
    print_error "GitHub username is required"
    exit 1
fi

print_status "Using GitHub username: $GITHUB_USER"

# Create temp directory for work
TEMP_DIR="/tmp/sofia_mteb_submission"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

print_status "Working in temporary directory: $TEMP_DIR"

# Step 1: Fork and clone MTEB main repo
print_status "Step 1: Setting up MTEB main repo"

if [ ! -d "mteb" ]; then
    print_status "Cloning MTEB main repo..."
    git clone "https://github.com/$GITHUB_USER/mteb.git" 2>/dev/null || {
        print_warning "Fork not found. Please fork https://github.com/embeddings-benchmark/mteb first"
        print_status "Opening browser to fork page..."
        xdg-open "https://github.com/embeddings-benchmark/mteb/fork" 2>/dev/null || echo "Please visit: https://github.com/embeddings-benchmark/mteb/fork"
        read -p "Press Enter after forking..."
        git clone "https://github.com/$GITHUB_USER/mteb.git"
    }
fi

cd mteb
git checkout main
git pull origin main
git checkout -b add-sofia-model

# Add SOFIA model meta
print_status "Adding SOFIA ModelMeta to mteb/models/overview.py"

MODEL_META_CODE='
# SOFIA Embedding Model
from mteb.model_meta import ModelMeta

sofia_meta = ModelMeta(
    name="MaliosDark/sofia-embedding-v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="8dfaf9f573cff22bafb5a9b8d3dd66565553b5b9",
    release_date="2025-09-21",
    n_parameters=110_000_000,
    memory_usage_mb=382,
    embed_dim=1024,
    license="apache-2.0",
    max_tokens=384,
    reference="https://huggingface.co/MaliosDark/sofia-embedding-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"sentence-transformers/stsb": ["train"], "paws": ["labeled_final"], "banking77": ["train"]},
)
'

# Check if already exists
if ! grep -q "sofia_meta" mteb/models/overview.py; then
    echo "$MODEL_META_CODE" >> mteb/models/overview.py
    print_status "Added SOFIA ModelMeta"
else
    print_warning "SOFIA ModelMeta already exists"
fi

# Commit and push
git add .
git commit -m "Add SOFIA embedding model

- Model: MaliosDark/sofia-embedding-v1
- Type: Sentence Transformer with LoRA fine-tuning
- Base: all-mpnet-base-v2
- Embedding dim: 1024
- License: Apache 2.0

ModelMeta includes all required fields for MTEB integration."
git push origin add-sofia-model

print_status "✅ MTEB main repo PR branch pushed!"
print_status "Create PR: https://github.com/embeddings-benchmark/mteb/compare/main...$GITHUB_USER:add-sofia-model"

# Step 2: Fork and clone MTEB results repo
print_status "Step 2: Setting up MTEB results repo"

cd "$TEMP_DIR"

if [ ! -d "mteb-results" ]; then
    print_status "Cloning MTEB results repo..."
    git clone "https://github.com/$GITHUB_USER/mteb-results.git" 2>/dev/null || {
        print_warning "Fork not found. Please fork https://github.com/embeddings-benchmark/mteb-results first"
        print_status "Opening browser to fork page..."
        xdg-open "https://github.com/embeddings-benchmark/mteb-results/fork" 2>/dev/null || echo "Please visit: https://github.com/embeddings-benchmark/mteb-results/fork"
        read -p "Press Enter after forking..."
        git clone "https://github.com/$GITHUB_USER/mteb-results.git"
    }
fi

cd mteb-results
git checkout main
git pull origin main
git checkout -b add-sofia-results

# Copy results from the project
print_status "Copying evaluation results..."
RESULTS_SOURCE="/home/nexland/sofi-labs/mteb_results"
if [ -d "$RESULTS_SOURCE" ]; then
    cp -r "$RESULTS_SOURCE"/* results/
    print_status "Results copied successfully"
else
    print_error "Results directory not found: $RESULTS_SOURCE"
    print_error "Please run evaluation first:"
    echo "cd /home/nexland/sofi-labs && source .venv/bin/activate"
    echo "python -c \"from mteb import MTEB; from sentence_transformers import SentenceTransformer; model = SentenceTransformer('MaliosDark/sofia-embedding-v1'); evaluation = MTEB(tasks=['STS12', 'STS13', 'BIOSSES']); results = evaluation.run(model, output_folder='./mteb_results')\""
    exit 1
fi

# Commit and push
git add .
git commit -m "Add SOFIA evaluation results

Evaluation results for SOFIA embedding model on:
- STS12: Semantic Textual Similarity
- STS13: Semantic Textual Similarity  
- BIOSSES: Biomedical Semantic Similarity

Model: MaliosDark/sofia-embedding-v1
Revision: 8dfaf9f573cff22bafb5a9b8d3dd66565553b5b9

Results generated using MTEB framework for leaderboard submission."
git push origin add-sofia-results

print_status "✅ MTEB results repo PR branch pushed!"
print_status "Create PR: https://github.com/embeddings-benchmark/mteb-results/compare/main...$GITHUB_USER:add-sofia-results"

# Step 3: Summary
print_status "🎉 Submission process complete!"
echo ""
echo "Next steps:"
echo "1. Create PR for MTEB main repo: https://github.com/embeddings-benchmark/mteb/compare/main...$GITHUB_USER:add-sofia-model"
echo "2. Create PR for MTEB results repo: https://github.com/embeddings-benchmark/mteb-results/compare/main...$GITHUB_USER:add-sofia-results"
echo ""
echo "PR Checklist (include in both PR descriptions):"
echo "✅ I have filled out the ModelMeta object to the extent possible"
echo "✅ I have ensured that my model can be loaded using mteb.get_model(model_name, revision)"
echo "✅ I have tested the implementation works on a representative set of tasks"
echo "✅ The model is public, i.e. is available either as an API or the weights are publicly available to download"
echo ""
print_status "Once both PRs are merged, SOFIA will appear on the MTEB leaderboard within 24 hours!"

cd /home/nexland/sofi-labs
print_status "Returned to project directory"
