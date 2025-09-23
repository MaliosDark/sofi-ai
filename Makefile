# SOFIA AGI Training Pipeline
.PHONY: help setup data-prep train-hard-negatives augment-data train train-agi evaluate serve clean daemon cli chat status growth backup restore progress

help:
	@echo "SOFIA AGI Training Commands:"
	@echo "  setup              - Install dependencies and setup environment"
	@echo "  data-prep          - Prepare training data"
	@echo "  train-hard-negatives - Mine hard negative examples"
	@echo "  augment-data       - Augment training data for better generalization"
	@echo "  train              - Train SOFIA model with advanced techniques"
	@echo "  train-agi          - Train SOFIA with full AGI capabilities (emotional, RL, conversation)"
	@echo "  evaluate           - Run comprehensive evaluation"
	@echo "  serve              - Start API server"
	@echo "  daemon             - Start SOFIA autonomous daemon"
	@echo "  cli                - Open SOFIA command interface"
	@echo "  chat               - Chat with SOFIA"
	@echo "  status             - Check SOFIA status"
	@echo "  growth             - Trigger SOFIA growth cycle"
	@echo "  backup             - Backup SOFIA state"
	@echo "  restore            - Restore SOFIA state"
	@echo "  progress           - Show SOFIA progress"
	@echo "  clean              - Clean temporary files"

setup:
	pip install -r sofia/requirements.txt
	pip install mteb sentence-transformers faiss-cpu

data-prep:
	@echo "Preparing training data..."
	cd sofia && python prepare_data.py

train-hard-negatives:
	@echo "Mining hard negative examples..."
	cd sofia && python mine_hard_negatives.py \
		--model Qwen/Qwen2.5-1.5B-Instruct \
		--bm25_k 100 \
		--faiss_k 100 \
		--per_pos 3 \
		--diversity_threshold 0.8

augment-data:
	@echo "Augmenting training data..."
	cd sofia && python augment_data.py \
		--input ./data/pairs.jsonl \
		--output ./data/pairs_augmented.jsonl \
		--factor 3

train:
	@echo "Training SOFIA AGI model..."
	cd sofia && python train_sofia.py \
		--config config.yaml \
		--train ./data/pairs_augmented.jsonl \
		--out ./SOFIA

train-agi:
	@echo "Training SOFIA with full AGI capabilities..."
	@echo "This includes: emotional intelligence, reinforcement learning, conversational training"
	cd sofia && python train_sofia.py \
		--config config.yaml \
		--train ./data/pairs_augmented.jsonl \
		--out ./SOFIA-AGI \
		--agi-training

evaluate:
	@echo "Running comprehensive evaluation..."
	cd sofia && python eval_sofia.py \
		--model ./SOFIA \
		--config config.yaml

serve:
	@echo "Starting SOFIA API server..."
	cd sofia && uvicorn serve_api:app --host 0.0.0.0 --port 8000

# Legacy commands for compatibility
eval:
	python eval_mteb.py MaliosDark/sofia-embedding-v1
	python eval_mteb.py ./SOFIA-v2

index:
	python build_index.py
	python search.py "best burgers"

train-lora:
	python train_lora_kd.py

infer:
	echo "machine learning is awesome" | python sofia_infer.py query

onnx:
	python export_onnx.py

clean:
	@echo "Cleaning temporary files..."
	rm -rf sofia/__pycache__/
	rm -rf sofia/data/temp/
	rm -rf ./results/
	rm -f sofia/data/pairs_hard.jsonl
	rm -f sofia/data/pairs_augmented.jsonl

# Autonomous Growth System Commands
daemon:
	@echo "Starting SOFIA autonomous daemon..."
	cd sofia && ./launch_daemon.sh

cli:
	@echo "Opening SOFIA command interface..."
	cd sofia && python sofia_cli.py

chat:
	@echo "Starting chat with SOFIA..."
	python chat_sim.py

status:
	@echo "Checking SOFIA status..."
	cd sofia && python sofia_cli.py status

growth:
	@echo "Triggering SOFIA growth cycle..."
	cd sofia && python sofia_growth_system.py --trigger-growth

backup:
	@echo "Backing up SOFIA state..."
	cd sofia && python sofia_cli.py backup

restore:
	@echo "Restoring SOFIA state..."
	cd sofia && python sofia_cli.py restore

progress:
	@echo "Showing SOFIA progress..."
	cd sofia && python sofia_cli.py progress

startup:
	@echo "Starting SOFIA auto-startup system..."
	cd sofia && python sofia_auto_startup.py
