# SOFIA AGI - Autonomous Growth Intelligence
# Makefile for complete SOFIA system management

.PHONY: help install train chat daemon startup status stop clean growth

# Default target
help:
	@echo "🤖 SOFIA AGI - Autonomous Growth Intelligence"
	@echo "=========================================="
	@echo "Available commands:"
	@echo "  make install    - Install all dependencies"
	@echo "  make train      - Train SOFIA with AGI capabilities"
	@echo "  make daemon     - Start SOFIA daemon (background)"
	@echo "  make startup    - Start auto-startup system"
	@echo "  make chat       - Start interactive chat with SOFIA"
	@echo "  make cli        - Start SOFIA command line interface"
	@echo "  make status     - Show SOFIA status"
	@echo "  make stop       - Stop SOFIA daemon"
	@echo "  make growth     - Force SOFIA growth"
	@echo "  make clean      - Clean all generated files"
	@echo "  make test       - Run tests"
	@echo ""

# Install dependencies
install:
	@echo "📦 Installing SOFIA dependencies..."
	pip install -r requirements.txt
	pip install pyyaml psutil gputil
	@echo "✅ Dependencies installed"

# Train SOFIA
train:
	@echo "🎓 Training SOFIA with AGI capabilities..."
	python train_sofia.py --config config.yaml --agi-training

# Start SOFIA daemon
daemon:
	@echo "🔄 Starting SOFIA daemon..."
	python sofia_daemon.py

# Start auto-startup system
startup:
	@echo "🚀 Starting SOFIA auto-startup system..."
	python sofia_auto_startup.py monitor

# Interactive chat
chat:
	@echo "💬 Starting chat with SOFIA..."
	python chat_sim.py

# Command line interface
cli:
	@echo "💻 Starting SOFIA CLI..."
	python sofia_cli.py

# Show status
status:
	@echo "📊 SOFIA Status:"
	@echo "==============="
	@python sofia_auto_startup.py status
	@echo ""
	@if [ -f ./SOFIA/growth_state.json ]; then \
		echo "Growth Status:"; \
		python -c "import json; s=json.load(open('./SOFIA/growth_state.json')); print(f'Phase: {s[\"current_phase\"]}'); m=s['metrics']; print(f'Capability: {m[\"capability_score\"]:.1f}/100'); print(f'Interactions: {m[\"interaction_count\"]}')"; \
	else \
		echo "No growth state found"; \
	fi

# Stop SOFIA
stop:
	@echo "🛑 Stopping SOFIA..."
	python sofia_auto_startup.py stop

# Force growth
growth:
	@echo "🚀 Forcing SOFIA growth..."
	python -c "from sofia_growth_system import SOFIAGrowthSystem; import yaml; c=yaml.safe_load(open('config.yaml')); gs=SOFIAGrowthSystem(c); gs.force_growth()"

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf ./SOFIA/
	rm -f ./sofia/data/pairs.jsonl
	rm -f *.log
	@echo "✅ Cleaned"

# Run tests
test:
	@echo "🧪 Running SOFIA tests..."
	python test_agi_training.py

# Quick setup for new installations
setup: install
	@echo "⚙️  Setting up SOFIA..."
	mkdir -p ./SOFIA
	@echo "Generating initial training data..."
	python prepare_data.py
	@echo "✅ SOFIA setup complete"
	@echo ""
	@echo "🚀 To start SOFIA:"
	@echo "  make daemon    # Start in background"
	@echo "  make cli       # Interactive interface"
	@echo "  make chat      # Chat with SOFIA"

# Full system startup
start-all: daemon startup
	@echo "🎉 SOFIA fully started!"
	@echo "Use 'make cli' to interact with SOFIA"

# Emergency stop all
stop-all: stop
	@echo "🛑 All SOFIA processes stopped"

# Show growth progress
progress:
	@echo "📈 SOFIA Growth Progress:"
	@echo "========================"
	@if [ -f ./SOFIA/growth_state.json ]; then \
		python -c "\
		import json; \
		s = json.load(open('./SOFIA/growth_state.json')); \
		print(f'Current Phase: {s[\"current_phase\"]}'); \
		m = s['metrics']; \
		print(f'Capability Score: {m[\"capability_score\"]:.1f}/100'); \
		print(f'Autonomy Level: {m[\"autonomy_level\"]:.1f}/100'); \
		print(f'Knowledge Volume: {m[\"knowledge_volume\"]} KB'); \
		print(f'Interactions: {m[\"interaction_count\"]}'); \
		print(); \
		print('Completed Phases:'); \
		phases = s.get('growth_phases', {}); \
		completed = [name for name, p in phases.items() if p.get('completed')]; \
		print(', '.join(completed) if completed else 'None'); \
		"; \
	else \
		echo "No growth data available"; \
	fi

# Backup SOFIA state
backup:
	@echo "💾 Backing up SOFIA state..."
	mkdir -p backup
	cp -r ./SOFIA/ backup/ 2>/dev/null || true
	cp config.yaml backup/
	cp *.py backup/
	@echo "✅ Backup created in ./backup/"

# Restore from backup
restore:
	@echo "🔄 Restoring SOFIA from backup..."
	@if [ -d backup ]; then \
		cp -r backup/SOFIA/ ./ 2>/dev/null || true; \
		echo "✅ Restored from backup"; \
	else \
		echo "❌ No backup found"; \
	fi
