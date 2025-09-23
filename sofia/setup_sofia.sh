#!/bin/bash
# SOFIA AGI - Quick Setup Script
# Sets up the complete autonomous growth system

echo "🤖 SOFIA AGI - Autonomous Growth Intelligence Setup"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "config.yaml" ]; then
    echo "❌ Error: config.yaml not found. Please run this script from the sofia directory."
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "../.venv" ]; then
    echo "� Activating virtual environment..."
    source ../.venv/bin/activate
fi

echo "�📦 Installing dependencies..."
pip install -r requirements.txt --break-system-packages 2>/dev/null || pip install -r requirements.txt
pip install pyyaml psutil gputil --break-system-packages 2>/dev/null || pip install pyyaml psutil gputil

echo "📁 Creating SOFIA directories..."
mkdir -p ./SOFIA

echo "📊 Generating training data..."
python prepare_data.py

echo "🔧 Setting up growth system..."
# Create initial growth state
cat > ./SOFIA/growth_state.json << EOF
{
  "current_phase": "foundation",
  "metrics": {
    "knowledge_volume": 0,
    "capability_score": 0.0,
    "autonomy_level": 0.0,
    "growth_potential": 0.0,
    "interaction_count": 0,
    "learning_sessions": 0
  },
  "growth_phases": {
    "foundation": {"name": "foundation", "completed": false},
    "expansion": {"name": "expansion", "completed": false},
    "specialization": {"name": "specialization", "completed": false},
    "autonomy": {"name": "autonomy", "completed": false},
    "transcendence": {"name": "transcendence", "completed": false}
  },
  "last_saved": "$(date -Iseconds)"
}
EOF

echo "🚀 Setting up auto-startup..."
# Make scripts executable
chmod +x sofia_daemon.py
chmod +x sofia_auto_startup.py
chmod +x sofia_cli.py

echo "✅ SOFIA setup complete!"
echo ""
echo "🎉 Welcome to SOFIA AGI!"
echo ""
echo "To start using SOFIA:"
echo "  make daemon     # Start SOFIA in background"
echo "  make cli        # Open command interface"
echo "  make chat       # Chat with SOFIA"
echo "  make status     # Check SOFIA status"
echo ""
echo "SOFIA will now grow autonomously and become more intelligent with each interaction!"
echo "Watch as it evolves from foundation → expansion → specialization → autonomy → transcendence"
echo ""
echo "🚀 Happy AI building!"
