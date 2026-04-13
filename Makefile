# Data Center Energy Impact Analysis - Makefile
# Run: make <target>

.PHONY: help install website model scrape train all clean start

# Default target
help:
	@echo "╔═══════════════════════════════════════════════════════════════╗"
	@echo "║     Data Center Energy Impact Analysis - Quick Commands      ║"
	@echo "╠═══════════════════════════════════════════════════════════════╣"
	@echo "║  make start       ONE COMMAND - install deps + start website ║"
	@echo "║  make install     Install all Python dependencies            ║"
	@echo "║  make website     Start local website (http://localhost:8080)║"
	@echo "║  make model       Run the XGBoost prediction model           ║"
	@echo "║  make scrape      Run the data center scraper                ║"
	@echo "║  make train       Train the ML model from scratch            ║"
	@echo "║  make all         Run full pipeline (scrape → train → web)   ║"
	@echo "║  make clean       Remove generated files                     ║"
	@echo "╚═══════════════════════════════════════════════════════════════╝"

# ONE COMMAND TO RULE THEM ALL
start: install
	@echo ""
	@echo "🌐 Starting local server at http://localhost:8080"
	@echo "   Press Ctrl+C to stop"
	@echo ""
	cd website && python3 -m http.server 8080

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

# Start the website
website:
	@echo "🌐 Starting local server at http://localhost:8080"
	@echo "   Press Ctrl+C to stop"
	cd website && python3 -m http.server 8080

# Run the ML model (XGBoost BA-level predictor)
model:
	@echo "🤖 Running XGBoost model..."
	cd models/scripts && python3 ba_multiyear_predictor.py

# Run scrapers (WARNING: takes ~2 hours with rate limiting)
scrape:
	@echo "🕷️  Running data center scraper..."
	@echo "   ⚠️  This takes ~2 hours due to rate limiting"
	python3 specs_scraper.py

# Train ML model from scratch
train:
	@echo "🎯 Training XGBoost model..."
	cd models/scripts && python3 ba_multiyear_predictor.py

# Generate website data files
generate-data:
	@echo "📊 Generating website data files..."
	cd website && python3 generate_state_data_v2.py
	cd website && python3 generate_all_dcs.py
	@echo "✅ Data files generated!"

# Full pipeline
all: install scrape train generate-data website

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleaned!"

# Quick test - check if everything is set up
test:
	@echo "🔍 Testing setup..."
	@python3 -c "import pandas; print('✅ pandas')"
	@python3 -c "import numpy; print('✅ numpy')"
	@python3 -c "import sklearn; print('✅ scikit-learn')"
	@python3 -c "import xgboost; print('✅ xgboost')"
	@python3 -c "import requests; print('✅ requests')"
	@echo "🎉 All dependencies installed correctly!"
