.PHONY: help install clean test lint format

help:
	@echo "SafeGuard Vision AI - Fall Detection System"
	@echo ""
	@echo "Available commands:"
	@echo "  make install        Install dependencies"
	@echo "  make clean          Clean generated files and caches"
	@echo "  make test           Run tests"
	@echo "  make lint           Run linter (flake8)"
	@echo "  make format         Format code with black"
	@echo "  make demo           Run Gradio demo"
	@echo "  make train-lstm     Train LSTM baseline"
	@echo "  make train-transformer  Train Transformer model"
	@echo "  make extract-pose   Extract poses from raw videos"

install:
	pip install -r requirements.txt
	pip install -e .

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf build/ dist/

test:
	pytest tests/ -v

lint:
	flake8 src/ --max-line-length=100

format:
	black src/ demo/ --line-length=100

demo:
	python demo/app.py --model results/checkpoints/best_model.pth --config configs/default.yaml

train-lstm:
	python src/training/trainer.py --config configs/lstm_baseline.yaml

train-transformer:
	python src/training/trainer.py --config configs/transformer.yaml

extract-pose:
	python src/pose/mediapipe_extractor.py --input data/raw/ --output data/processed/
