# SafeGuard Vision AI - Fall Detection System

A computer vision system for detecting falls using pose estimation and temporal classification.

## Overview

This project implements a fall detection system that analyzes human pose keypoints over time to identify falls in video streams. The system uses state-of-the-art pose estimation models (MediaPipe or YOLOv8-Pose) combined with temporal classifiers (LSTM or Transformer) to achieve high accuracy fall detection.

**Pipeline:**
```
Video Input → Pose Extraction → Keypoint Sequence → Temporal Classification → Fall/No-Fall Prediction
```

## Key Features

- **Privacy-Preserving**: Uses pose skeletons instead of raw video for classification
- **Real-Time Capable**: Optimized for >20 FPS processing
- **High Accuracy**: Target >90% recall (critical for safety)
- **Interpretable**: Visual pose overlay for debugging and demonstration
- **Flexible Architecture**: Supports both LSTM and Transformer models

## Project Structure

```
├── data/                    # Dataset storage
│   ├── raw/                 # Raw video files
│   ├── processed/           # Extracted keypoints
│   └── splits/              # Train/val/test splits
├── src/                     # Source code
│   ├── data/                # Dataset and preprocessing
│   ├── models/              # Model architectures
│   ├── training/            # Training loop and metrics
│   ├── inference/           # Prediction and real-time detection
│   └── pose/                # Pose extraction (MediaPipe/YOLO)
├── notebooks/               # Jupyter notebooks for exploration
├── configs/                 # YAML configuration files
├── demo/                    # Gradio demo application
├── results/                 # Checkpoints, logs, and figures
└── docs/                    # Documentation
```

## Installation

### Prerequisites

- Python 3.10+
- Google Colab account (recommended for training with GPU)
- Google Drive account (for dataset and model storage)
- Git

### Setup

#### Option 1: Google Colab (Recommended)

1. Open any notebook in the `notebooks/` directory in Google Colab
2. The notebook will automatically:
   - Mount your Google Drive
   - Clone the repository
   - Install dependencies
   - Setup project paths

3. Upload your datasets to: `Google Drive/MyDrive/safeguard-vision-ai/data/raw/`

#### Option 2: Local Setup

1. Clone the repository:
```bash
git clone https://github.com/hugoangeles0810/safeguard-vision-ai.git
cd safeguard-vision-ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### 1. Data Preparation

Download and place datasets in `data/raw/`:
- [UR Fall Detection Dataset](http://fenix.ur.edu.pl/~mkepski/ds/uf.html)
- [Le2i Fall Detection Dataset](https://imvia.u-bourgogne.fr/en/database/fall-detection-dataset-2.html)

### 2. Extract Pose Keypoints

```bash
python src/pose/mediapipe_extractor.py --input data/raw/ --output data/processed/
```

### 3. Train a Model

```bash
# LSTM baseline
python src/training/trainer.py --config configs/lstm_baseline.yaml

# Transformer
python src/training/trainer.py --config configs/transformer.yaml
```

### 4. Run Demo

```bash
python demo/app.py --model results/checkpoints/best_model.pth --config configs/default.yaml
```

## Datasets

The project uses the following fall detection datasets:

| Dataset | Videos | Falls | ADL | Resolution | FPS |
|---------|--------|-------|-----|------------|-----|
| UR Fall Detection | 70 | 30 | 40 | 640×480 | 30 |
| Le2i Fall Detection | ~200 | ~100 | ~100 | Varied | Varied |

**Data Split:** 70% train / 15% validation / 15% test (stratified)

## Model Architectures

### LSTM Baseline
- Bidirectional LSTM (2 layers, 128 hidden units)
- Attention pooling over temporal dimension
- Dropout for regularization
- Binary classification head

### Transformer
- 2-layer Transformer Encoder
- 4 attention heads
- Positional encoding
- CLS token pooling

## Target Performance Metrics

| Metric | Target | Priority |
|--------|--------|----------|
| **Recall** | >90% | **Critical** (don't miss falls) |
| Precision | >85% | High |
| F1-Score | >85% | High |
| AUC-ROC | >90% | Medium |
| Specificity | >85% | Medium |
| Latency | <50ms | For real-time |
| FPS | >20 | For real-time |

## Configuration

All configurations are stored in `configs/*.yaml`:
- `default.yaml`: Base configuration
- `lstm_baseline.yaml`: LSTM-specific settings
- `transformer.yaml`: Transformer-specific settings

Key configuration parameters:
```yaml
data:
  sequence_length: 60  # 2 seconds at 30 FPS
  batch_size: 32

model:
  type: "lstm"  # or "transformer"
  hidden_dim: 128

training:
  num_epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 15
```

## Development Notebooks

All notebooks are designed to work seamlessly in Google Colab with automatic Google Drive integration:

1. `01_eda.ipynb`: Exploratory data analysis
2. `02_pose_extraction.ipynb`: Pose extraction comparison (MediaPipe vs YOLOv8)
3. `03_model_experiments.ipynb`: Model training and comparison with GPU
4. `04_error_analysis.ipynb`: Error analysis and improvements

**To use in Colab:**
1. Upload notebooks to Google Colab
2. Run the first cells to mount Drive and setup environment
3. Notebooks will automatically access data from your Google Drive

## Project Timeline

This is a 3-week intensive project:
- **Week 1**: Data pipeline and pose extraction
- **Week 2**: Model development and training
- **Week 3**: Optimization, demo, and presentation

## Citation

If you use this code for research, please cite:

```bibtex
@misc{safeguard-vision-ai,
  author = {Fall Detection Team},
  title = {SafeGuard Vision AI: Fall Detection System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/hugoangeles0810/safeguard-vision-ai}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- MediaPipe by Google for pose estimation
- Ultralytics for YOLOv8-Pose
- Dataset authors: UR Fall Detection, Le2i Fall Detection
