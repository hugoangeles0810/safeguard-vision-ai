# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Fall Detection System** using Computer Vision and Pose Estimation. The project detects falls in video streams by analyzing human pose keypoints through temporal sequence classification.

**Pipeline:** Video Input → Pose Extraction (MediaPipe/YOLOv8) → Keypoint Sequence → Temporal Classification (LSTM/Transformer) → Fall/No-Fall Prediction

## Tech Stack

- **Language:** Python 3.10+
- **ML Framework:** PyTorch 2.0+
- **Pose Estimation:** MediaPipe 0.10+ (primary), Ultralytics YOLOv8 8.0+ (alternative)
- **Demo Interface:** Gradio 4.0+
- **Video Processing:** OpenCV, imageio
- **Experiment Tracking:** TensorBoard or Weights & Biases
- **Training Environment:** Google Colab Pro (GPU)

## Repository Structure

```
├── data/
│   ├── raw/              # Original videos (not committed)
│   │   ├── ur_fall/
│   │   └── le2i/
│   ├── processed/        # Extracted keypoints
│   │   ├── keypoints/
│   │   └── metadata.csv
│   └── splits/           # Train/val/test splits
├── src/
│   ├── data/             # Dataset, DataLoader, preprocessing, augmentation
│   ├── models/           # LSTM and Transformer architectures
│   ├── training/         # Training loop, losses, metrics
│   ├── inference/        # Prediction and real-time inference
│   └── pose/             # MediaPipe and YOLOv8 extractors
├── notebooks/            # EDA, experiments, error analysis
├── configs/              # YAML configuration files
├── demo/                 # Gradio demo application
├── results/
│   ├── checkpoints/      # Saved models
│   ├── logs/             # TensorBoard logs
│   └── figures/          # Visualizations
└── docs/                 # Research and technical documentation
```

## Key Datasets

- **UR Fall Detection Dataset:** 70 sequences (30 falls, 40 ADL) from Universidad de Rzeszów
- **Le2i Fall Detection Dataset:** ~200 videos with varied environments
- **MCFD:** Multi-camera perspectives for robustness

**Data Split:** 70% train, 15% validation, 15% test (stratified)

## Architecture Details

### Pose Preprocessing
- Keypoints normalized: centered on hip, scaled by torso height
- Input shape: (batch, T, num_keypoints × 2) where T = 60 frames (2 seconds @ 30fps)
- MediaPipe provides 33 keypoints (or 17 COCO keypoints for YOLOv8)

### Model Architectures

**LSTM Baseline:**
- Bidirectional LSTM with 2 layers, hidden size 128
- Dropout 0.3
- Attention pooling over temporal dimension
- Classifier: Linear(256, 64) → ReLU → Dropout(0.5) → Linear(64, 1) → Sigmoid

**Transformer (Advanced):**
- 2-layer Transformer Encoder
- 4 attention heads, FFN hidden dim 256
- Positional encoding + CLS token pooling
- Lighter and potentially more effective for temporal patterns

### Data Augmentation
- Temporal jittering and speed variation (0.8x-1.2x)
- Horizontal flip of keypoints
- Gaussian noise injection
- Keypoint dropout (simulating occlusions)
- Random temporal cropping

## Critical Metrics

**Priority:** Recall (>0.90) is most critical - missing a real fall is unacceptable
- **Precision:** >0.85 (avoid false alarms)
- **F1-Score:** >0.85
- **AUC-ROC:** >0.90
- **Specificity:** >0.85

**Demo Metrics:**
- Latency: <50ms per frame
- FPS: >20
- Detection time: <1 second from fall start

## Development Commands

This project is in early stages. Once implementation begins, typical commands will include:

```bash
# Environment setup
pip install -r requirements.txt

# Pose extraction
python src/pose/mediapipe_extractor.py --input data/raw/ --output data/processed/

# Training
python src/training/trainer.py --config configs/lstm_baseline.yaml

# Evaluation
python src/training/trainer.py --config configs/lstm_baseline.yaml --mode eval --checkpoint results/checkpoints/best_model.pth

# Demo
python demo/app.py --model results/checkpoints/best_model.pth
```

## Design Principles

1. **Privacy-First:** Pose skeletons don't reveal identity
2. **Interpretability:** Visualize skeleton overlay for debugging
3. **Robustness:** Handle occlusions, different camera angles, varied lighting
4. **Real-time Capable:** Target >20 FPS for demo
5. **False Positive Management:** Include challenging ADL examples (sitting quickly, bending, jumping)

## Critical Implementation Notes

- Use stratified splitting to maintain class balance
- Implement class weights or focal loss due to imbalanced data (more ADL than falls)
- Store pose confidence scores from extractors for quality filtering
- Normalize keypoints relative to torso height for scale invariance
- Use early stopping with patience=10 to prevent overfitting
- Track experiments meticulously (TensorBoard/W&B) - compare architectures systematically

## Common Pitfalls to Avoid

- **Overfitting:** Small dataset (~300 sequences total) - use aggressive regularization
- **False positives on rapid movements:** Include diverse ADL examples (squatting, jumping, sitting)
- **Occlusion failures:** Augment with keypoint dropout, use confidence thresholds
- **Dataset distribution shift:** Mix UR Fall and Le2i datasets during training
- **Forgetting the temporal window:** Falls happen over ~1-2 seconds, sequence length matters

## Timeline Context

This is a 3-week intensive project (~120 hours total, 4-person team):
- **Week 1:** Data pipeline, pose extraction, preprocessing
- **Week 2:** Model development, training, hyperparameter tuning
- **Week 3:** Optimization, demo, presentation

Currently the project appears to be at the initialization stage with project planning complete.
