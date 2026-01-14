# Project Plan: Fall Detection with Pose Estimation

## 1. General Information

| Aspect | Detail |
|---------|---------|
| **Topic** | Fall Detection using Computer Vision |
| **Duration** | 3 intensive weeks (~120 total hours) |
| **Team** | 4 people |
| **Deliverables** | Trained model, presentation, demo (nice to have) |
| **Technical Approach** | Pose estimation + Temporal classifier |

---

## 2. Approach Overview

### 2.1 Proposed Pipeline

```
┌─────────┐    ┌──────────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Video  │ → │ Pose Extraction      │ → │ Keypoint        │ → │ Classification  │
│ (Input) │    │ (MediaPipe/YOLOv8)   │    │ Sequence        │    │ Fall/No-Fall    │
└─────────┘    └──────────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2.2 Approach Rationale

Pose estimation offers several advantages for this problem:

1. **Interpretability**: We can visualize what the model detects (human skeleton)
2. **Transfer learning**: Very robust pre-trained models available
3. **Efficiency**: We reduce problem dimensionality (from pixels to keypoints)
4. **Privacy**: The skeleton doesn't reveal the person's identity
5. **Attractive demo**: Clear visualization of system operation

---

## 3. Detailed Schedule

### 3.1 Week 1: Foundations and Data Pipeline

**Goal:** Have the dataset processed and ready for training.

| Day | Tasks | Hours | Responsible |
|-----|--------|-------|--------------|
| **1** | Set up development environment (conda/venv, dependencies) | 4h | Person A |
| **1** | Create Git repository, folder structure, initial README | 4h | Person B |
| **1** | Download datasets (UR Fall, Le2i) | 4h | Person C |
| **1** | Initial research: reference papers | 4h | Person D |
| **2** | Exploratory data analysis (EDA) | 4h | Persons A, B |
| **2** | Document state of the art in fall detection | 4h | Persons C, D |
| **3** | Implement pose extraction with MediaPipe | 6h | Persons A, B |
| **3** | Implement pose extraction with YOLOv8-Pose | 6h | Persons C, D |
| **4** | Compare MediaPipe vs YOLO results, choose one | 4h | Entire team |
| **4** | Define data format: keypoint structure | 4h | Entire team |
| **5** | Create complete preprocessing pipeline | 8h | Persons A, B |
| **5** | Implement temporal data augmentation | 8h | Persons C, D |
| **6** | Process all videos → keypoints | 6h | Entire team |
| **6** | Create stratified train/val/test splits | 4h | Entire team |
| **7** | Validate processed dataset integrity | 4h | Persons A, B |
| **7** | Document data pipeline | 4h | Persons C, D |
| **7** | Week 1 review, adjust plan if necessary | 2h | Entire team |

**Week 1 Deliverables:**
- [ ] Processed dataset with keypoint sequences
- [ ] Functional and documented preprocessing pipeline
- [ ] State of the art document
- [ ] Well-founded decision on pose extractor

---

### 3.2 Week 2: Model Development and Training

**Goal:** Have a trained model with documented metrics.

| Day | Tasks | Hours | Responsible |
|-----|--------|-------|--------------|
| **8** | Implement DataLoader in PyTorch | 4h | Persons A, B |
| **8** | Implement baseline model (simple LSTM) | 4h | Persons C, D |
| **9** | Define evaluation metrics | 3h | Person A |
| **9** | Implement training loop | 5h | Person B |
| **9** | Configure logging (TensorBoard/W&B) | 4h | Persons C, D |
| **10** | First baseline training | 4h | Entire team |
| **10** | Baseline error analysis | 4h | Entire team |
| **11** | Implement variants: Bidirectional LSTM, GRU | 6h | Persons A, B |
| **11** | Implement variant: Small Transformer | 6h | Persons C, D |
| **12** | Train and compare architectures | 8h | Entire team |
| **13** | Hyperparameter tuning (learning rate, hidden size, etc.) | 8h | Entire team |
| **14** | Select best model, train final version | 4h | Persons A, B |
| **14** | Document experiments and results | 4h | Persons C, D |
| **14** | Week 2 review, prepare week 3 | 2h | Entire team |

**Week 2 Deliverables:**
- [ ] Trained model (saved checkpoint)
- [ ] Documented metrics (accuracy, precision, recall, F1, AUC)
- [ ] Comparison of tested architectures
- [ ] Error analysis and difficult cases

---

### 3.3 Week 3: Optimization, Demo and Presentation

**Goal:** Polished final product, functional demo, presentation ready.

| Day | Tasks | Hours | Responsible |
|-----|--------|-------|--------------|
| **15** | Ablation studies (which components contribute most) | 4h | Persons A, B |
| **15** | Start demo development (basic structure) | 4h | Persons C, D |
| **16** | Final model optimization | 4h | Persons A, B |
| **16** | Demo: integrate model with webcam/video | 4h | Persons C, D |
| **17** | Create result visualizations for presentation | 4h | Persons A, B |
| **17** | Demo: add UI with Gradio/Streamlit | 4h | Persons C, D |
| **18** | Design presentation structure | 4h | Persons A, B |
| **18** | Polish demo, handle edge cases | 4h | Persons C, D |
| **19** | Create presentation slides | 6h | Persons A, B |
| **19** | Demo testing, document usage | 6h | Persons C, D |
| **20** | Integrate everything: presentation + demo | 4h | Entire team |
| **20** | Presentation rehearsal (first round) | 4h | Entire team |
| **21** | Final adjustments based on feedback | 4h | Entire team |
| **21** | Final presentation rehearsal | 2h | Entire team |
| **21** | Buffer for unforeseen issues | 2h | Entire team |

**Week 3 Deliverables:**
- [ ] Final optimized model
- [ ] Functional demo (Gradio/Streamlit)
- [ ] Complete presentation
- [ ] Clean and documented repository
- [ ] README with usage instructions

---

## 4. Technology Stack

### 4.1 Main Tools

| Component | Tool | Suggested Version | Rationale |
|------------|-------------|------------------|---------------|
| **Language** | Python | 3.10+ | ML standard |
| **ML Framework** | PyTorch | 2.0+ | Team experience |
| **Pose extraction** | MediaPipe | 0.10+ | Fast, easy to use |
| **Pose alternative** | Ultralytics YOLOv8 | 8.0+ | More accurate, heavier |
| **Training** | Google Colab Pro | - | Accessible GPU |
| **Visualization** | TensorBoard or W&B | - | Experiment tracking |
| **Demo** | Gradio | 4.0+ | Fast and simple UI |
| **Code versioning** | Git + GitHub | - | Collaboration |
| **Data versioning** | DVC (optional) | - | Reproducibility |

### 4.2 Python Dependencies

```txt
# Core
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0

# Pose estimation
mediapipe>=0.10.0
ultralytics>=8.0.0  # For YOLOv8-Pose

# Visualization and tracking
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.13.0
# wandb>=0.15.0  # Alternative to TensorBoard

# Video processing
opencv-python>=4.8.0
imageio>=2.31.0

# Demo
gradio>=4.0.0
# streamlit>=1.25.0  # Alternative

# Utilities
tqdm>=4.65.0
scikit-learn>=1.3.0
pyyaml>=6.0
```

---

## 5. Datasets

### 5.1 Recommended Datasets

#### Primary Dataset: UR Fall Detection Dataset

| Feature | Detail |
|----------------|---------|
| **Source** | University of Rzeszów, Poland |
| **Content** | 70 sequences (30 falls, 40 ADL) |
| **Cameras** | RGB + Depth (Kinect) |
| **Resolution** | 640×480 |
| **FPS** | 30 |
| **Advantage** | Widely cited, easy comparison with papers |
| **Link** | http://fenix.ur.edu.pl/~mkepski/ds/uf.html |

#### Secondary Dataset: Le2i Fall Detection Dataset

| Feature | Detail |
|----------------|---------|
| **Source** | Université de Bourgogne, France |
| **Content** | ~200 videos in varied scenarios |
| **Scenarios** | Home, office, conference room |
| **Advantage** | More variety of environments and angles |
| **Link** | https://imvia.u-bourgogne.fr/en/database/fall-detection-dataset-2.html |

#### Complementary Dataset: Multiple Cameras Fall Dataset (MCFD)

| Feature | Detail |
|----------------|---------|
| **Content** | Falls from multiple angles |
| **Advantage** | Robustness to different perspectives |

### 5.2 Data Strategy

```
Estimated total data:
├── Falls: ~100-150 sequences
├── Non-falls (ADL): ~150-200 sequences
└── Total: ~300-350 sequences

Proposed split:
├── Train: 70% (~210-245 sequences)
├── Validation: 15% (~45-52 sequences)
└── Test: 15% (~45-52 sequences)

Note: Use stratified split to maintain class proportion
```

### 5.3 Data Augmentation

Recommended temporal augmentation techniques:

1. **Temporal jittering**: Slightly vary timestamps
2. **Speed variation**: Play sequences faster/slower (0.8x - 1.2x)
3. **Random temporal cropping**: Take random subsequences
4. **Horizontal flip**: Mirror the keypoints
5. **Noise injection**: Add Gaussian noise to coordinates
6. **Keypoint dropout**: Simulate partial occlusions

---

## 6. Model Architecture

### 6.1 Keypoint Preprocessing

```python
# Keypoint normalization (pseudocode)

def normalize_keypoints(keypoints):
    """
    Input: keypoints shape (T, num_joints, 2 or 3)

    Steps:
    1. Center on hip (hip center)
    2. Scale by torso height
    3. Optionally: convert to velocities/accelerations
    """

    # Center on hip
    hip_center = (keypoints[:, HIP_LEFT] + keypoints[:, HIP_RIGHT]) / 2
    keypoints_centered = keypoints - hip_center[:, np.newaxis, :]

    # Scale by torso height
    torso_height = np.linalg.norm(
        keypoints[:, NECK] - keypoints[:, HIP_CENTER],
        axis=-1, keepdims=True
    )
    keypoints_normalized = keypoints_centered / (torso_height + 1e-6)

    return keypoints_normalized
```

### 6.2 Baseline Architecture: LSTM

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                               │
│  Keypoint sequence: (batch, T, num_keypoints × 2)           │
│  Example: (32, 60, 34) for 60 frames, 17 keypoints × 2D     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  NORMALIZATION                               │
│  BatchNorm1d or LayerNorm                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    LSTM LAYERS                               │
│  Bidirectional LSTM                                         │
│  - Layers: 2                                                │
│  - Hidden size: 128                                         │
│  - Dropout: 0.3                                             │
│  Output: (batch, T, 256)                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    POOLING                                   │
│  Option A: Last hidden state                                │
│  Option B: Average temporal pooling                         │
│  Option C: Attention pooling (recommended)                  │
│  Output: (batch, 256)                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 CLASSIFIER                                   │
│  Linear(256, 64) → ReLU → Dropout(0.5)                      │
│  Linear(64, 1) → Sigmoid                                    │
│  Output: (batch, 1) → Fall probability                      │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Advanced Architecture: Lightweight Transformer

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT + EMBEDDING                         │
│  Linear projection: (batch, T, input_dim) → (batch, T, 128) │
│  + Positional Encoding (sinusoidal or learnable)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              TRANSFORMER ENCODER (×2 layers)                 │
│  - Attention heads: 4                                       │
│  - FFN hidden: 256                                          │
│  - Dropout: 0.1                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CLS TOKEN POOLING                         │
│  Use [CLS] token as sequence representation                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 CLASSIFIER                                   │
│  Linear(128, 1) → Sigmoid                                   │
└─────────────────────────────────────────────────────────────┘
```

### 6.4 Initial Hyperparameters

| Hyperparameter | Initial Value | Range to Explore |
|----------------|---------------|------------------|
| Learning rate | 1e-3 | 1e-4 to 1e-2 |
| Batch size | 32 | 16, 32, 64 |
| Sequence length (T) | 60 frames (2s @ 30fps) | 30, 60, 90 |
| LSTM hidden size | 128 | 64, 128, 256 |
| LSTM layers | 2 | 1, 2, 3 |
| Dropout | 0.3 | 0.2, 0.3, 0.5 |
| Optimizer | Adam | Adam, AdamW |
| Scheduler | ReduceLROnPlateau | CosineAnnealing |
| Epochs | 100 | Early stopping patience=10 |

---

## 7. Evaluation Metrics

### 7.1 Main Metrics

| Metric | Formula | Importance | Target |
|---------|---------|-------------|----------|
| **Recall** | TP / (TP + FN) | **Critical** - Don't miss real falls | > 0.90 |
| **Precision** | TP / (TP + FP) | High - Avoid false alarms | > 0.85 |
| **F1-Score** | 2 × (P × R) / (P + R) | Main for comparing models | > 0.85 |
| **AUC-ROC** | Area under ROC curve | Overall performance | > 0.90 |
| **Specificity** | TN / (TN + FP) | Important for balance | > 0.85 |

### 7.2 Secondary Metrics (for demo)

| Metric | Target |
|---------|----------|
| **Inference latency** | < 50ms per frame |
| **Processing FPS** | > 20 FPS |
| **Detection time** | < 1 second from fall start |

### 7.3 Expected Confusion Matrix

```
                    Prediction
                 Fall     No Fall
              ┌─────────┬──────────┐
    Fall      │   TP    │    FN    │  ← Minimize FN (critical)
Reality       ├─────────┼──────────┤
    No Fall   │   FP    │    TN    │  ← Minimize FP (important)
              └─────────┴──────────┘
```

---

## 8. Repository Structure

```
fall-detection/
│
├── 📁 data/
│   ├── 📁 raw/                    # Original videos (don't commit)
│   │   ├── ur_fall/
│   │   └── le2i/
│   ├── 📁 processed/              # Extracted keypoints
│   │   ├── keypoints/
│   │   └── metadata.csv
│   └── 📁 splits/                 # Split files
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
│
├── 📁 src/
│   ├── 📁 data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # PyTorch Dataset
│   │   ├── dataloader.py          # DataLoader factory
│   │   ├── preprocessing.py       # Keypoint normalization
│   │   └── augmentation.py        # Data augmentation
│   │
│   ├── 📁 models/
│   │   ├── __init__.py
│   │   ├── lstm.py                # LSTM model
│   │   ├── transformer.py         # Transformer model
│   │   └── utils.py               # Model utilities
│   │
│   ├── 📁 training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training loop
│   │   ├── losses.py              # Loss functions
│   │   └── metrics.py             # Metrics calculation
│   │
│   ├── 📁 inference/
│   │   ├── __init__.py
│   │   ├── predictor.py           # Prediction class
│   │   └── realtime.py            # Real-time inference
│   │
│   └── 📁 pose/
│       ├── __init__.py
│       ├── mediapipe_extractor.py # MediaPipe extraction
│       └── yolo_extractor.py      # YOLOv8 extraction
│
├── 📁 notebooks/
│   ├── 01_eda.ipynb               # Exploratory analysis
│   ├── 02_pose_extraction.ipynb   # Extraction tests
│   ├── 03_model_experiments.ipynb # Model experiments
│   └── 04_error_analysis.ipynb    # Error analysis
│
├── 📁 configs/
│   ├── default.yaml               # Default configuration
│   ├── lstm_baseline.yaml         # LSTM config
│   └── transformer.yaml           # Transformer config
│
├── 📁 demo/
│   ├── app.py                     # Gradio application
│   ├── utils.py                   # Demo utilities
│   └── assets/                    # Visual resources
│
├── 📁 results/
│   ├── 📁 checkpoints/            # Saved models
│   ├── 📁 logs/                   # TensorBoard logs
│   └── 📁 figures/                # Visualizations
│
├── 📁 docs/
│   ├── state_of_the_art.md        # Research document
│   └── architecture.md            # Technical documentation
│
├── 📁 tests/                      # Unit tests (optional)
│
├── .gitignore
├── README.md                      # Main documentation
├── requirements.txt               # Dependencies
├── setup.py                       # Package installation
└── Makefile                       # Useful commands (optional)
```

---

## 9. Risk Analysis

### 9.1 Risk Matrix

| Risk | Probability | Impact | Mitigation |
|--------|--------------|---------|------------|
| **Insufficient data** | Medium | High | Combine multiple datasets, aggressive data augmentation |
| **Overfitting** | High | High | Regularization, dropout, early stopping, cross-validation |
| **Imbalanced classes** | High | Medium | Class weights, focal loss, fall oversampling |
| **Pose estimation fails on occlusions** | Medium | Medium | Augmentation with keypoint dropout, use confidences |
| **Model too slow for demo** | Low | Medium | MediaPipe is fast; use small model; quantization |
| **False positives with sudden movements** | High | High | Include examples of "sitting quickly", "bending", "jumping" |
| **Variability between datasets** | Medium | Medium | Robust normalization, train with mixed data |
| **Team collaboration issues** | Low | High | Short daily meetings, modular code, PR reviews |

### 9.2 Contingency Plan

**If data is insufficient:**
- Generate synthetic data by rotating/scaling existing poses
- Use action recognition datasets (NTU RGB+D) filtering relevant classes

**If severe overfitting occurs:**
- Simplify architecture (fewer layers, fewer hidden units)
- Use k-fold cross-validation
- Try more aggressive L2 regularization

**If demo doesn't work in real-time:**
- Process every N frames instead of all
- Use lighter model just for demo
- Show demo with pre-recorded video

---

## 10. References and Resources

### 10.1 Fundamental Papers

1. **"A Survey on Vision-based Fall Detection"** - Complete state of the art review
2. **"Skeleton-based Action Recognition with Shift Graph Convolutional Network"** - ST-GCN and variants
3. **"Pose Estimation and Fall Detection using Deep Learning"** - Similar approach to proposed
4. **"MediaPipe: A Framework for Building Perception Pipelines"** - Original MediaPipe paper

### 10.2 Reference Repositories

- MediaPipe Pose: https://github.com/google/mediapipe
- YOLOv8: https://github.com/ultralytics/ultralytics
- PyTorch Lightning (optional): https://github.com/Lightning-AI/lightning

### 10.3 Useful Tutorials

- MediaPipe Pose estimation: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
- PyTorch LSTM for sequences: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
- Gradio quickstart: https://www.gradio.app/guides/quickstart

---

## 11. Final Checklist

### Week 1

- [ ] Environment configured and working for everyone
- [ ] Datasets downloaded and explored
- [ ] Pose extractor implemented and validated
- [ ] Complete preprocessing pipeline
- [ ] Keypoint dataset generated
- [ ] Train/val/test splits created

### Week 2

- [ ] DataLoader working
- [ ] Baseline model implemented
- [ ] First training completed
- [ ] Evaluation metrics implemented
- [ ] At least 3 architectures compared
- [ ] Best model selected
- [ ] Experiments documented

### Week 3

- [ ] Ablation studies completed
- [ ] Final optimized model
- [ ] Functional demo
- [ ] Presentation created
- [ ] Presentation rehearsals done
- [ ] Clean and documented repository
- [ ] Complete README with instructions

---

## 12. Immediate Next Steps

1. **Today:** Create repository, configure shared environment
2. **Tomorrow:** Download UR Fall Dataset and Le2i, do exploratory analysis
3. **Day 3:** Have first working script that extracts poses from a video

---

*Document generated for the Fall Detection project*
*Version: 1.0*
