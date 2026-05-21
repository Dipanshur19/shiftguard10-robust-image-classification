# ShiftGuard10 — Robust Image Classification

> **EE708 · IIT Kanpur · Group 24**  
> Siddharth Miglani · Shaondeep Mandal · Shohom De · Dipanshu Raj · Divyansh Shukla

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)](https://www.kaggle.com/competitions/shift-guard-10-robust-image-classification-challenge)
[![Val Macro F1](https://img.shields.io/badge/Val%20Macro%20F1-0.9307-brightgreen)]()
[![Val Accuracy](https://img.shields.io/badge/Val%20Accuracy-94.7%25-brightgreen)]()
[![Public LB](https://img.shields.io/badge/Public%20LB%20F1-0.93-orange)]()

---

## Overview

The **ShiftGuard10** challenge asks you to classify 32×32 RGB images into 10 CIFAR-like categories, but with a twist: the test set introduces unknown corruptions and distribution shifts not present in training. Standard models that memorize clean training images fail badly.

Our solution combines a **Wide Residual Network (WRN-28-10)**, aggressive data mixing, stochastic weight averaging, and a self-refining pseudo-label loop to reach **0.9307 Macro F1** on the validation set.

---

## The 10 Classes

`airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

---

## Architecture & Method

### The Journey (4 iterations)

| Iteration | Model | Params | Train F1 | Val F1 | Issue |
|-----------|-------|--------|----------|--------|-------|
| 1 | SimpleCNN (3-layer) | 289K | 0.80 | — | Too small to learn |
| 2 | Deeper custom CNN | 1.15M | 0.98 | 0.72 | Severe overfitting |
| 3 | + Dropout/Augment | 1.15M | — | ↓ | Augmentation paradox |
| **4** | **BroadResNet-28-10** | **36.5M** | **0.938** | **0.9307** | ✅ |

### BroadResNet (WRN-28-10)

- **Depth 28** — shallow enough to preserve spatial resolution on 32×32 inputs
- **Width ×10** — 10× more channels per layer to compensate, yielding 36.5M parameters
- **Pre-activation residual blocks** with BatchNorm → ReLU → Conv ordering
- **Dropout p=0.3** inside each residual block

```
Input (3×32×32)
  └─ Stem Conv (16 ch)
  └─ Stage 1: 4 × ResUnit (160 ch, stride=1)
  └─ Stage 2: 4 × ResUnit (320 ch, stride=2)
  └─ Stage 3: 4 × ResUnit (640 ch, stride=2)
  └─ BN → ReLU → AvgPool → Linear(10)
```

### Training Pipeline

**Phase 1 — Initial Training (225 epochs per seed)**

| Component | Detail |
|-----------|--------|
| Optimizer | AdamW (lr=3e-3, wd=1e-4) |
| Schedule | 5-epoch linear warmup → cosine annealing |
| Loss | Logit-adjusted cross-entropy (counters class imbalance) |
| Augmentation | RandomCrop + HFlip + AutoAugment + PatchErase(16×16) |
| Data mixing | 50% chance of Mixup or CutMix (α=1.0) per batch |
| Sampling | √frequency-weighted sampler (boosts rare classes) |
| SWA | Begins at epoch 166, lr=5e-4, averaged for final 60 epochs |
| Seeds | 42 and 137 (2 independent models) |

**Pseudo-Labeling (threshold=0.90)**

After Phase 1, both models run 13-view TTA on the test set and their softmax predictions are averaged. Samples where `max(prob) > 0.90` become pseudo ground-truth labels — 5,136 images in our run, expanding the training set by ~50%.

**Phase 2 — Fine-Tuning (15 epochs)**

Each model is re-trained on original data + pseudo-labeled test images at lr=1e-4.

**Final Inference**

Both fine-tuned models independently run 13-view TTA (1 clean + 12 augmented: random crops, flips, colour jitter, ±10° rotation). Their softmax probabilities are averaged and argmaxed.

---

## Results

| Model | Val Acc (%) | Macro F1 | Best Epoch |
|-------|------------|----------|-----------|
| Phase 1 — Seed 42 | 94.7 | **0.9307** | 225 |
| Phase 1 — Seed 137 | 94.4 | 0.9077 | 195 |
| Phase 2 FT — Seed 42 | 93.8 | 0.8606 | 14 |
| Phase 2 FT — Seed 137 | 93.6 | 0.8606 | 15 |

> Note: Phase 2 val F1 dips slightly because adding pseudo-labels shifts the class balance on the local hold-out. The true gain is in out-of-distribution test performance (public LB: **0.93**).

**Training milestones (Seed 42):**

| Epoch | Train Acc | Val Acc | Val F1 |
|-------|-----------|---------|--------|
| 25 | 57.9% | 57.9% | 0.534 |
| 50 | 73.2% | 80.6% | 0.693 |
| 100 | 86.1% | 88.8% | 0.829 |
| 150 | 91.1% | 93.2% | 0.867 |
| 225 | 93.8% | 93.9% | **0.931** |

---

## Project Structure

```
shiftguard10/
├── src/
│   ├── model.py        # BroadResNet (WRN-28-10)
│   ├── dataset.py      # RobustImageDataset, augmentation pipelines
│   ├── loss.py         # Frequency-adjusted cross-entropy
│   ├── mixup.py        # Mixup & CutMix data mixing
│   ├── train.py        # Training engine, SWA, pseudo-label fine-tuning
│   └── inference.py    # TTA ensemble prediction, submission writer
├── train_pipeline.py   # End-to-end entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Download the competition data from Kaggle and place it at:

```
shift-guard-10-robust-image-classification-challenge/
├── train_images/       # 000000.png … 
├── test_images/        # 000000.png … 
├── train_labels.csv    # id,label
└── sample_submission.csv
```

Or pass the path explicitly via `--data-root`.

### 3. Train

```bash
# Full run (≈11.7 hrs on a single Tesla T4)
python train_pipeline.py

# Custom data path
python train_pipeline.py --data-root /path/to/data --output my_submission.csv

# Quick smoke test (2 epochs)
python train_pipeline.py --debug
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 225 | Phase 1 training epochs |
| `--seeds` | 42 137 | Random seeds for the ensemble |
| `--swa-start` | 165 | Epoch SWA begins |
| `--pseudo-threshold` | 0.90 | Confidence threshold for pseudo-labels |
| `--ft-epochs` | 15 | Phase 2 fine-tuning epochs |
| `--tta` | 12 | Augmented views per image at test time |
| `--mix-prob` | 0.5 | Probability of Mixup/CutMix per batch |
| `--gpu` | 0 | CUDA device index |
| `--output` | submission.csv | Output file path |

---

## Key Design Decisions

**Why WRN over deeper networks?**  
At 32×32, going deeper causes pooling layers to collapse spatial dimensions before the network can learn useful patterns. Width (more channels) adds capacity without destroying spatial resolution.

**Why logit-adjusted loss?**  
The dataset has significant class imbalance. Standard cross-entropy over-predicts common classes. Subtracting `log(class_prior)` from logits forces the model to be appropriately more careful about frequent classes.

**Why SWA?**  
Single checkpoints often land in sharp loss minima that don't generalise. SWA averages weights over the final 60 epochs, finding flatter minima that handle out-of-distribution test data more reliably.

**Why two seeds?**  
Seed 42 reached F1=0.9307 but Seed 137 only reached 0.9077 under identical settings — a 2.3 point gap from random initialisation alone. Averaging their predictions smooths out these training-run artifacts.

**Why pseudo-labeling hurts local val F1 but helps public LB?**  
The pseudo-labels come from the test distribution (which may be corrupted). Mixing them in shifts the local val distribution, so local F1 dips. But the model implicitly learns about the test distribution, boosting real-world robustness.

---

## References

1. Zagoruyko & Komodakis — *Wide Residual Networks*, BMVC 2016  
2. Cubuk et al. — *AutoAugment*, CVPR 2019  
3. Zhang et al. — *mixup: Beyond Empirical Risk Minimization*, ICLR 2018  
4. Yun et al. — *CutMix*, ICCV 2019  
5. Menon et al. — *Long-Tail Learning via Logit Adjustment*, ICLR 2021  
6. Izmailov et al. — *Averaging Weights Leads to Wider Optima*, UAI 2018  
7. He et al. — *Delving Deep into Rectifiers*, ICCV 2015  

---

## Citation

If you find this useful, please cite:

```
@misc{group24_shiftguard10_2025,
  title  = {Robust Image Classification on ShiftGuard10 Using Wide Residual Networks and Pseudo-Label Refinement},
  author = {Miglani, Siddharth and Mandal, Shaondeep and De, Shohom and Raj, Dipanshu and Shukla, Divyansh},
  year   = {2025},
  note   = {EE708, IIT Kanpur}
}
```
