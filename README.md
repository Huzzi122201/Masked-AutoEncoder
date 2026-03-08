<div align="center">

# 🎭 Masked Autoencoder (MAE)
### Self-Supervised Image Representation Learning on Tiny ImageNet

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Kaggle](https://img.shields.io/badge/Kaggle-T4%20×2-20beff?style=flat-square&logo=kaggle&logoColor=white)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

<br>

> *Reconstructing images from only 25% of visible patches — no labels, no supervision, pure visual learning.*

<br>


</div>

---

## 📌 Overview

This project implements a **Masked Autoencoder (MAE)** from scratch in pure PyTorch, based on the paper [*"Masked Autoencoders Are Scalable Vision Learners"*](https://arxiv.org/abs/2111.06377) by He et al. (CVPR 2022).

The model learns visual representations by masking **75% of image patches** and reconstructing the missing content — without any class labels. An asymmetric ViT encoder-decoder architecture makes this both powerful and computationally efficient.

**Built for:** FAST NUCES · Generative AI (AI4009) · Spring 2026

---

## ✨ Key Features

- 🔬 **Full MAE implementation** in base PyTorch — no pretrained backbones
- ⚡ **Asymmetric ViT architecture** — large encoder, lightweight decoder
- 🎭 **75% masking ratio** — forces genuine scene understanding
- 🚀 **Dual GPU training** via `nn.DataParallel` on Kaggle T4×2
- 🧪 **Mixed precision** training with `torch.amp`
- 📊 **PSNR & SSIM** quantitative evaluation
- 🖥️ **Streamlit app** — interactive reconstruction with adjustable masking ratio

---

## 🏗️ Architecture

```
INPUT IMAGE  [B, 3, 224, 224]
      │
      ▼
 PATCHIFY     →  196 patches of 16×16px
      │
      ▼
RANDOM MASK   →  keep 49 visible (25%)  │  hide 147 masked (75%)
      │
      ▼
  ENCODER     ViT-Base  │  768-dim  │  12 layers  │  12 heads  │  ~86M params
  (visible patches only — mask tokens never enter encoder)
      │
      ▼
  PROJECT     768-dim  →  384-dim
      │
      ▼
  DECODER     ViT-Small  │  384-dim  │  12 layers  │  6 heads  │  ~22M params
  (49 encoder tokens + 147 learnable mask tokens → 196 full patches)
      │
      ▼
PIXEL HEAD    384-dim  →  768 pixel values per patch
      │
      ▼
   LOSS        MSE on masked patches only
```

### Component Summary

| Component | Model | Dim | Layers | Heads | Params |
|-----------|-------|-----|--------|-------|--------|
| Encoder | ViT-Base B/16 | 768 | 12 | 12 | ~86M |
| Decoder | ViT-Small S/16 | 384 | 12 | 6 | ~22M |
| **Total** | | | | | **~108M** |

---

## 📁 Project Structure

```
mae-tiny-imagenet/
│
├── 📓 notebook.ipynb          # Full training notebook (Kaggle)
├── 🖥️  app.py                  # Streamlit interactive app
├── 🤖 best_mae_model.pth      # Trained model checkpoint
├── 📋 requirements.txt        # Python dependencies
└── 📖 README.md               # This file
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mae-tiny-imagenet.git
cd mae-tiny-imagenet
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

Place your trained `best_mae_model.pth` in the root directory, then:

```bash
streamlit run app.py
```

Open `http://localhost:8501` — upload any image, adjust the masking ratio, and watch the model reconstruct it in real time.

---

## 🧠 Model Details

### Patchification

A 224×224 image is split into a 14×14 grid of non-overlapping 16×16 patches — **196 patches total**. Each patch flattens to 768 pixel values.

```python
# 224×224 image → 196 patches of shape [16×16×3 = 768]
num_patches = (224 // 16) ** 2  # = 196
```



## ⚙️ Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Image Size | 224 × 224 |
| Patch Size | 16 × 16 |
| Mask Ratio | 75% |
| Epochs | 20 |
| Batch Size | 64 |
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 0.05 |
| LR Scheduler | CosineAnnealingLR |
| Gradient Clip | 1.0 |
| Precision | Mixed (FP16/FP32) |
| Hardware | Kaggle T4 × 2 |



## 📊 Evaluation

Reconstruction quality is measured with two complementary metrics:

| Metric | Description | Good Range |
|--------|-------------|------------|
| **PSNR** | Peak Signal-to-Noise Ratio (dB) — pixel accuracy | > 25 dB |
| **SSIM** | Structural Similarity Index — perceptual quality | > 0.70 |

Evaluation runs on 5 randomly sampled images from the Tiny ImageNet **test** folder (unseen during training):

```python
# Random test images — different every run
random.seed()
chosen = random.sample(all_test_images, 5)
```

---

## 🖥️ Streamlit App

The interactive app lets you experiment with the trained model in real time.

**Features:**
- 📤 Upload any image (JPG, PNG, WEBP)
- 🎚️ Adjustable masking ratio slider (10% → 95%)
- 📊 Live PSNR, SSIM, and MSE scores
- 🗺️ 196-patch visual map (green = visible, red = masked)
- ⬇️ Download the reconstructed image

**Run locally:**
```bash
# Place best_mae_model.pth in the same folder as app.py
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit>=1.32.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.24.0
scikit-image>=0.21.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 📚 Dataset

**Tiny ImageNet** — a scaled-down version of ImageNet-1K.

| Split | Images | Classes |
|-------|--------|---------|
| Train | 100,000 | 200 |
| Val | 10,000 | 200 |
| Test | 10,000 | — |

Available on Kaggle: [`akash2sharma/tiny-imagenet`](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)

> Note: MAE is fully self-supervised — **class labels are never used** during training. The model learns purely from pixel reconstruction.

---

## 📖 References

- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). **Masked Autoencoders Are Scalable Vision Learners.** *CVPR 2022.* [[arXiv]](https://arxiv.org/abs/2111.06377)
- Dosovitskiy, A. et al. (2021). **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.** *ICLR 2021.* [[arXiv]](https://arxiv.org/abs/2010.11929)
- Devlin, J. et al. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.** *NAACL 2019.* [[arXiv]](https://arxiv.org/abs/1810.04805)

---

## 👥 Authors

**FAST NUCES — Generative AI (AI4009) · Spring 2026**


<div align="center">

*The journey from a blank Jupyter notebook to a working Masked Autoencoder has been both humbling and exhilarating. Embrace the struggle, debug with patience, and celebrate each small victory. You're not just training a model — you're learning to think like an AI engineer.*

<br>

⭐ **Star this repo if you found it helpful!**

</div>
