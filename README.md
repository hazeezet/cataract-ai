# Deep Learning-Based Cataract Detection from Retinal Fundus Images

This repository contains pipeline code and trained models acting as a robust, automated triage tool for detecting cataracts in retinal fundus images. It employs advanced biostatistical algorithms (dynamic oversampling, rigorous regularizations) to combat extreme class imbalance.

## 🎯 Versioning Strategy

- **[v2.1.0] (Current): Near-Perfect Detection Pipeline** (`cataract_v3.py`)
  - **Architecture Upgrade:** Replaced ResNet50 with **EfficientNet-B3** at 300×300 native resolution (~80% more pixels). Added a deep 2-layer classification head (1536→512→1) with BatchNorm.
  - **Focal Loss:** Replaced BCE with Focal Loss (α=0.75, γ=2.0) to down-weight easy negatives and focus on hard cataract cases.
  - **Aggressive Augmentation:** 10-transform pipeline including ColorJitter, GaussianBlur, RandomErasing, RandomAffine, vertical flips, and RandomGrayscale.
  - **Mixup Regularization:** Randomly blends training image pairs and their labels to combat overfitting.
  - **Label Smoothing:** Soft labels (0.05↔0.95) for better probability calibration.
  - **Mixed Precision (AMP):** `torch.amp` for faster training with no accuracy loss.
  - **AdamW + Cosine Annealing:** Decoupled weight decay with warm restart scheduling.
  - **Enhanced TTA (5x):** Test-time ensemble of original, H-flip, V-flip, both flips, and 90° rotation.
  - **Optimal Threshold Search:** Automated sweep on validation set to maximize F1.
  - **Gradient Clipping:** `max_norm=1.0` for stable fine-tuning.

- **[v2.0] (Legacy): Explainable Fine-Tuned Triaging** (`cataract.ipynb`)
  - ResNet50 fine-tuning with `layer3`+`layer4` unfrozen. `WeightedRandomSampler`. CLAHE preprocessing. Grad-CAM explainability. Streamlit GUI prototype.

- **[v1.0] (Legacy): Baseline Model Integration** 
  - Standard ResNet-50 Transfer Learning with entirely frozen base layers.
  - Imbalanced data handled via `BCEWithLogitsLoss(pos_weight)`.
  - Simple Confusion Matrix terminal evaluation.

## 📊 The Dataset

The model is trained on a combined, augmented dataset:
1. **Ocular Disease Intelligent Recognition (ODIR-5K)**: Used for the base environment and negative/mixed-disease classes.
   🔗 **Source:** [Kaggle: ODIR-5K](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
2. **Supplementary Cataract Dataset**: Additional training and validation cataract images merged strictly to bolster the minority class before data splitting.
   🔗 **Source:** [Mendeley Data](https://data.mendeley.com/datasets/yj35kjgrv3/1)

### The Imbalance Challenge
Originally, the ODIR-5K dataset had an extreme class imbalance (only ~7% out of ~6392 images were positive for Cataracts). By injecting the combined Mendeley cataract images prior to our stratified splitting, we mathematically fortified the baseline data. Despite this boost, pathological imbalance remains a core challenge. The pipeline evaluates strictly on **Precision, Recall, and F1-Scores**, relying on native samplers to force the model into diagnosing cataracts efficiently.

## 🧠 Current Architectural Decisions

### 1. "Targeted Screening" Framework (Cataract vs. Everything Else)
The ODIR-5K dataset has 8 optical labels. This is narrowed down to a binary classifier where Class `0` contains normal eyes *and* eyes with non-cataract diseases (glaucoma, myopia, etc.). The model focuses exclusively on routing likely candidates toward cataract surgery, making it a pure triage screening tool. 

### 2. EfficientNet-B3 Transfer Learning & Fine Tuning (v2.1.0)
We initialize a pre-trained `EfficientNet_B3_Weights.IMAGENET1K_V1` network at 300×300 resolution. We freeze the first 5 feature blocks but unfreeze blocks 5–8 with differential learning rates using `AdamW`. The custom classifier uses a deep 2-layer head (1536→512→1) with BatchNorm and dual Dropout for robust regularization. Training uses **Focal Loss** (α=0.75, γ=2.0) which specifically down-weights easy-to-classify normal cases and forces the model to focus on hard cataract cases.

## 🏗️ Architecture & Pipeline (`cataract.ipynb`)

1. **Phase 1: Data Preparation & Stratification:**
   Enforces a replicable configuration via explicit random seed states (`set_seed`). Executes a stratified Train/Val/Test split to conserve proportion.
   
2. **Phase 2: Datasets, CLAHE, & Transforms:**
   Uses a PyTorch Dataset integrating OpenCV for CLAHE transformations to equalize lighting dynamics. We establish the `WeightedRandomSampler` to natively inject minority-class Cataract images evenly into batches, utilizing intensive random transformations to enforce robustness.

3. **Phase 3: Transfer Learning (ResNet50):**
   We map ResNet50, engage Gradual Unfreezing for `layer4`, and attach a 1D customized Dropout linear head.

4. **Phase 4: Training & Optimization:**
   Differential `Adam` Optimization is deployed with `weight_decay` L2 regularization to aggressively punish overfitting. Incorporates learning rate schedulers and captures historical epoch values, later generating graphical Train vs. Validation Loss curves to prove convergence.

5. **Phase 5: Meaningful Evaluation:**
   Executes inference via probability curves instead of hard limits. Maps test outputs onto `sklearn` Precision-Recall curves alongside a fully visualized Seaborn Confusion Matrix.

6. **Phase 6: Model Explainability (Grad-CAM):**
   Integrates `pytorch-grad-cam` natively tracking gradients backward from the unfrozen `layer4`. Iterates over medical predictions and overlays a generated heatmap highlighting the specific pixels the model used to confirm its Cataract diagnosis.

## 🌍 Interactive Prototype
This project comes with an encapsulated Streamlit GUI wrapper:
```bash
streamlit run app.py
```
This launches a lightweight browser interface for non-technical users to upload images, process them through the CLAHE pipeline, run an instant evaluation, and visualize the output probability.

## 🚀 Getting Started

This workspace natively runs on Google Colab or kaggle. 

### Remote Quick Start (Colab)
1. In Google Drive, create: `/Colab_Contents/cataract/` and place `full_df.csv` and `preprocessed_images/` inside.
2. Upload `cataract.ipynb` to Colab, set hardware to `T4 GPU`.
3. Mount the drive (first notebook cell) to map the local runtime to Drive.
4. Execute strictly from Top to Bottom! Results (`best_cataract_model.pth`) are automatically saved to your Drive.
