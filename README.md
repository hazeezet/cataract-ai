# Deep Learning-Based Cataract Detection from Retinal Fundus Images

This repository contains pipeline, and trained models. It focuses on evaluating the physiological manifestations of cataracts through an automated, deep learning-based screening tool using retinal fundus images. 

It utilizes biostatistical practices to handle extreme pathological class imbalances. Standard "accuracy" metrics are rejected in favor of Precision and Recall mapping to avoid missing crucial positive diagnoses.

## 🎯 Versioning Strategy
To ensure a structured, iterative scientific process, every major configuration change, data augmentation update, or architectural adjustment is managed through **Tags / Versions**:

- **[v1.0] (Current): Baseline Model Integration** 
  - Validated dataset loading and stratified 70/15/15 splitting.
  - Implemented a ResNet-50 architecture via PyTorch Transfer Learning.
  - Successfully countered the massive Normal vs. Cataract discrepancy using mathematically weighted loss functions (`BCEWithLogitsLoss`).
  - Evaluated the model strictly as a high-sensitivity medical triage system (prioritizing Recall in order to drastically reduce False Negatives).

## 📊 The Dataset

The model is trained on the **Ocular Disease Intelligent Recognition (ODIR-5K)** dataset.

🔗 **Dataset Source:** [Kaggle: ODIR-5K](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)

### The Imbalance Challenge
Out of the dataset's 6392 images, only ~7% are positive for Cataracts (`C=1`), while the remaining ~93% are negative (`C=0`). A naive model would simply guess "0" every time and easily achieve 93% accuracy without learning anything. 

I strictly evaluates performance using **Precision, Recall, and F1-Scores**, relying on mathematical penalties to force the model to identify True Cataracts.

## 🧠 Current Architectural Decisions

### 1. "Targeted Screening" Framework (Cataract vs. Everything Else)
The ODIR-5K dataset contains labels for 8 different ocular diagnoses (Normal, Diabetic Retinopathy, Glaucoma, Cataract, AMD, Hypertension, Myopia, and Others). However, I purposefully framed it as a **Binary Classifier** focusing purely on cataracts to see what happens:

- **Class `1` (Positive):** The image reliably exhibits a Cataract.
- **Class `0` (Negative):** The image does **not** exhibit a Cataract. 

**Does Class 0 mean "Normal"?** No. Class `0` clusters perfectly healthy eyes *together* with eyes suffering from Glaucoma, Diabetic Retinopathy, etc. This is the correct medical standard for a **targeted triage tool** if an automated clinic sequence specifically needs to route patients for cataract surgery, the model's absolute only job is to flag cataracts. Everything else, diseased or healthy, is safely flagged as "non-cataract".

### 2. Why ResNet50 Transfer Learning instead of a Custom CNN?
When embarking on a Deep Learning computer vision project, there is often pressure to build a Convolutional Neural Network (CNN) from scratch. I chose **PyTorch Transfer Learning (ResNet50)** for several rigorous scientific reasons:

1. **Massive Data Scarcity:** We only have ~400 positive cataract images. A deep CNN built from scratch requires tens of thousands of images just to learn basic physiological geometries (edges, curves, lighting, spherical shading).
2. **Pre-trained Vision (ImageNet):** ResNet50 has already been trained on over 1 million images to extract fundamental visual features. By utilizing `ResNet50_Weights.IMAGENET1K_V1`, we inherit this pre-calculated "vision." We simply freeze those foundational rules and only train the final classification layer to recognize the specific cloudy opacity of a cataract.
3. **The Vanishing Gradient:** Building a shallow CNN from scratch vastly limits architectural complexity. ResNet's *Residual Connections* (the "Res" in ResNet) permit a 50-layer deep architecture without gradients shrinking into nothing during backpropagation—a mathematical stability that is extraordinarily difficult to achieve when assembling layers manually.
4. **Research Viability/Defense:** In a medical or academic setting, leveraging a mathematically proven architecture (ResNet) dramatically reduces architectural bias and makes the research methodology inherently more defensible.

## 🏗️ Architecture & Pipeline

The code is organized sequentially inside the `cataract.ipynb` Jupyter Notebook into five rigorously commented phases:

1. **Data Preparation & Stratification:**
   Cleans missing references (validating against the physical `preprocessed_images/` directory) and executes a stratified `Train (70%) / Val (15%) / Test (15%)` split to preserve proportional cataract distributions.

2. **Datasets & Transforms:**
   Utilizes custom PyTorch `Dataset` classes and native `torchvision.transforms`. We apply aggressive data augmentations (rotations, resized crops, flips) **exclusively** to the training flow, diversifying the minority class dynamically without contaminating the Validation/Test sets.

3. **Transfer Learning (ResNet50):**
   Downloads pre-trained ImageNet `ResNet50_Weights.IMAGENET1K_V1`. The base convolutional layers are completely frozen (`requires_grad = False`), and the head is aggressively customized with Dropout scaling into a 1D Linear layer.

4. **Training & Optimization:**
   Instead of over-sampling, the pipeline calculates a native continuous `pos_weight` ratio and feeds it natively into PyTorch's `BCEWithLogitsLoss`. The model uses the `Adam` optimizer alongside `ReduceLROnPlateau`, and aggressively monitors the `Validation Loss` to invoke custom Early-Stopping functionality (saving the `best_cataract_model.pth` parameter snapshot).

5. **Meaningful Evaluation:**
   Performs a fully detached batch pass over the Test DataLoaders utilizing `torch.sigmoid` thresholds. Generates a strict `sklearn` Classification Report mapping alongside a **Seaborn Confusion Matrix** heatmap to properly visualize False Positives vs. False Negatives.

## 🚀 Getting Started

This project is configured to run out-of-the-box on **Google Colab** to leverage free GPU compute. No local `pip install` commands are necessary.

### Google Drive Setup & Execution
To flawlessly reproduce the environment and ensure your trained models are saved permanently, follow these environment mapping steps:

1. **Prepare Google Drive:** In the root of your Google Drive, create a folder path exactly as follows: `/Colab_Contents/cataract/`.
2. **Upload Dataset:** Upload your `full_df.csv` file and the entire `preprocessed_images/` folder directly into that `cataract` folder on your Drive.
3. **Open Colab:** Upload and open `cataract.ipynb` in Google Colab.
4. **Enable GPU:** Go to `Runtime > Change runtime type` and select the **T4 GPU** hardware accelerator.
5. **Mount & Symlink:** Run the first few cells of the notebook. It will prompt you to authorize Google Drive access. The code will automatically mount your Drive and create a symbolic link (`!ln -s`) routing your Drive folder to `/content/cataract/` within the Colab environment.
6. **Train & Evaluate:** Execute the remaining cells chronologically from top to bottom. Once Phase 4 completes, your trained model (`best_cataract_model.pth`) will automatically be saved directly back into your Google Drive folder, safely preventing data loss when the Colab session eventually disconnects!
