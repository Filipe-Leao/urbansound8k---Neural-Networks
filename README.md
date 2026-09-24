# Urban Sound Classification with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow\&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-5B5EA6)](https://librosa.org/)

## Overview

This project investigates the use of Deep Learning techniques for automatic classification of urban sounds using the **UrbanSound8K** dataset.

The work explores two complementary approaches to audio classification:

* **Multilayer Perceptron (MLP)** using engineered acoustic features.
* **Convolutional Neural Network (CNN)** using Log-Mel Spectrogram representations.

The complete pipeline includes audio preprocessing, feature extraction, feature selection, model development, hyperparameter optimization, Nested Cross-Validation, model evaluation, and adversarial robustness analysis.

```text
                         UrbanSound8K
                              |
                              v
                    Audio Preprocessing
                              |
                 +------------+------------+
                 |                         |
                 v                         v
          Feature Engineering       Spectrogram Generation
                 |                         |
                 v                         v
                MLP                       CNN
                 |                         |
                 +------------+------------+
                              |
                              v
                       Model Evaluation
                              |
                              v
                   Nested Cross-Validation
                              |
                              v
                  Adversarial Robustness
```

---

## Objectives

The main objectives of the project are:

1. Perform exploratory analysis of the UrbanSound8K dataset.
2. Standardize the audio data into a consistent format.
3. Extract acoustic features for MLP-based classification.
4. Generate Log-Mel Spectrogram representations for CNN-based classification.
5. Investigate different Deep Learning architectures.
6. Perform feature selection on the engineered feature representation.
7. Optimize MLP hyperparameters using Nested Cross-Validation.
8. Evaluate model performance using Accuracy and Macro F1-score.
9. Investigate model robustness against adversarial perturbations.

---

## Dataset

The project uses the **UrbanSound8K** dataset, containing **8,732 audio clips** distributed across **10 predefined folds** and **10 urban sound classes**.

The metadata includes:

| Feature           | Description                    |
| ----------------- | ------------------------------ |
| `slice_file_name` | Audio clip filename            |
| `fsID`            | Freesound recording identifier |
| `start`           | Start time of the clip         |
| `end`             | End time of the clip           |
| `salience`        | Foreground/background salience |
| `fold`            | Dataset fold                   |
| `classID`         | Numerical class identifier     |
| `class`           | Sound class                    |

The metadata contains 8,732 observations and 8 columns, with no missing values detected in the analyzed metadata.

---

## Methodology

### 1. Audio Preprocessing

All audio files are converted into a consistent representation before feature extraction.

The preprocessing pipeline consists of:

```text
Original Audio
      |
      v
Convert to Mono
      |
      v
Resample to 22,050 Hz
      |
      v
Pad / Crop to 4 seconds
      |
      v
Normalized Waveform
```

The main parameters are:

| Parameter         |     Value |
| ----------------- | --------: |
| Sampling Rate     | 22,050 Hz |
| Duration          | 4 seconds |
| FFT Size          |     2,048 |
| Hop Length        |       512 |
| MFCC Coefficients |        40 |
| Random State      |        42 |

---

## 2. MLP Feature Engineering

The MLP operates on fixed-size one-dimensional feature vectors extracted from each audio clip.

The feature extraction pipeline includes:

### MFCC Features

* MFCC mean
* MFCC standard deviation
* MFCC delta mean
* MFCC delta standard deviation

### Spectral Features

* Spectral Centroid
* Spectral Bandwidth
* Spectral Rolloff
* Spectral Contrast

### Temporal and Energy Features

* RMS Energy
* Zero Crossing Rate

### Harmonic Features

* Chroma

The resulting feature matrix is exported as:

```text
mlp_features.csv
```

---

## 3. Feature Selection

Feature selection is applied to reduce the dimensionality of the MLP input representation.

The final selected representation contains:

| Feature Selection | Number |
| ----------------- | -----: |
| Selected features |    141 |
| Removed features  |     48 |

The selected features are subsequently used by the MLP training pipeline.

---

## 4. CNN Feature Representation

For the CNN pipeline, each waveform is transformed into a time-frequency representation using a **Log-Mel Spectrogram**.

The configuration is:

| Parameter     |     Value |
| ------------- | --------: |
| Sampling Rate | 22,050 Hz |
| FFT Size      |     2,048 |
| Hop Length    |       512 |
| Mel Bands     |       128 |
| Time Frames   |       169 |

The representation has the following dimensions:

```text
3 × 128 × 169
```

The three channels correspond to:

```text
Channel 1: Log-Mel Spectrogram
Channel 2: Delta
Channel 3: Delta-Delta
```

This representation allows the CNN to learn temporal and spectral patterns directly from the audio.

The pre-computed representations are stored as compressed `.npz` files in:

```text
precomputed_mels/
```

---

## 5. Model Architectures

### Multilayer Perceptron

The MLP pipeline can be summarized as:

```text
Audio
  |
  v
Acoustic Feature Extraction
  |
  v
Feature Selection
  |
  v
Feature Standardization
  |
  v
MLP
  |
  v
10 Output Classes
```

### Convolutional Neural Network

The CNN pipeline is:

```text
Audio
  |
  v
Log-Mel + Delta + Delta-Delta
  |
  v
3 × 128 × 169
  |
  v
CNN
  |
  v
10 Output Classes
```

Several CNN configurations were investigated by varying the network architecture and training parameters.

---

## 6. Nested Cross-Validation

Nested Cross-Validation is used for MLP hyperparameter optimization and performance evaluation.

The outer evaluation uses:

```text
10 Outer Folds
```

The hyperparameter search considers the following configurations.

### Learning Rate

```text
0.001
0.0005
0.0001
```

### Batch Size

```text
64
128
256
```

### Dropout

```text
0.1
0.2
0.3
```

The nested structure separates hyperparameter selection from final model evaluation.

---

# Results

## MLP Nested Cross-Validation

The final Nested Cross-Validation results for the MLP were:

| Metric   | Mean ± Standard Deviation |
| -------- | ------------------------: |
| Accuracy |        **86.06% ± 1.86%** |
| Macro F1 |        **86.57% ± 1.84%** |

### Results by Fold

| Fold | Accuracy | Macro F1 |
| ---: | -------: | -------: |
|    1 |   87.87% |   88.37% |
|    2 |   86.04% |   86.61% |
|    3 |   87.29% |   87.91% |
|    4 |   84.99% |   85.73% |
|    5 |   84.99% |   85.00% |
|    6 |   89.23% |   89.47% |
|    7 |   87.86% |   88.59% |
|    8 |   83.16% |   83.57% |
|    9 |   85.45% |   85.76% |
|   10 |   83.73% |   84.66% |

The highest individual outer-fold accuracy was **89.23%**, obtained on Fold 6.

The best model identified during the Nested Cross-Validation procedure used:

| Hyperparameter |  Value |
| -------------- | -----: |
| Learning Rate  | 0.0005 |
| Batch Size     |    128 |
| Dropout        |    0.1 |

The corresponding model achieved an accuracy of **89.23%**.

---

## Adversarial Robustness

The project also evaluates the robustness of the trained MLP against adversarial perturbations using **Foolbox**.

The reported experiment produced:

| Metric            |       Result |
| ----------------- | -----------: |
| Robust Accuracy   |   **13.00%** |
| Mean Perturbation | **0.050000** |

The experiment demonstrates a substantial reduction in classification performance under the adversarial perturbation evaluated in the project.

---

# Technologies

| Technology         | Purpose                                 |
| ------------------ | --------------------------------------- |
| Python             | Main programming language               |
| NumPy              | Numerical computation                   |
| Pandas             | Data manipulation                       |
| Librosa            | Audio processing and feature extraction |
| Soundata           | Dataset management                      |
| Scikit-Learn       | Machine Learning and evaluation         |
| PyTorch            | MLP implementation                      |
| TensorFlow / Keras | CNN experimentation                     |
| Foolbox            | Adversarial robustness                  |
| Matplotlib         | Data visualization                      |
| Seaborn            | Data visualization                      |
| Joblib             | Data and model utilities                |
| tqdm               | Progress monitoring                     |

---

# Project Structure

The repository can be organized as follows:

```text
urban-sound-classification/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── Trabalho_Final.ipynb
│
├── data/
│   └── README.md
│
├── features/
│   └── mlp_features.csv
│
├── models/
│   └── best_nestedcv_model.pt
│
├── precomputed_mels/
│   └── *.npz
│
└── results/
    └── cnn2d_variants_summary.csv
```

The UrbanSound8K dataset should not be committed directly to the repository.

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/<USERNAME>/<REPOSITORY>.git
cd <REPOSITORY>
```

## Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Launch Jupyter

```bash
jupyter notebook
```

Then open:

```text
notebooks/Trabalho_Final.ipynb
```

---

# Reproducing the Experiments

The notebook is organized into the following stages:

```text
1. Exploratory Data Analysis
2. Audio Preprocessing
3. MLP Feature Extraction
4. CNN Feature Extraction
5. Feature Selection
6. Model Training
7. CNN Experiments
8. Nested Cross-Validation
9. Model Evaluation
10. Adversarial Robustness
```

The UrbanSound8K dataset must be available in the environment before running the audio-processing sections.

---

# Generated Files

### MLP Features

```text
mlp_features.csv
```

Contains the engineered acoustic features used by the MLP pipeline.

### CNN Representations

```text
precomputed_mels/
```

Contains the pre-computed Log-Mel, Delta and Delta-Delta representations stored in compressed `.npz` files.

### Best Model

```text
best_nestedcv_model.pt
```

Contains the best-performing MLP model identified during the Nested Cross-Validation procedure.

---

# Summary

This project presents an end-to-end Deep Learning pipeline for urban sound classification.

The work covers:

* Audio preprocessing
* Exploratory data analysis
* Acoustic feature engineering
* Feature selection
* MLP classification
* CNN classification
* Log-Mel Spectrogram processing
* Hyperparameter optimization
* Nested Cross-Validation
* Model evaluation
* Adversarial robustness analysis

The MLP achieved a mean accuracy of **86.06%** and a mean Macro F1-score of **86.57%** across the 10 outer folds of the Nested Cross-Validation procedure.

---
**Course:** Machine Learning II

**Project:** Urban Sound Classification with Deep Learning

**Dataset:** UrbanSound8K

**Main Topics:**

```text
Machine Learning
Deep Learning
Audio Classification
Feature Engineering
Multilayer Perceptrons
Convolutional Neural Networks
Feature Selection
Cross-Validation
Hyperparameter Optimization
Adversarial Machine Learning
```

---

# Authors

**Miguel Lopes**    **Felipe Neto**    **Leonor Arreiol**


# Machine Learning II — Deep Learning Project
