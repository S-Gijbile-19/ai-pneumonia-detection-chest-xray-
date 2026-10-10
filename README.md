# AI-Based Pneumonia Detection Using Chest X-Ray Images

## Project Overview

This project uses deep learning to classify chest X-ray images into two categories: **Normal** and **Pneumonia**. It applies image preprocessing techniques and a pretrained ResNet-18 convolutional neural network to explore automated medical image classification.

## Objectives

* Preprocess chest X-ray images to enhance image details.
* Apply transfer learning using ResNet-18.
* Classify X-ray images into Normal and Pneumonia categories.
* Evaluate performance using standard classification metrics.

## Dataset

A subset of the NIH ChestX-ray14 dataset containing 2,862 images was used.

* Normal: 1,431 images
* Pneumonia: 1,431 images

The dataset was divided into training, validation, and testing sets:

* Training: 1,401 images
* Validation: 351 images
* Testing: 1,110 images

The dataset and patient images are not included in this repository.

## Methodology

1. Convert images to grayscale.
2. Apply gamma correction with gamma = 1.5.
3. Apply Laplacian sharpening with alpha = 1.0.
4. Resize images to 224 × 224 pixels.
5. Use a pretrained ResNet-18 model with a two-class output layer.
6. Train using the Adam optimizer and Cross-Entropy Loss.
7. Evaluate using accuracy, precision, sensitivity, specificity, F1-score, and a confusion matrix.

## Results

| Metric               | Result |
| -------------------- | -----: |
| Accuracy             | 65.86% |
| Precision            | 61.17% |
| Sensitivity (Recall) | 86.85% |
| Specificity          | 44.86% |
| F1-score             | 71.78% |

The model showed higher sensitivity than specificity, indicating that it identified many Pneumonia cases but also incorrectly classified a substantial number of Normal images.

## Technologies Used

* Python
* PyTorch
* Torchvision
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn
* Google Colab

## Repository Structure

* `pneumonia_detection.ipynb` — Jupyter notebook containing preprocessing, model training, and evaluation.
* `README.md` — Project documentation.
* `requirements.txt` — Python dependencies (to be added).

## Limitations

The model was trained for only three epochs and showed signs of possible overfitting. Further tuning and validation on larger, diverse datasets are required.

## Disclaimer

This project is intended for educational and research purposes only. It is not a medical diagnostic tool and must not be used to make clinical decisions.
