# CIFAR-10 Image Recognition using CNN

## Overview
This project implements a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset**, a widely used benchmark dataset in computer vision.  
It demonstrates skills in **deep learning, image preprocessing, and model optimization**, using **TensorFlow/Keras**.  

The project is part of my preparation for research and AI-related work, and showcases my practical experience in **multimedia and AI applications**.

---

## Dataset
**CIFAR-10** contains 60,000 32x32 color images divided into 10 classes:  

| Class       | Example |
|------------|---------|
| Airplane    | ✈️      |
| Automobile  | 🚗      |
| Bird        | 🐦      |
| Cat         | 🐱      |
| Deer        | 🦌      |
| Dog         | 🐶      |
| Frog        | 🐸      |
| Horse       | 🐴      |
| Ship        | 🚢      |
| Truck       | 🚚      |

- **Training set:** 50,000 images  
- **Test set:** 10,000 images  

---

## Model Architecture

| Layer       | Type        | Parameters / Activation |
|------------|-------------|------------------------|
| 1          | Conv2D      | 32 filters, 3x3, ReLU |
| 2          | MaxPooling2D| 2x2                   |
| 3          | Conv2D      | 64 filters, 3x3, ReLU |
| 4          | MaxPooling2D| 2x2                   |
| 5          | Flatten     | —                      |
| 6          | Dense       | 128 units, ReLU       |
| 7          | Dropout     | 0.5                    |
| 8          | Dense       | 10 units, Softmax      |

- **Loss function:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy  

---

## Training

- **Epochs:** up to 20  
- **Batch size:** 64  
- **Callback:** EarlyStopping (patience = 3)  
- **Hardware:** GPU recommended (Google Colab)

```bash
python src/model_cifar10.py
