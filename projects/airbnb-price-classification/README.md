# Airbnb Price Classification

## Overview

This project builds and compares machine learning models that classify New York City Airbnb listings into **high-price** and **low-price** categories. The goal was to create an end-to-end ML workflow covering exploratory analysis, leakage-aware feature selection, preprocessing, model tuning, neural-network training, and evaluation on imbalanced data.

## Dataset

- **28,022 Airbnb listings**
- Target distribution: approximately **74% low-price / 26% high-price**
- Numerical and categorical listing attributes
- Final transformed feature space: **47 features**

To avoid target leakage, the raw `price` field used to define the label was removed from the model inputs.

## Modeling

### Logistic Regression
- Standardized numerical features and one-hot encoded categorical variables
- Used a stratified train/test split to preserve class balance
- Tuned regularization with **5-fold GridSearchCV**
- **Test Accuracy:** 82.96%
- **F1 Score:** 0.612

### Neural Network
- TensorFlow/Keras feed-forward network
- Hidden layers: **64 → 32** ReLU units
- Sigmoid output for binary classification
- **5,185 trainable parameters**
- SGD optimizer with a 0.01 learning rate
- **Test Accuracy:** 85.11%
- **F1 Score:** 0.687

The neural network improved accuracy by about **2.15 percentage points** and F1 by about **0.075** over Logistic Regression. Validation curves also showed signs of overfitting after roughly 30–40 epochs, motivating future use of early stopping and class weighting.

## Skills Demonstrated

- Exploratory data analysis
- Leakage prevention
- Missing-value imputation
- One-hot encoding and standardization
- Imbalanced classification
- Logistic Regression
- Cross-validation and hyperparameter tuning
- TensorFlow/Keras neural networks
- Accuracy and F1 evaluation
- Model comparison and business interpretation

## Source

[Review the cleaned Python source](airbnb_price_classification.py)

> The original notebook used a course-provided Airbnb dataset. The repository keeps the modeling workflow and documented results while avoiding redistribution of course assets that may not be intended for public reuse.
