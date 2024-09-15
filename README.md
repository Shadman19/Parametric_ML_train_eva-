# Parameterized Machine Learning Model Training with PyTorch

This Google Colab notebook provides a flexible, parameterized workflow for loading datasets and models from `torchvision`, training machine learning models, and evaluating them with robust error handling. It also supports optional early stopping to improve training efficiency.

## Overview

The notebook implements a parameterized machine learning pipeline using PyTorch, with options to:
- Select from all available datasets in `torchvision.datasets`
- Choose from a wide range of pre-trained models in `torchvision.models`
- Customize training settings such as epochs, learning rate, and batch size
- Enable optional early stopping based on validation loss with `patience` and `min_delta` parameters

The goal of this project is to provide a flexible, user-friendly, and error-resilient workflow for machine learning training in PyTorch, ideal for experimenting with different datasets and models.

## Features

- **Dataset Selection**: 
    - Users can select from any dataset available in `torchvision.datasets`.
    - If the dataset name is incorrect, the notebook provides suggestions and handles errors gracefully.

- **Model Selection**: 
    - Users can choose from popular models in `torchvision.models` (e.g., ResNet, VGG, EfficientNet).
    - If an invalid model name is provided, the notebook suggests similar models and handles errors.

- **Training**:
    - Customizable training parameters: epochs, batch size, and learning rate.
    - Uses `tqdm` progress bars to display training progress.
    - Option to enable verbose output for detailed logs of the training process.
  
- **Early Stopping**: 
    - Optionally enable early stopping based on validation loss to prevent overfitting.
    - Parameters include `patience` (number of epochs to wait for improvement) and `min_delta` (minimum improvement threshold).

- **Evaluation**:
    - Evaluates the model on a test set and prints the final accuracy and loss.

## Setup and Dependencies

The notebook is designed to run in **Google Colab** but can be adapted to other environments with minor modifications.

### Install Dependencies
Make sure you have the following dependencies installed:
- `torch`
- `torchvision`
- `tqdm`
- `matplotlib`
- `numpy`

To install dependencies in your environment, you can run:
```bash
pip install torch torchvision tqdm matplotlib numpy
```
### Example input when running the notebook
Enter a dataset name from torchvision: CIFAR10
Enter batch size: 32
Enter a model name from torchvision: resnet18
Enter the number of epochs: 20
Enter the learning rate: 0.001
Enter patience for early stopping: 5
Enter minimum delta for improvement: 0.01
Enable verbose output? (yes/no): yes

