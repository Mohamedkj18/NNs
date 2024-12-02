# Neural Network with Backpropagation

This repository contains the implementation of a neural network with backpropagation for training on the MNIST dataset. The network uses ReLU activation functions and cross-entropy loss for multi-class classification. The backpropagation algorithm is used for updating the network's parameters during training.

## Description

In this project, we implement a feedforward neural network from scratch with the following features:

- **Backpropagation**: The gradient of the loss function is calculated using the backpropagation algorithm and used to update the model's weights.
- **ReLU Activation**: The ReLU activation function is used for the hidden layer.
- **Softmax Output**: The softmax activation function is used for the output layer, with cross-entropy loss for multi-class classification.
- **MNIST Dataset**: The network is trained on the MNIST dataset, which contains 60,000 28x28 grayscale images of handwritten digits and their labels.

## Project Structure
├── backprop_data.py        # Code for loading and preparing the MNIST dataset
├── backprop_network.py     # Implementation of the neural network and backpropagation
├── backprop_main.py        # Code for running the network, training, and testing
├── mnist.pkl.gz            # MNIST dataset in gzipped pickle format
├── README.md               # Project description and instructions

### **File Descriptions**:

1. **`backprop_data.py`**:
   - **Purpose**: This file contains code for loading and preparing the MNIST dataset. It handles tasks like:
     - Loading the dataset from the `mnist.pkl.gz` file.
     - Preprocessing the data (e.g., normalizing, reshaping).
     - Splitting the data into training and testing sets.
   - **Key Functions**: Functions for loading and converting the MNIST dataset into a usable format for training.

2. **`backprop_network.py`**:
   - **Purpose**: This file implements the neural network and the backpropagation algorithm. It includes:
     - The **network architecture**: defining layers, activation functions, and other components.
     - The **forward pass**: computing the output of the network for a given input.
     - The **backpropagation algorithm**: calculating the gradients of the loss function with respect to the weights and updating the model parameters.
     - The **training loop**: iterating over the data and applying backpropagation to update the weights.
   - **Key Functions**:
     - `forward_propagation()`: Computes the output for a given input by passing it through the network.
     - `backpropagation()`: Computes the gradients and updates the parameters using the backpropagation algorithm.
     - `train()`: Trains the model by calling the forward and backward pass, applying the updates to the parameters.

3. **`backprop_main.py`**:
   - **Purpose**: This is the script that runs the training and testing process. It ties everything together by:
     - Setting the hyperparameters (e.g., number of epochs, learning rate, batch size).
     - Calling the `train()` function from `backprop_network.py` to train the model.
     - Optionally evaluating the model on a test dataset and printing the results (accuracy, loss).
   - **Key Functions**: This file orchestrates the training process, including setting up configurations and visualizing the results (such as plotting graphs of accuracy and loss).

4. **`mnist.pkl.gz`**:
   - **Purpose**: This file contains the **MNIST dataset** in gzipped pickle format. It includes:
     - 60,000 training images with corresponding labels.
     - 10,000 testing images with corresponding labels.
   - **Note**: The code will attempt to load the dataset from this file. If it's missing, it needs to be manually downloaded from the MNIST repository (http://yann.lecun.com/exdb/mnist/).

