import numpy as np
from scipy.special import logsumexp, softmax
import matplotlib.pyplot as plt
from backprop_data import *


class Network(object):
    
    def __init__(self, sizes):
        """
        The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        is [784, 40, 10] then it would be a three-layer network, with the
        first layer (the input layer) containing 784 neurons, the second layer 40 neurons,
        and the third layer (the output layer) 10 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution centered around 0.
        """
        self.num_layers = len(sizes) - 1
        self.sizes = sizes
        self.parameters = {}
        for l in range(1, len(sizes)):
            self.parameters['W' + str(l)] = np.random.randn(sizes[l], sizes[l-1]) * np.sqrt(2. / sizes[l-1])
            self.parameters['b' + str(l)] = np.zeros((sizes[l], 1))
        

    def relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU."""
        return (x > 0).astype(float)

    def cross_entropy_loss(self, logits, y_true):
        """Compute cross-entropy loss."""
        m = y_true.shape[0]
        log_probs = logits - logsumexp(logits, axis=0, keepdims=True)
        y_one_hot = np.eye(10)[y_true].T
        loss = -np.sum(y_one_hot * log_probs) / m
        return loss

    def cross_entropy_derivative(self, logits, y_true):
        """Derivative of the cross-entropy loss."""
        y_one_hot = np.eye(10)[y_true].T
        ZL = softmax(logits, axis=0)
        return (ZL - y_one_hot)

    def forward_propagation(self, X):
        """
        Perform forward propagation.
        Returns logits before softmax and intermediate results.
        """
        forward_outputs = []
        A = X
        for l in range(1, self.num_layers + 1):
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(W, A) + b
            A = self.relu(Z) if l < self.num_layers else Z
            forward_outputs.append({'Z': Z, 'A': A})
        return A, forward_outputs

    def backpropagation(self, ZL, Y, forward_outputs, X):
        """
        Perform backpropagation to compute gradients.
        """
        grads = {}
        m = ZL.shape[1]  # Batch size
        dZ = self.cross_entropy_derivative(ZL, Y)

        for l in range(self.num_layers, 0, -1):
            # For Layer 1, A_prev should be the input X (not the previous layer's activations)
            A_prev = forward_outputs[l - 2]['A'] if l > 1 else X  # Use input for first layer

            # Compute gradients for weights and biases
            if l == 1:  # Special handling for the first layer
                grads['dW' + str(l)] = np.dot(dZ, A_prev.T) / m  # A_prev is X for the first layer
            else:
                grads['dW' + str(l)] = np.dot(dZ, A_prev.T) / m
            
            grads['db' + str(l)] = np.sum(dZ, axis=1, keepdims=True) / m

            if l > 1:
                W = self.parameters['W' + str(l)]
                dA_prev = np.dot(W.T, dZ)
                dZ = dA_prev * self.relu_derivative(forward_outputs[l - 2]['Z'])

        return grads

    def sgd_step(self, grads, learning_rate):
        """
        Updates the network parameters via SGD with the given gradients and learning rate.
        """
        parameters = self.parameters
        L = self.num_layers
        for l in range(L):
            parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
        return parameters

    def train(self, x_train, y_train, epochs, batch_size, learning_rate, x_test, y_test):
        epoch_train_cost = []
        epoch_test_cost = []
        epoch_train_acc = []
        epoch_test_acc = []
        for epoch in range(epochs):
            costs = []
            acc = []
            for i in range(0, x_train.shape[1], batch_size):
                X_batch = x_train[:, i:i+batch_size]
                Y_batch = y_train[i:i+batch_size]
                
                ZL, caches = self.forward_propagation(X_batch)
                cost = self.cross_entropy_loss(ZL, Y_batch)
                costs.append(cost)
                grads = self.backpropagation(ZL, Y_batch, caches, X_batch)

                self.parameters = self.sgd_step(grads, learning_rate)

                preds = np.argmax(ZL, axis=0)
                train_acc = self.calculate_accuracy(preds, Y_batch, batch_size)
                acc.append(train_acc)

            average_train_cost = np.mean(costs)
            average_train_acc = np.mean(acc)
            print(f"Epoch: {epoch + 1}, Training loss: {average_train_cost:.20f}, Training accuracy: {average_train_acc:.20f}")

            epoch_train_cost.append(average_train_cost)
            epoch_train_acc.append(average_train_acc)

            # Evaluate test error
            ZL, caches = self.forward_propagation(x_test)
            test_cost = self.cross_entropy_loss(ZL, y_test)
            preds = np.argmax(ZL, axis=0)
            test_acc = self.calculate_accuracy(preds, y_test, len(y_test))
            # print(f"Epoch: {epoch + 1}, Test loss: {test_cost:.20f}, Test accuracy: {test_acc:.20f}")

            epoch_test_cost.append(test_cost)
            epoch_test_acc.append(test_acc)

        return self.parameters, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc


    def calculate_accuracy(self, y_pred, y_true, batch_size):
      """Returns the average accuracy of the prediction over the batch """
      return np.sum(y_pred == y_true) / batch_size


if __name__ == '__main__':
    

    # Loading Data
    np.random.seed(0)  # For reproducibility
    n_train = 50000
    n_test = 10000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)





    # Training configuration
    epochs = 30
    batch_size = 100
    learning_rate = 0.1

    # Network configuration
    layer_dims = [784, 40, 10]
    net = Network(layer_dims)
    net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)
