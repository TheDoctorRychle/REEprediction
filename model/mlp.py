import numpy as np
from model.layers import relu, relu_derivative

class MLP:
    def __init__(self, input_size, hidden_layers, output_size):
        # hidden_layers
        self.layers = []
        self.biases = []
        self.activations = []

        # generating random stuff for nodes
        prev_size = input_size
        for h in hidden_layers:
            self.layers.append(np.random.randn(prev_size, h))
            self.biases.append(np.zeros((1, h)))
            self.activations.append(relu)
            prev_size = h

        # input layer
        self.layers.append(np.random.randn(prev_size, output_size))
        self.biases.append(np.zeros((1, output_size)))
        self.activations.append(lambda x: x) # linear lambda for forward propagation matrix conversions
    def forward_propagation(self, X):
        self.z = []
        self.a = [X]
        for W, b, act in zip(self.layers, self.biases, self.activations):
            # multiplication of matrices
            z = self.a[-1] @ W + b
            self.z.append(z)
            self.a.append(act(z))
        return self.a[-1]
    def backward_propagation(self, y_true):
        m = y_true.shape[0]
        grads_W = [np.zeros_like(W) for W in self.layers]
        grads_b = [np.zeros_like(b) for b in self.biases]

        # gradient
        delta = 2*(self.a[-1] - y_true)/m
        for i in reversed(range(len(self.layers))):
            grads_W[i] = self.a[i].T @ delta
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)
            if i != 0:
                delta = (delta @ self.layers[i].T) * relu_derivative(self.z[i-1])
        return grads_W, grads_b

    def update(self, grads_W, grads_b, lr):
        for i in range(len(self.layers)):
            self.layers[i] -= lr * grads_W[i]
            self.biases[i] -= lr * grads_b[i]