import numpy as np

class Layer:
    def __init__(self, input_dim, output_dim, activation):
        self.input_dim = input_dim
        self.output_dim = output_dim

        if activation == 'relu':
            self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2. / input_dim)
        elif activation in ['sigmoid', 'tanh']:
            self.W = np.random.randn(output_dim, input_dim) * np.sqrt(1. / input_dim)
        else:
            self.W = np.random.randn(output_dim, input_dim) * 0.01

        self.b = np.zeros((output_dim, 1))
        self.activation_name = activation

    def forward(self, A_prev, activation_fn):
        Z = self.W @ A_prev + self.b
        A = activation_fn(Z)
        return A, Z
    
    def backward(self, dA, Z, A_prev, activation_backward_fn, m):
        dZ = activation_backward_fn(dA, Z)

        dW = (dZ @ A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = self.W.T @ dZ

        return dA_prev, dW, db