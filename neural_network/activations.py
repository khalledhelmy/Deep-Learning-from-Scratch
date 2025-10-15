import numpy as np

def linear(Z):
    return Z

def linear_backward(dA, Z):
    return dA

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def sigmoid_backward(dA, Z):
    A = sigmoid(Z)
    return dA * A * (1- A)


def relu(Z):
    return np.maximum(0, Z)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def tanh(Z):
    return np.tanh(Z)

def tanh_backward(dA, Z):
    A = np.tanh(Z)
    return  dA * (1 - A ** 2)


ACTIVATIONS = {
    'linear': (linear, linear_backward),
    'sigmoid': (sigmoid, sigmoid_backward),
    'relu': (relu, relu_backward),
    'tanh': (tanh, tanh_backward)
}