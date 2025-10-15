import numpy as np

EPS = 1e-8

def binary_cross_entropy(Y, Y_hat):
    m = Y.shape[1]

    Y_hat_clipped = np.clip(Y_hat, EPS, 1 - EPS)
    loss = - np.sum(Y * np.log(Y_hat_clipped) + (1 - Y) * np.log(1 - Y_hat_clipped)) / m

    return float(loss)

def binary_cross_entropy_backward(Y, Y_hat):
    m = Y.shape[1]

    Y_hat_clipped = np.clip(Y_hat, EPS, 1 - EPS)
    dA = - (Y / Y_hat_clipped) + ((1 - Y) / (1 - Y_hat_clipped))
    dA /= m
    return dA

def mse(Y, Y_hat):
    m = Y.shape[1]
    return np.sum((Y_hat - Y) ** 2) / (2 * m)

def mse_backward(Y, Y_hat):
    m = Y.shape[1]

    return (Y_hat - Y) / m



LOSSES = {
    'binary_cross_entropy': (binary_cross_entropy, binary_cross_entropy_backward),
    'mse': (mse, mse_backward)
}