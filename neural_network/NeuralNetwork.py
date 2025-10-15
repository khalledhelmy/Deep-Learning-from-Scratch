from .layers import Layer
from .activations import ACTIVATIONS
from .losses import LOSSES
import numpy as np

class NeuralNetwork:
    def __init__(self, layers_dims, activations, loss='binary_cross_entropy'):
        self.layers = []
        self.loss_name = loss

        L = len(layers_dims)
        for l in range(1, L):
            self.layers.append(Layer(layers_dims[l - 1], layers_dims[l], activations[l - 1]))

        

    def forward_propagation(self, X):
        A = X
        caches = {'A0': X}

        for i, layer in enumerate(self.layers, start=1):
            act = self._get_activation(layer.activation_name)

            A, Z = layer.forward(A, act)
            
            caches[f'A{i}'] = A
            caches[f'Z{i}'] = Z

        self.caches = caches

        return A
    
    def compute_loss(self, Y_hat, Y):
        loss_fn = self._get_loss(self.loss_name)
        return loss_fn(Y, Y_hat)
        

    def backward_propagation(self, Y):
        grads = {}
        
        m = Y.shape[1]

        L = len(self.layers)
        Y_hat = self.caches[f'A{L}']
        
        loss_backward_fn = self._get_loss_backward(self.loss_name)
        dA = loss_backward_fn(Y, Y_hat)

        for l in reversed(range(L)):
            layer = self.layers[l]

            Z = self.caches[f'Z{l+1}']
            A_prev = self.caches[f'A{l}']

            act_backward = self._get_activation_backward(layer.activation_name)

            dA, dW, db = layer.backward(dA, Z, A_prev, act_backward, m)
            grads[f'dW{l+1}'] = dW
            grads[f'db{l+1}'] = db

        self.grads = grads
        return grads
    
    def update_parameters(self, learning_rate):
        for l, layer in enumerate(self.layers, start=1):
            layer.W -= learning_rate * self.grads[f'dW{l}']
            layer.b -= learning_rate * self.grads[f'db{l}']

    def train(self, X, Y, epochs=1000, lr=0.01, print_loss=True):
        for i in range(epochs):
            Y_hat = self.forward_propagation(X)
            loss = self.compute_loss(Y_hat, Y)
            
            self.backward_propagation(Y)
            self.update_parameters(lr)

            if print_loss:
                print(f'Epoch {i}: loss = {loss:0.4f}')
    
    def predict(self, X, threshold=0.5):
        Y_hat = self.forward_propagation(X)
        if self.loss_name == 'binary_cross_entropy':
            return (Y_hat > threshold).astype(int)
        return Y_hat
    
    def evaluate(self, X, Y, threshold = 0.5, print_results=True):
        Y_pred = self.predict(X)
        
        loss_fn = self._get_loss(self.loss_name)
        loss = loss_fn(Y, Y_pred)

        accuracy = None
        if self.loss_name == 'binary_cross_entropy':
            Y_pred = (Y_pred > threshold).astype(int)
            accuracy = np.mean(Y_pred == Y)

        if print_results:
            print(f"Evaluation results:")
            print(f" Loss = {loss:.4f}")
            if accuracy is not None:
                print(f" Accuracy = {accuracy*100:.2f}%")

        return loss, accuracy

    def summary(self):
        print("Neural Network Architecture:")
        for i, layer in enumerate(self.layers, start=1):
            print(f" Layer {i}: {layer.input_dim} → {layer.output_dim} ({layer.activation_name})")
        print(f" Loss Function: {self.loss_name}")

    
    def _get_activation(self, name): 
        if name not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation function: {name}")
        return ACTIVATIONS[name][0]

    def _get_activation_backward(self, name):
        if name not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation function: {name}")
        return ACTIVATIONS[name][1]
    
    def _get_loss(self, name): 
        if name not in LOSSES:
            raise ValueError(f"Unsupported loss function: {name}")
        return LOSSES[name][0]

    def _get_loss_backward(self, name):
        if name not in LOSSES:
            raise ValueError(f"Unsupported loss function: {name}")
        return LOSSES[name][1]