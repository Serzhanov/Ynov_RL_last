import numpy as np
import streamlit as st


class Layer:
    def __init__(self, input_size, output_size, activation):
        """
        Initializes a neural network layer.

        Parameters:
        - input_size (int): Number of input neurons.
        - output_size (int): Number of output neurons.
        - activation (object): Activation function object.
        """
        self.weights = np.random.uniform(
            0.0, 1.0, size=(input_size, output_size))
        self.bias = np.zeros((1, output_size))
        self.activation = activation()

    def forward(self, inputs):
        """
        Performs the forward pass through the layer.

        Parameters:
        - inputs (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Output data after applying activation function.
        """
        self.inputs = inputs
        self.linear_output = np.dot(inputs, self.weights) + self.bias
        self.output = self.activation(self.linear_output)*2
        return self.output

    def backward(self, output_error, learning_rate):
        """
        Performs the backward pass through the layer.

        Parameters:
        - output_error (numpy.ndarray): Error in the output layer.
        - learning_rate (float): Learning rate for weight and bias updates.

        Returns:
        - numpy.ndarray: Gradient of the loss with respect to the inputs.
        """
        linear_grad = np.array(
            output_error) * np.array(self.activation.derivative(self.linear_output))

        weight_grad = np.dot(self.inputs.T, linear_grad)
        bias_gradient = np.sum(linear_grad, axis=0, keepdims=True)

        # Update the weights and bias using gradient descent
        self.weights -= learning_rate * weight_grad
        self.bias -= learning_rate * bias_gradient
        # Compute the gradient of the loss with respect to the inputs
        inputs_gradient = np.dot(linear_grad, self.weights.T)
        return inputs_gradient


class SigmoidActivation:
    def __call__(self, x):
        """
        Sigmoid activation function.

        Parameters:
        - x (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Output data after applying sigmoid activation.
        """
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        """
        Derivative of the sigmoid activation function.

        Parameters:
        - x (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Derivative of the sigmoid activation.
        """
        return np.array(x) * np.array((1 - x))


class SigmoidActivation:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return np.array(x) * np.array((1 - x))


class ReLUActivation:
    def __call__(self, x):
        return np.maximum(x, 0)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)


class TanhActivation:
    def __call__(self, x):
        return 2 / (1 + np.exp(-2*x)) - 1

    def derivative(self, x):
        return 1 - np.power(self(x), 2)


class NeuralNetwork:
    def __init__(self, layers):
        """
        Initializes a neural network.

        Parameters:
        - layers (list): List of Layer objects representing the network architecture.
        """
        self.layers = layers

    def forward(self, fit_matrix):
        """
        Performs the forward pass through the entire neural network.

        Parameters:
        - fit_matrix (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Output data after forward pass.
        """
        for layer in self.layers:
            fit_matrix = layer.forward(fit_matrix)
        return fit_matrix

    def backward(self, targets, learning_rate):
        """
        Performs the backward pass through the entire neural network.

        Parameters:
        - targets (numpy.ndarray): Target values for the training data.
        - learning_rate (float): Learning rate for weight and bias updates.
        """
        output_error = targets - self.layers[-1].output
        error = output_error

        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)

    def train(self, inputs, targets, epochs, learning_rate):
        """
        Trains the neural network using backpropagation.

        Parameters:
        - inputs (numpy.ndarray): Input data for training.
        - targets (numpy.ndarray): Target values for training data.
        - epochs (int): Number of training epochs.
        - learning_rate (float): Learning rate for weight and bias updates.
        """
        progress_bar = st.progress(0)
        for epoch in range(1, epochs+1, 1):
            for i in range(len(inputs)):
                input_data = inputs[i:i+1]
                target_data = targets[i:i+1]

                prediction = self.forward(input_data)
                error = target_data - prediction
                error_mean = abs(np.mean(target_data - prediction))
                self.backward(error, learning_rate)

                progress_bar.progress(int((epoch / epochs) * 100))

            st.write(
                f"Epoch {epoch}/{epochs},  Error: {error_mean}")
        st.success('Training Complete')


def init_ANN(inputs, targets, epochs, num_hidden_layers, size, hidden_layers_size, learning_rate=0.1):
    """
    Initializes and trains an Artificial Neural Network (ANN).

    Parameters:
    - inputs (numpy.ndarray): Input data for training.
    - targets (numpy.ndarray): Target values for training data.
    - epochs (int): Number of training epochs.
    - num_hidden_layers (int): Number of hidden layers in the neural network.
    - size (int): Size of the input and output layers.
    - hidden_layers_size (int): Size of the hidden layers.
    - learning_rate (float): Learning rate for weight and bias updates.

    Returns:
    - NeuralNetwork: Trained neural network.
    """
    layers = [Layer(input_size=size, output_size=hidden_layers_size,
                    activation=TanhActivation)]
    for _ in range(num_hidden_layers):
        layers.append(Layer(input_size=hidden_layers_size, output_size=hidden_layers_size,
                      activation=TanhActivation))
    layers.append(Layer(input_size=hidden_layers_size, output_size=size,
                  activation=TanhActivation))
    nn = NeuralNetwork(layers)
    nn.train(inputs, targets, epochs=epochs, learning_rate=learning_rate)
    return nn
