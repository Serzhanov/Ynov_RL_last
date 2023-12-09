import numpy as np
import streamlit as st


class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.uniform(
            0.0, 1.0, size=(input_size, output_size))
        self.bias = np.zeros((1, output_size))
        self.activation = activation()

    def forward(self, inputs):
        self.inputs = inputs
        self.linear_output = np.dot(inputs, self.weights) + self.bias
        self.output = self.activation(self.linear_output)*2
        return self.output

    def backward(self, output_error, learning_rate):

        linear_grad = np.array(
            output_error) * np.array(self.activation.derivative(self.linear_output))

        weight_grad = np.dot(self.inputs.T, linear_grad)
        bias_gradient = np.sum(linear_grad, axis=0, keepdims=True)

        # Update the weights and bias using gradient descent
        self.weights += learning_rate * weight_grad
        self.bias += learning_rate * bias_gradient
        # Compute the gradient of the loss with respect to the inputs
        inputs_gradient = np.dot(linear_grad, self.weights.T)
        return inputs_gradient


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
        self.layers = layers

    def forward(self, fit_matrix):
        for layer in self.layers:
            fit_matrix = layer.forward(fit_matrix)
        return fit_matrix

    def backward(self, targets, learning_rate):
        output_error = targets - self.layers[-1].output
        error = output_error

        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)

    def train(self, inputs, targets, epochs, learning_rate):
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
