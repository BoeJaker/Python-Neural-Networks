"""
In this example, the modular neural network is composed of three modules, 
each of which is a fully connected layer with a ReLU activation function. 
The forward method of the NeuralNetwork class applies each module to the inputs in sequence to generate the final output. 
The backward method performs backpropagation through the modules to update the weights and biases. 
The train method trains the network on a set of input-target pairs using stochastic gradient descent, and the predict method generates predictions for a set of input data.

To use this modular neural network, you could create an instance of the NeuralNetwork class with the appropriate input and output shapes, 
then train it on a set of input-target pairs using the train method. 
Once trained, you could use the predict method to generate predictions for new input data.
"""

import numpy as np

class Module:
    def __init__(self, num_inputs, num_outputs):
        self.weights = np.random.normal(size=(num_inputs, num_outputs))
        self.biases = np.zeros((1, num_outputs))
    
    def forward(self, inputs):
        outputs = np.dot(inputs, self.weights) + self.biases
        return outputs
    
    def backward(self, inputs, gradients):
        weight_gradients = np.dot(inputs.T, gradients)
        bias_gradients = np.sum(gradients, axis=0, keepdims=True)
        input_gradients = np.dot(gradients, self.weights.T)
        self.weights -= weight_gradients
        self.biases -= bias_gradients
        return input_gradients

class NeuralNetwork:
    def __init__(self, input_shape, output_shape):
        self.modules = [
            Module(input_shape[0], 128),
            Module(128, 64),
            Module(64, output_shape[0])
        ]
    
    def forward(self, inputs):
        outputs = inputs
        for module in self.modules:
            outputs = module.forward(outputs)
        return outputs
    
    def backward(self, inputs, outputs, target):
        error = target - outputs
        gradients = error
        for module in reversed(self.modules):
            gradients = module.backward(inputs, gradients)
            inputs = outputs
            outputs = module.forward(inputs)
    
    def train(self, inputs, targets, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            for i in range(inputs.shape[0]):
                inputs_i = inputs[i:i+1]
                targets_i = targets[i:i+1]
                outputs_i = self.forward(inputs_i)
                self.backward(inputs_i, outputs_i, targets_i)
                loss = np.mean((targets_i - outputs_i)**2)
                if (i+1) % 1000 == 0:
                    print("Epoch", epoch+1, "Step", i+1, "Loss", loss)
    
    def predict(self, inputs):
        outputs = self.forward(inputs)
        return np.argmax(outputs, axis=-1)