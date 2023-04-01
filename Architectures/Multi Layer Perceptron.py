import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(output_size)

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

def train(model, X, y, learning_rate, epochs):
    loss_history = []
    for i in range(epochs):
        # Forward pass
        y_pred = model.forward(X)

        # Compute loss
        loss = np.mean((y_pred - y)**2)
        loss_history.append(loss)

        # Backward pass
        grad_y_pred = 2 * (y_pred - y)
        grad_z2 = grad_y_pred * sigmoid(model.z2) * (1 - sigmoid(model.z2))
        grad_a1 = np.dot(grad_z2, model.W2.T)
        grad_z1 = grad_a1 * sigmoid(model.z1) * (1 - sigmoid(model.z1))

        # Update weights and biases
        model.W2 -= learning_rate * np.dot(model.a1.T, grad_z2)
        model.b2 -= learning_rate * np.sum(grad_z2, axis=0)
        model.W1 -= learning_rate * np.dot(X.T, grad_z1)
        model.b1 -= learning_rate * np.sum(grad_z1, axis=0)

    return loss_history

def test(model, X_test, y_test):
    y_pred = model.forward(X_test)
    loss = np.mean((y_pred - y_test)**2)
    accuracy = np.mean(np.round(y_pred) == y_test)
    return loss, accuracy, y_pred

X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])

nn = MLP(3,4,1)

train(nn,X,y,0.5,100)
print(test(nn,X,y))