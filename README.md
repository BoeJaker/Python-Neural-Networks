# Python Neural Networks
A place for me to document my basic neural network model implementations.

The jupyter notebook in the root directory goes over each implementation, describes each feature and provides example output.

## Single Layer Perceptron
A single layer perceptron (SLP) is a type of neural network with a single layer of neurons that can be used for binary classification problems. The SLP takes the input features, applies weights to them, and passes them through an activation function to produce a single output value that represents the classification decision. The SLP uses a linear activation function that simply sums the weighted inputs, so it can only learn linearly separable decision boundaries.

## Multi Layer Perceptron
In contrast, a multilayer perceptron (MLP) is a neural network with multiple layers of neurons, including an input layer, one or more hidden layers, and an output layer. MLPs can learn more complex decision boundaries and can be used for a variety of classification and regression tasks. Each neuron in an MLP receives inputs from the neurons in the previous layer, applies weights to them, and passes them through an activation function. The output, lying in the final layer of neurons, represents the network's prediction.  The main disadvantage of MLPs is that they can be prone to overfitting if the number of neurons and layers is too large.

## Recurrant Neural Network
 Unlike feedforward and convolutional networks, RNNs have the ability to process inputs that occur in a sequence, such as natural language sentences or time-series data. This is achieved through the use of recurrent connections between neurons that allow the network to maintain a hidden state that summarizes the information from previous inputs.

In feedforward networks and convolutional networks, the inputs and outputs are independent of each other, and the network processes all inputs at once. They are generally used for tasks such as image classification and feature extraction. In contrast, RNNs are designed to process inputs that occur over time, and they are used for tasks such as speech recognition, natural language processing, and video analysis.

Another difference between RNNs and other neural network architectures is that RNNs have a memory that allows them to maintain information over time. This memory is stored in the hidden state of the network, which is updated at each step of the sequence. This allows the network to capture long-term dependencies and patterns in sequential data.

## Long Short-Term Memory Neural Network
 LSTM networks are specifically designed to handle sequential data with long-term dependencies. While traditional feedforward networks and even recurrent neural networks (RNNs) can handle sequential data, they are often limited by the vanishing gradient problem, which occurs when gradients become smaller and smaller as they are backpropagated through time,making it difficult for the network to learn long-term dependencies.

LSTM networks solve this problem by introducing a memory cell and several gating mechanisms that control the flow of information. The memory cell allows the network to store information over long periods of time, while the gating mechanisms regulate the flow of information in and out of the memory cell. The gates include the input gate, forget gate, and output gate, which are used to control the flow of information into the memory cell, erase information from the memory cell, and control the flow of information out of the memory cell, respectively.

LSTM networks have been shown to be effective in a variety of applications involving sequential data, such as natural language processing, speech recognition, and time series prediction. They are particularly useful when the relationships between inputs and outputs are complex and involve long-term dependencies, which can be difficult to capture with other types of neural networks.


