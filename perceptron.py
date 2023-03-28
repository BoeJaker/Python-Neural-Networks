import numpy as np

imput_layer = 3
output_layer = 1

# Input matrix, 4 entires each with 3 inputs
X = np.array([  [1,0,0],
                [0,0,1],
                [0,1,0],
                [1,0,1] ])
    
# Output set, 1 output per input entry            
y = np.array([  [0],
                [1],
                [0],
                [1]])

# Sigmoid activation function
def sigmoid(x,deriv=False):
    if(deriv==True): 
        return x*(1-x)
    else: 
        return 1/(1+np.exp(-x))

np.random.seed(1)

# Initialize the array of weights randomly
W0 = np.random.random((imput_layer,output_layer))

# Define our forward propogation function
def forward_propagate(X):
    L0 = X
    L1 = sigmoid(np.dot(L0,W0))
    return L1

def train():
    global W0
    for iter in range(10000):

        # Forward propagation
        L0 = X
        L1 = sigmoid(np.dot(L0,W0))

        # Calculate the difference between the predicted output (L1) 
        #  and target output (y)
        L1_error = y - L1

        # Multiply how much we missed by the slope of the sigmoid 
        #  at the values in L1
        L1_delta = L1_error * sigmoid(L1,True)

        # Update weights using the delta
        W0 += np.dot(L0.T,L1_delta)

    print("Output After Training:")
    print(L1)

train()    

result = np.round(forward_propagate(np.array([[1,0,1]])))

print(result)
