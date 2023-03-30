import numpy as np
import time
def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))
    
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])

np.random.seed(int(time.time()))

# randomly initialize our weights
W0 = np.random.random((3,4))
W1 = np.random.random((4,1))

for j in range(60000):

	# Forward proagate through layers 0, 1, and 2
    L0 = X
    L1 = sigmoid(np.dot(L0,W0))
    L2 = sigmoid(np.dot(L1,W1))

    # Backpropogate

    # Multiply how much we missed by the slope of the sigmoid 
    #  at the values in L2
    L2_error = y - L2
    
    # If the iteration is a multiple of 10000, display the progress
    if (j% 10000) == 0:
        print("Error:" + str(np.mean(np.abs(L2_error))))
        
    # Calculate the direction of the target value in respect to the error.
    # The rate of change is reduced as the error decreases.
    L2_delta = L2_error*sigmoid(L2,deriv=True)

    # Calculate how much each L1 value contributed to the L2 error using the weights (W1)
    L1_error = L2_delta.dot(W1.T)
    
    # Calculate the direction of the target value in respect to the error.
    # The rate of change is reduced as the error decreases.
    L1_delta = L1_error * sigmoid(L1,deriv=True)

    # update the weights
    W1 += L1.T.dot(L2_delta)
    W0 += L0.T.dot(L1_delta)

print(L2)
