import numpy as np

# Define the neural network architecture
input_size = 9
hidden_size = 18
output_size = 9

# Initialize the weights
w1 = np.random.randn(input_size, hidden_size)
w2 = np.random.randn(hidden_size, output_size)

# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# Define the forward pass function
def forward(x):
    h = sigmoid(np.dot(x, w1))
    y = softmax(np.dot(h, w2))
    return y

# Define the function to select the next move
def select_move(board, player):
    valid_moves = []
    input_board = np.zeros(input_size)
    for i in range(input_size):
        if board[i] == 0:
            valid_moves.append(i)
            if player == 1:
                input_board[i] = 1
            else:
                input_board[i] = -1
    y = forward(input_board)
    move = np.argmax(y)
    while move not in valid_moves:
        y[move] = 0
        move = np.argmax(y)
    return move

def play_game():
    board = np.zeros(input_size)
    player = 1
    winner = 0
    print_board(board)
    while winner == 0 or winner == None:
        if player == 1:
            move = int(input("Enter your move (0-8): "))
            while board[move] != 0:
                print("Invalid move. Please try again.")
                move = int(input("Enter your move (0-8): "))
            board[move] = 1
        else:
            move = select_move(board, player)
            board[move] = -1
            print("The neural network chooses square", move)
        print_board(board)
        winner = check_winner(board)
        # print(winner)7
        player = -player
    if winner == 1:
        print("You win!")
    elif winner == -1:
        print("The neural network wins!")
    else:
        print("It's a draw.")

def print_board(board):
    for i in range(3):
        print(" ".join(["X" if board[i*3+j] == 1 else "O" if board[i*3+j] == -1 else "-" for j in range(3)]))
    print("")


# Define the function to check if there's a winner
def check_winner(board):
     # Check rows
    for i in range(3):
        if board[i*3:(i+1)*3].sum() == 3:
            return 1
        elif board[i*3:(i+1)*3].sum() == -3:
            return -1
    # Check columns
    for i in range(3):
        if board[i::3].sum() == 3:
            return 1
        elif board[i::3].sum() == -3:
            return -1
    # Check diagonals
    if board[::4].sum() == 3 or board[2:7:2].sum() == 3:
        return 1
    elif board[::4].sum() == -3 or board[2:7:2].sum() == -3:
        return -1
    # Check for draw
    if (board == 0).sum() == 0:
        return 2
    # No winner yet
    return 0

def train_game():
    board = np.zeros(input_size)
    player = 1
    winner = 0
    while winner == 0:
        move = select_move(board, player)
        if player == 1:
            board[move] = 1
            player = -1
        else:
            board[move] = -1
            player = 1
        winner = check_winner(board)
    return winner

def train_network_numpy(X_train, y_train, weights, learning_rate, epochs):
    for epoch in range(epochs):
        # Forward pass
        # hidden_layer = np.dot(X_train, weights['W1'])
        # hidden_layer_activation = sigmoid(hidden_layer)
        # output_layer = np.dot(hidden_layer_activation, weights['W2'])
        # output_layer_activation = sigmoid(output_layer)
        train_game()
        # Backward pass
        error = y_train - output_layer_activation
        slope_output_layer = derivative_sigmoid(output_layer_activation)
        slope_hidden_layer = derivative_sigmoid(hidden_layer_activation)
        delta_output = error * slope_output_layer
        error_hidden_layer = delta_output.dot(weights['W2'].T)
        delta_hidden_layer = error_hidden_layer * slope_hidden_layer

        # Update weights
        weights['W2'] += hidden_layer_activation.T.dot(delta_output) * learning_rate
        weights['W1'] += X_train.T.dot(delta_hidden_layer) * learning_rate

        # Print progress
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch} Error: {np.mean(np.abs(error))}")

    return weights["W1"], weights["W2"]

w1, w2 = train_network_numpy(X_train, y_train, weights, learning_rate, epochs):
while True:
    play_game()
