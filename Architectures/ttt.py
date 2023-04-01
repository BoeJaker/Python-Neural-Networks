import numpy as np
import time
np.random.seed(int(time.time()))
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

# Define the function to play against the network
def play_human():
    board = np.zeros(input_size)
    player = 1
    winner = 0
    while winner == 0:
        if player == 1:
            print("Your turn!")
            move = int(input("Enter a move (0-8): "))
            while board[move] != 0:
                move = int(input("Invalid move. Enter a move (0-8): "))
            board[move] = 1
        else:
            print("My turn!")
            move = select_move(board, -1)
            board[move] = -1
            print(f"I played move {move}")
        print_board(board)
        winner = check_winner(board)
        if winner != 0:
            print("Game over!")
            if winner == 1:
                print("You win!")
            else:
                print("I win!")
        else:
            player *= -1

# Define the function to play a game against adversarial moves
def play_game():
    board = np.zeros(input_size)
    player = 1
    winner = 0
    while winner == 0:
        if player == 1:
            move = int(input("Your move (0-8): "))
            while board[move] != 0:
                move = int(input("Invalid move. Your move (0-8): "))
            board[move] = 1
        else:
            move = select_move(board, -1)
            board[move] = -1
            print(f"I played move {move}")
        winner = check_winner(board)
        if winner != 0:
            if winner == 1:
                print("You win!")
                reward = 1
            else:
                print("I win!")
                reward = -1
        else:
            reward = 0
            player *= -1
        update_weights(board, reward, player)

# Define the function to update the weights
def update_weights(board, reward, player):
    input_board = np.zeros(input_size)
    for i in range(input_size):
        if board[i] == 1:
            input_board[i] = 1

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
    # return winner
    if winner != 0:
            if winner == 1:
                print("You win!")
                reward = 1
            else:
                print("I win!")
                reward = -1
    else:
        reward = 0
        player *= -1
    update_weights(board, reward, player)

for i in range(500):
    train_game()

play_human()