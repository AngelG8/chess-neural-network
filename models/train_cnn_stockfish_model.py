import os

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l1, l2


from typing import Dict, List

import chess
import chess.pgn
import chess.engine

# Evaluate up to 1,000,000 boards, but we only have data for about 225,000
NUM_BOARDS = 1000000

script_path: str = os.path.abspath(__file__)
data_path: str = os.path.join(os.path.dirname(script_path), 'data', 'fen_evaluations_lichess_db_standard_rated_2014-08.txt')

def board_to_input(board: chess.Board) -> np.ndarray:
    """
    Converts a chess board into a (12x8x8) tensor input layer

    Arguments:
        board (Board): A chess board
    
    Returns:
        (ndarray): The board represented as an (12x8x8) tensor
    """
    # Define piece mapping for indexing the array
    piece_mapping: Dict = {
        'p': 0, 'r': 1, 'n': 2, 'b': 3, 'q': 4, 'k': 5,
        'P': 6, 'R': 7, 'N': 8, 'B': 9, 'Q': 10, 'K': 11,
    }

    # Initialize the input array with zeros
    input_array: np.ndarray = np.zeros((12, 8, 8), dtype=np.int8)

    # Iterate over the board and fill the input array
    for row in range(8):
        for col in range(8):
            square: chess.Square = chess.square(col, 8 - 1 - row) 
            piece: (chess.Piece | None) = board.piece_at(square)
            if piece:
                piece_index = piece_mapping[piece.symbol()]
                input_array[piece_index, 8 - 1 - row, col] = 1 * int(board.turn)

    return input_array

# Load labeled data from the text file
with open(data_path, 'r') as file:
    lines = file.readlines()

# Extract FEN strings and Stockfish evaluations
fen_strings: List[str] = []
stockfish_evaluations: List[str] = []
with open(data_path, 'r') as file:
    for _ in range(NUM_BOARDS):
        fen = file.readline().strip()
        eval = file.readline().strip()

        if not fen or not eval:
            break

        fen_strings.append(fen)
        stockfish_evaluations.append(eval)

# Convert FEN strings to numpy array of board inputs
x_board = np.array([board_to_input(chess.Board(fen.strip())) for fen in fen_strings])

# Convert Stockfish evaluations to numpy array
y_evaluation = np.array([float(eval) / 100 if eval != 'None' else 0.0 for eval in stockfish_evaluations])

# Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(np.array(x_board), np.array(y_evaluation), test_size=0.2, random_state=42)

# Data needs to be in a specific format when we fit the model
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Replace None with 0
y_train = np.nan_to_num(y_train, nan=0.0)
y_test = np.nan_to_num(y_test, nan=0.0)

print("Building model...")
# Build the CNN model
# Does not use regularization, but is faster to train (Can add more layers)
# model = models.Sequential([
#     layers.Conv2D(64, (5, 5), activation='relu', input_shape=(12, 8, 8), padding='same'),
#     layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#     # layers.MaxPooling2D((2, 2)),
#     layers.BatchNormalization(),
#     layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
#     # layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     # layers.Dropout(0.5),
#     layers.Dense(128, activation='relu'),
#     # layers.Dropout(0.5),
#     layers.Dense(1)
# ])

# This model uses regularization which is a CPU task. This takes much longer to train
l1_strength: float = 0.05
l2_strength: float = 0.10

model = models.Sequential([
    layers.Conv2D(64, (5, 5), activation='relu', input_shape=(12, 8, 8), padding='same', kernel_regularizer=l1(l1_strength)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l1(l1_strength)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    # layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(l2_strength)),
    # layers.Dropout(0.5),
    layers.Dense(1)
])

print("Compiling model...")
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate the model on the validation set
loss = model.evaluate(X_test, y_test)
print(f"Validation Loss: {loss}")

save_path = os.path.join(os.path.dirname(script_path), 'save', 'model_all_l2')
model.save(save_path)

