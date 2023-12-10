import os

import numpy as np
from sklearn.model_selection import train_test_split

import concurrent.futures

import tensorflow as tf
from tensorflow.keras import layers, models


from typing import Dict, List

import chess
import chess.pgn
import chess.engine

NUM_GAMES = 1000

script_path = os.path.abspath(__file__)
dataset_path = os.path.join(os.path.dirname(script_path), 'data', 'lichess_db_standard_rated_2013-01.pgn')
stockfish_path = os.path.join(os.path.dirname(script_path), 'engines', 'stockfish', '16', 'bin', 'stockfish')

pgn_file = open(dataset_path)

# Extract moves from games where both White and Black Elo are above 2000
filtered_moves: List[chess.pgn.Mainline[chess.Move]] = []
num_games: int = 0
while True:
    game = chess.pgn.read_game(pgn_file)
    
    if game is None:
        break

    if num_games >= NUM_GAMES:
        break

    # Check if both White and Black Elo are above 2000
    if (
        "WhiteElo" in game.headers
        and "BlackElo" in game.headers
        and game.headers["WhiteElo"] != "?"
        and game.headers["BlackElo"] != "?"
        and int(game.headers["WhiteElo"]) > 2000
        and int(game.headers["BlackElo"]) > 2000
    ):
        num_games += 1
        print("games loaded:", num_games)
        filtered_moves.append([move for move in game.mainline_moves()])

pgn_file.close()

def get_stockfish_evaluation(board: chess.Board, time_limit_seconds: float) -> int:
    """
    Gets the stockfish evaluation for a board as a centipawn score.
    A centipawn score measures how good a position is for a player in
    terms of pawns. A score of 50 means White is up half a pawn, while
    a score of -200 means Black is up two pawns.

    Arguments:
        board (Board): A chess board to evaluate
        time (float): a maximum time limit for stockfish to run in seconds
    
    Returns:
        (int): The stockfish evaluation for a board as a centipawn score.
    """
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        # Analyze the position
        info: chess.engine.InfoDict = engine.analyse(board=board, limit=chess.engine.Limit(time=time_limit_seconds))
        # Get the evaluation score (in centipawns)
        score: int = info["score"].relative.score()

    return score

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

games_evaluated: int = 0

def evaluate_game(game):
    global games_evaluated
    board = chess.Board()
    print(type(board.fen()))
    positions = []
    evaluations = []

    for move in game:
        board.push(move)
        positions.append(board_to_input(board))
        evaluations.append(get_stockfish_evaluation(board, time_limit_seconds=0.1))

    games_evaluated += 1
    print("games evaluated:", games_evaluated)
    return positions, evaluations

# Use concurrent.futures to parallelize stockfish evaluations
with concurrent.futures.ThreadPoolExecutor() as executor:
    all_results = list(executor.map(evaluate_game, filtered_moves))

# Unpack the results into separate lists for positions and evaluations
x_board, y_evaluation = zip(*all_results)

# Flatten the lists of positions and evaluations
x_board = [pos for game_positions in x_board for pos in game_positions]
y_evaluation = [score for game_scores in y_evaluation for score in game_scores]


# Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(np.array(x_board), np.array(y_evaluation), test_size=0.2, random_state=42)

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

y_train = np.nan_to_num(y_train, nan=0.0)
y_test = np.nan_to_num(y_test, nan=0.0)

print("Y_TRAIN")
print(y_train)

print("Building model...")
# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(12, 8, 8)),
    layers.MaxPooling2D((1, 1)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((1, 1)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output layer for evaluation
])


print("Compiling model...")
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model on the validation set
loss = model.evaluate(X_test, y_test)
print(f"Validation Loss: {loss}")

save_path = os.path.join(os.path.dirname(script_path), 'save', 'model')

# # Save the entire model to the absolute path
model.save(save_path)
