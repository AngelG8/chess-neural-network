import os

import chess
import numpy as np
import tensorflow as tf

from typing import Dict

from models.chess_ai import ChessAI

script_path: str = os.path.abspath(__file__)
model_path: str = os.path.join(os.path.dirname(script_path), 'save', 'model_all_l2')

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

class CNNAI(ChessAI):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.models.load_model(model_path)

    def evaluate_board(self, board: chess.Board) -> float:
        """
        Evaluates a board using a neural network model.

        Arguments:
            board (Board): A board to evaluate
        
        Returns:
            (float): An evaluation score
        """
        # Convert the chess board to the input format expected by the model
        input_array = board_to_input(board)
        # automatically select winning moves
        if board.is_checkmate():
            return float('inf') if board.turn == chess.WHITE else float('-inf')
        # Make a prediction using the loaded model
        evaluation = self.model.predict(np.expand_dims(input_array, axis=0))[0][0]
        return evaluation

    def find_move(self, board: chess.Board) -> chess.Move:
        """
        Finds the best move for a given board using a neural network model.

        Arguments:
            board (chess.Board): A board to evaluate

        Returns:
            (chess.Move): The best move given the board
        """
        legal_moves = list(board.legal_moves)
        best_move = None
        turn: chess.Color = board.turn
        best_evaluation = -float('inf') if turn == chess.WHITE else float('inf')

        for move in legal_moves:
            board.push(move)
            evaluation = self.evaluate_board(board)
            board.pop()

            if turn == chess.WHITE and evaluation > best_evaluation:
                best_evaluation = evaluation
                best_move = move
            elif turn == chess.BLACK and evaluation < best_evaluation:
                best_evaluation = evaluation
                best_move = move

        return best_move
