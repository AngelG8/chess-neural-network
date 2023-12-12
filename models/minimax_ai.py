import random
import chess

from typing import Callable, List, Optional
from models.chess_ai import ChessAI

from utils.constants import MINIMAX_DEPTH, PIECE_VALUES
from utils.evaluation_functions import get_material_score, get_piece_square_material_score

class MinimaxAI(ChessAI):
    """
    Represents a chess AI that makes moves based on a minimax algorithm.
    """
    def __init__(self, evaluation_function: Callable = get_piece_square_material_score):
        self.depth = MINIMAX_DEPTH
        self.evaluation_function = evaluation_function
        self.is_playing_white: bool = True

    def __minimax(self, board: chess.Board, depth: int, maximizing_player: bool) -> float:
        """
        Performs a minimax algorithm to determine the best move for a plater given a board.

        Arguments:
            board (chess.Board): The board to evaluate
            depth (int): What depth to search to
            maximizing_player (bool): Is this is maximizing player (i.e. white to move)
        
        Returns:
            (float): The minimax evaluation for the given player
        """
        if depth == 0 or board.is_game_over():
            return self.evaluation_function(board)

        legal_moves: List[chess.Move] = list(board.legal_moves)
        if maximizing_player:
            value = float('-inf')
            for move in legal_moves:
                board.push(move)
                value = max(value, self.__minimax(board, depth - 1, not maximizing_player))
                board.pop()
            return value
        else:
            value = float('inf')
            for move in legal_moves:
                board.push(move)
                value = min(value, self.__minimax(board, depth - 1, not maximizing_player))
                board.pop()
            return value

    def find_move(self, board: chess.Board) -> chess.Move:
        """
        Finds the best move for a given board.

        Arguments:
            board (chess.Board): A board to evaluate

        Returns:
            (chess.Move): The best move given the board
        """
        legal_moves: List[chess.Move] = list(board.legal_moves)
        random.shuffle(legal_moves)
        best_move: chess.Move = legal_moves[0]
        best_value: float = float('-inf') if self.is_playing_white else float('inf')
        
        for move in legal_moves:
            board.push(move)
            value: float = self.__minimax(board, self.depth - 1, not self.is_playing_white)
            board.pop()
            
            if (self.is_playing_white and value > best_value) or (not self.is_playing_white and value < best_value):
                best_value = value
                best_move = move

        return best_move