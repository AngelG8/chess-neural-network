import chess
import random

from typing import Callable, List, Optional
from models.chess_ai import ChessAI

from utils.constants import ALPHA_BETA_DEPTH, PIECE_VALUES, PIECE_SQUARE_TABLES
from utils.evaluation_functions import get_material_score, get_piece_square_material_score

class AlphaBetaAI(ChessAI):
    """
    Represents a chess AI that makes moves based on a minimax algorithm with alpha beta pruning.
    """
    def __init__(self, evaluation_function: Callable = get_piece_square_material_score):
        self.depth: int = ALPHA_BETA_DEPTH
        self.evaluation_function: Callable = evaluation_function
        self.is_playing_white: bool = True

    def __alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool) -> float:
        """
        Performs the alpha-beta pruning algorithm to determine the best move for a player given a board.

        Arguments:
            board (chess.Board): The board to evaluate
            depth (int): What depth to search to
            alpha (float): The best value that the maximizing player currently has
            beta (float): The best value that the minimizing player currently has
            maximizing_player (bool): Is this the maximizing player (i.e. white to move)
        
        Returns:
            (float): The alpha-beta pruning evaluation for the given player
        """
        if depth == 0 or board.is_game_over():
            return self.evaluation_function(board)

        legal_moves = list(board.legal_moves)
        if maximizing_player:
            value = float('-inf')
            for move in legal_moves:
                board.push(move)
                value = max(value, self.__alpha_beta(board, depth - 1, alpha, beta, not maximizing_player))
                board.pop()
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float('inf')
            for move in legal_moves:
                board.push(move)
                value = min(value, self.__alpha_beta(board, depth - 1, alpha, beta, not maximizing_player))
                board.pop()
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value
        
    def find_move(self, board: chess.Board) -> chess.Move:
        """
        Finds the best move for a given board using alpha-beta pruning.

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
            value: float = self.__alpha_beta(board, self.depth - 1, float('-inf'), float('inf'), not self.is_playing_white)
            board.pop()
            
            if (self.is_playing_white and value > best_value) or (not self.is_playing_white and value < best_value):
                best_value = value
                best_move = move

        return best_move