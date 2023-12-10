import chess
import random

from typing import List, Optional
from models.chess_ai import ChessAI

from utils.constants import ALPHA_BETA_DEPTH, PIECE_VALUES

class AlphaBetaAI(ChessAI):
    """
    Represents a chess AI that makes moves based on a minimax algorithm with alpha beta pruning.
    """
    def __init__(self):
        self.depth = ALPHA_BETA_DEPTH
    
    def __evaluate_board(self, board: chess.Board) -> float:
        """
        Evaluates a chess board using chess piece values.
        
        Args:
            board (chess.Board): A chess board to evaluate

        Returns:
            (float): The evaluation value of the board
        """
        evaluation: float = 0
        for square in chess.SQUARES:
            piece: Optional[chess.Piece] = board.piece_at(square)
            if piece is not None:
                if piece.color == chess.WHITE:
                    evaluation += PIECE_VALUES[piece.piece_type]
                else:
                    evaluation -= PIECE_VALUES[piece.piece_type]

        return evaluation

    def __alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool) -> float:
        """
        Performs the alpha-beta pruning algorithm to determine the best move for a player given a board.

        Args:
            board (chess.Board): The board to evaluate
            depth (int): What depth to search to
            alpha (float): The best value that the maximizing player currently has
            beta (float): The best value that the minimizing player currently has
            maximizing_player (bool): Is this the maximizing player (i.e. white to move)
        
        Returns:
            (float): The alpha-beta pruning evaluation for the given player
        """
        if depth == 0 or board.is_game_over():
            return self.__evaluate_board(board)

        legal_moves: List[chess.Move] = list(board.legal_moves)

        if maximizing_player:
            for move in legal_moves:
                board.push(move)
                eval_score = -self.__alpha_beta(board, depth - 1, -beta, -alpha, False)
                board.pop()
                alpha = max(alpha, eval_score)
                if alpha >= beta:
                    break
            return alpha
        else:
            for move in legal_moves:
                board.push(move)
                eval_score = -self.__alpha_beta(board, depth - 1, -beta, -alpha, True)
                board.pop()
                beta = min(beta, eval_score)
                if alpha >= beta:
                    break
            return beta
        
    def find_move(self, board: chess.Board) -> chess.Move:
        """
        Finds the best move for a given board using alpha-beta pruning.

        Args:
            board (chess.Board): A board to evaluate

        Returns:
            (chess.Move): The best move given the board
        """
        legal_moves: List[chess.Move] = list(board.legal_moves)
        best_moves: List[chess.Move] = [legal_moves[0]]
        # best_move: chess.Move = random.choice(legal_moves) if legal_moves else None
        alpha = float('-inf')
        beta = float('inf')

        
        for move in legal_moves:
            board.push(move)
            eval_score = -self.__alpha_beta(board, self.depth - 1, -beta, -alpha, -1)
            board.pop()

            if eval_score == alpha:
                best_moves.append(move)
            elif eval_score > alpha:
                alpha = eval_score
                best_moves = [move]
        
        return random.choice(best_moves)