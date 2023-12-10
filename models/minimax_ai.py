import chess

from typing import List, Optional
from models.chess_ai import ChessAI

from utils.constants import MINIMAX_DEPTH, PIECE_VALUES

class MinimaxAI(ChessAI):
    """
    Represents a chess AI that makes moves based on a minimax algorithm.
    """
    def __init__(self):
        self.depth = MINIMAX_DEPTH

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

    def __minimax(self, board: chess.Board, depth: int, maximizing_player: bool) -> float:
        """
        Performs a minimax algorithm to determine the best move for a plater given a board.

        Args:
            board (chess.Board): The board to evaluate
            depth (int): What depth to search to
            maximizing_player (bool): Is this is maximizing player (i.e. white to move)
        
        Returns:
            (float): The minimax evaluation for the given player
        """
        if depth == 0 or board.is_game_over():
            return self.__evaluate_board(board)

        legal_moves: List[chess.Move] = list(board.legal_moves)

        if maximizing_player:
            max_eval: float = float('-inf')
            for move in legal_moves:
                board.push(move)
                eval_score: float = self.__minimax(board, depth - 1, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score = self.__minimax(board, depth - 1, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
            return min_eval

    def find_move(self, board: chess.Board) -> chess.Move:
        """
        Finds the best move for a given board.

        Args:
            board (chess.Board): A board to evaluate

        Returns:
            (chess.Move): The best move given the board
        """
        legal_moves: List[chess.Move] = list(board.legal_moves)
        best_move: chess.Move = legal_moves[0] if legal_moves else None
        best_eval: float = float('-inf')

        for move in legal_moves:
            board.push(move)
            eval_score = -self.__minimax(board, self.depth - 1, -1)
            board.pop()

            if eval_score > best_eval:
                best_eval = eval_score
                best_move = move
        
        return best_move