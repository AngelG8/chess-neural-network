import chess
import random

from typing import List
from models.chess_ai import ChessAI

class RandomAI(ChessAI):
    """
    Represents a chess AI that makes random moves.
    """

    def find_move(self, board: chess.Board) -> chess.Move:
        """
        Generates a random move.

        Returns:
            (move): A random move.
        """
        legal_moves: List[chess.Move] = list(board.legal_moves)
        return random.choice(legal_moves)