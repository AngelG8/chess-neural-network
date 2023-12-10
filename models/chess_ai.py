import chess
from abc import ABC, abstractmethod

class ChessAI(ABC):
    """
    Represents a chess AI that can play a game of chess.
    """
    @abstractmethod
    def find_move(self, board: chess.Board) -> chess.Move:
        pass
