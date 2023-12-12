

import chess

from typing import Optional

from utils.constants import PIECE_VALUES, PIECE_SQUARE_TABLES

def get_material_score(board: chess.Board) -> float:
    """
    Evaluates a chess board using chess piece values.
    
    Args:
        board (chess.Board): A chess board to evaluate

    Returns:
        (float): The evaluation value of the board
    """
    if board.is_checkmate():
        return float('-inf') if board.turn == chess.WHITE else float('inf')
    elif board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    evaluation: float = 0.0
    for square in chess.SQUARES:
        piece: Optional[chess.Piece] = board.piece_at(square)
        if piece is not None:
            if piece.color == chess.WHITE:
                evaluation += PIECE_VALUES[piece.piece_type]
            else:
                evaluation -= PIECE_VALUES[piece.piece_type]

    return evaluation
    
def get_piece_square_material_score(board: chess.Board) -> float:
    """
    Evaluates a chess board using chess piece values and piece-square tables.

    Args:
        board (chess.Board): A chess board to evaluate

    Returns:
        (float): The evaluation value of the board
    """
    if board.is_checkmate():
        return float('-inf') if board.turn == chess.WHITE else float('inf')
    elif board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    evaluation: float = 0.0
    for square in chess.SQUARES:
        piece: Optional[chess.Piece] = board.piece_at(square)
        if piece is not None:
            rank: int = chess.square_rank(square)
            file: int = chess.square_file(square)
            # Evaluate based on piece values
            color: int = 1 if piece.color == chess.WHITE else -1
            evaluation += color * PIECE_VALUES[piece.piece_type]
            evaluation += color * PIECE_SQUARE_TABLES[piece.piece_type][piece.color][rank][file]
    return evaluation