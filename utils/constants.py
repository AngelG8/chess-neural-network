import chess
import pygame

# GLOBALS
TITLE_BAR = pygame.NOFRAME
WIDTH = HEIGHT = 800
DIMENSION = 8
SQ_SIZE = HEIGHT / DIMENSION
MAX_FPS = 60
FRAMES_PER_SQUARE = 5
IMAGES = {}

LIGHT_SQUARE_COLOR = pygame.Color(223, 223, 223, 1)
DARK_SQUARE_COLOR = pygame.Color(167, 199, 231, 1)

SELECTED_HIGHLIGHT_COLOR = pygame.Color(0, 0, 255)
MOVE_HIGHLIGHT_COLOR = pygame.Color(255, 215, 0)
CHECK_HIGHLIGHT_COLOR = pygame.Color(255, 40, 0)
LAST_MOVE_HIGHLIGHT_COLOR = pygame.Color(167, 255, 199)

FONT = "Arial"
FONT_SIZE = 50
FONT_COLOR = pygame.Color(255, 215, 0, 1)
FONT_SHADOW = pygame.Color(192, 192, 192, 1)

MINIMAX_DEPTH = 3
ALPHA_BETA_DEPTH = 3

DATASET_PGN_1GB = "/data/lichess_db_standard_rated_2014-08.pgn"
STOCKFISH = "/engines/stockfish/16/bin/stockfish"

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 10000
}

PIECE_SQUARE_TABLES = {
    chess.PAWN: {
        chess.WHITE: [
            [ 0,  0,  0,  0,  0,  0,  0,  0,],
            [50, 50, 50, 50, 50, 50, 50, 50,],
            [10, 10, 20, 30, 30, 20, 10, 10,],
            [ 5,  5, 10, 25, 25, 10,  5,  5,],
            [ 0,  0,  0, 20, 20,  0,  0,  0,],
            [ 5, -5,-10,  0,  0,-10, -5,  5,],
            [ 5, 10, 10,-20,-20, 10, 10,  5,],
            [ 0,  0,  0,  0,  0,  0,  0,  0,],
        ],
    },
    chess.KNIGHT: {
        chess.WHITE: [
            [-50,-40,-30,-30,-30,-30,-40,-50,],
            [-40,-20,  0,  0,  0,  0,-20,-40,],
            [-30,  0, 10, 15, 15, 10,  0,-30,],
            [-30,  5, 15, 20, 20, 15,  5,-30,],
            [-30,  0, 15, 20, 20, 15,  0,-30,],
            [-30,  5, 10, 15, 15, 10,  5,-30,],
            [-40,-20,  0,  5,  5,  0,-20,-40,],
            [-50,-40,-30,-30,-30,-30,-40,-50,],
        ]
    },
    chess.BISHOP: {
        chess.WHITE: [
            [-20,-10,-10,-10,-10,-10,-10,-20,],
            [-10,  0,  0,  0,  0,  0,  0,-10,],
            [-10,  0,  5, 10, 10,  5,  0,-10,],
            [-10,  5,  5, 10, 10,  5,  5,-10,],
            [-10,  0, 10, 10, 10, 10,  0,-10,],
            [-10, 10, 10, 10, 10, 10, 10,-10,],
            [-10,  5,  0,  0,  0,  0,  5,-10,],
            [-20,-10,-10,-10,-10,-10,-10,-20,],
        ]
    },
    chess.ROOK: {
        chess.WHITE: [
            [ 0,  0,  0,  0,  0,  0,  0,  0,],
            [ 5, 10, 10, 10, 10, 10, 10,  5,],
            [-5,  0,  0,  0,  0,  0,  0, -5,],
            [-5,  0,  0,  0,  0,  0,  0, -5,],
            [-5,  0,  0,  0,  0,  0,  0, -5,],
            [-5,  0,  0,  0,  0,  0,  0, -5,],
            [-5,  0,  0,  0,  0,  0,  0, -5,],
            [ 0,  0,  0,  5,  5,  0,  0,  0,],
        ]
    },
    chess.QUEEN: {
        chess.WHITE: [
            [-20,-10,-10, -5, -5,-10,-10,-20,],
            [-10,  0,  0,  0,  0,  0,  0,-10,],
            [-10,  0,  5,  5,  5,  5,  0,-10,],
            [ -5,  0,  5,  5,  5,  5,  0, -5,],
            [  0,  0,  5,  5,  5,  5,  0, -5,],
            [-10,  5,  5,  5,  5,  5,  0,-10,],
            [-10,  0,  5,  0,  0,  0,  0,-10,],
            [-20,-10,-10, -5, -5,-10,-10,-20,],
        ]
    },
    chess.KING: {
        chess.WHITE: [
            [-30,-40,-40,-50,-50,-40,-40,-30,],
            [-30,-40,-40,-50,-50,-40,-40,-30,],
            [-30,-40,-40,-50,-50,-40,-40,-30,],
            [-30,-40,-40,-50,-50,-40,-40,-30,],
            [-20,-30,-30,-40,-40,-30,-30,-20,],
            [-10,-20,-20,-20,-20,-20,-20,-10,],
            [ 20, 20,  0,  0,  0,  0, 20, 20,],
            [ 20, 30, 10,  0,  0, 10, 30, 20,],
        ]
    },
}

# Black's piece square table's are mirrored versions of white's
for piece_type, piece_tables in PIECE_SQUARE_TABLES.items():
    white_table = piece_tables[chess.WHITE]
    black_table = white_table[::-1] 
    piece_tables[chess.BLACK] = black_table