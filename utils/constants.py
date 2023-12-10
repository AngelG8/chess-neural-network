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
ALPHA_BETA_DEPTH = 4

DATASET_PGN_1GB = "/data/lichess_db_standard_rated_2014-08.pgn"
STOCKFISH = "/engines/stockfish/16/bin/stockfish"

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100
}