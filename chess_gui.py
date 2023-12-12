import os

import chess
import chess.svg
import chess.engine
import multiprocessing
import pygame

from copy import copy
from typing import List, Optional, Tuple

from models.chess_ai import ChessAI
from models.random_ai import RandomAI
from models.minimax_ai import MinimaxAI
from models.alpha_beta_ai import AlphaBetaAI
from models.mtcs_ai import MCTSAI
from models.cnn_stockfish_ai import CNNAI

from utils.constants import WIDTH, HEIGHT, DIMENSION, SQ_SIZE, MAX_FPS, FRAMES_PER_SQUARE, IMAGES, \
    LIGHT_SQUARE_COLOR, DARK_SQUARE_COLOR, SELECTED_HIGHLIGHT_COLOR, MOVE_HIGHLIGHT_COLOR, \
    CHECK_HIGHLIGHT_COLOR, LAST_MOVE_HIGHLIGHT_COLOR, FONT, FONT_SIZE, FONT_COLOR, FONT_SHADOW, TITLE_BAR

from utils.evaluation_functions import get_material_score, get_piece_square_material_score

class ChessGUI:
    """
    A GUI that plays a game of chess.

    Attributes:
        player_one_human (bool): Whether player one is a human or computer
        player_two_human (bool): Whether player two is a human or computer
        computer_one_ai (ChessAI): What chess AI is player one (given they are a computer)?
        computer_two_ai (ChessAI): What chess AI is player two (given they are a computer)?
        animated (bool): Whether to animate chess moves (for GUI only)
        
        selected_piece (Optional[Piece]): Which piece a player has selected (if any)
        move_to_play (Optional[Move]): A chess move for the current player to make
        moves_made (List[Move]): A log of moves made in the game so far
        board (Board): The chess board the game is being played on

        screen (Surface): A pygame surface that displays the game
        clock (Clock): A pygame clock to keep track of timed GUI events (e.g. animating frames)
    """
    def __init__(self):
        self.player_one_human: bool = True
        self.player_two_human: bool = False
        self.computer_one_ai: ChessAI = AlphaBetaAI()
        self.computer_two_ai: ChessAI = AlphaBetaAI()
        self.animated: bool = True

        self.selected_piece: Optional[chess.Piece] = None
        self.move_to_play: Optional[chess.Move] = None
        self.moves_made: List[chess.Move] = []
        self.board: chess.Board = chess.Board()

        pygame.init()
        pygame.display.set_caption("Chess Game")
        self.screen: pygame.Surface = pygame.display.set_mode((WIDTH, HEIGHT), TITLE_BAR)
        self.clock: pygame.time.Clock = pygame.time.Clock()

    def __load_images(self) -> None:
        """
        Initialize a global dictionary of images for chess pieces.

        Returns:
            None
        """
        black_pieces: List[str] = ["p", "r", "n", "b", "q", "k"]
        white_pieces: List[str] = ["P", "R", "N", "B", "Q", "K"]
        for piece in black_pieces:
            IMAGES[piece] = pygame.transform.scale(pygame.image.load("images/pieces/black/" + piece + ".png"), (SQ_SIZE, SQ_SIZE))
        for piece in white_pieces:
            IMAGES[piece] = pygame.transform.scale(pygame.image.load("images/pieces/white/" + piece + ".png"), (SQ_SIZE, SQ_SIZE))

    def __create_square_highlight(self, color: pygame.Color, alpha: int = 100) -> pygame.Surface:
        """
        Creates a highlight to overlay on top of a chess board square.

        Arguments:
            color (Color): The color of the highlight
            alpha (int): The alpha value (transparency) of the highlight

        Returns:
            (Surface): The highlight as a pygame surface
        """
        highlight: pygame.Surface = pygame.Surface((SQ_SIZE, SQ_SIZE))
        highlight.set_alpha(alpha)
        highlight.fill(color)
        return highlight

    def __draw_board(self) -> None:
        """
        Visualizes a board.

        Arguments:
            screen (Surface): The screen to draw the board on.
        """
        for row in range(DIMENSION):
            for col in range(DIMENSION):
                color: pygame.Color = LIGHT_SQUARE_COLOR if (row + col) % 2 == 0 else DARK_SQUARE_COLOR
                pygame.draw.rect(self.screen, color, (col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def __draw_pieces(self, exclude: List[Tuple[int, int]]=[]) -> None:
        """
        Visualizes the pieces on a board.

        Arguments:
            exclude (List[Tuple[int, int]]): A list of (row, col) indicating spaces not to draw pieces on.

        Returns:
            None
        """
        for row in range(DIMENSION):
                for col in range(DIMENSION):
                    if (row, col) in exclude:
                        continue
                    square: chess.Square = chess.square(col, DIMENSION - 1 - row)
                    piece: chess.Piece = self.board.piece_at(square)
                    if piece:
                        piece_image = IMAGES[piece.symbol()]
                        self.screen.blit(piece_image, (col * SQ_SIZE, row * SQ_SIZE))

    def __highlight_tiles(self) -> None:
        """
        Highlights a selected piece's valid moves.

        Arguments:
            screen (Surface): The screen to draw the game state on.
            selected_piece (Piece): The current selected piece.
            board (Board): The board to draw on.

        Returns:
            None
        """
        for row in range(DIMENSION):
            for col in range(DIMENSION):
                square: chess.Square = chess.square(col, DIMENSION - 1 - row)
                piece: chess.Piece = self.board.piece_at(square)
                # Highlight the selected piece in another color
                if self.selected_piece == square:
                    self.screen.blit(self.__create_square_highlight(SELECTED_HIGHLIGHT_COLOR, 31), (col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

                # Highlight legal moves for the selected piece
                if self.selected_piece is not None and square in [move.to_square for move in self.board.legal_moves if move.from_square == self.selected_piece]:
                    self.screen.blit(self.__create_square_highlight(MOVE_HIGHLIGHT_COLOR), (col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

                # Highlight the king in red if it's in check
                if self.board.is_check() and piece and piece.piece_type == chess.KING and piece.color == self.board.turn:
                    self.screen.blit(self.__create_square_highlight(CHECK_HIGHLIGHT_COLOR), (col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))
                
                if self.moves_made:
                    from_square: chess.Move = self.moves_made[-1].from_square
                    to_square: chess.Move = self.moves_made[-1].to_square
                    self.screen.blit(self.__create_square_highlight(LAST_MOVE_HIGHLIGHT_COLOR, 31), 
                                     (chess.square_file(from_square) * SQ_SIZE, (DIMENSION - 1 - chess.square_rank(from_square)) * SQ_SIZE, SQ_SIZE, SQ_SIZE))
                    self.screen.blit(self.__create_square_highlight(LAST_MOVE_HIGHLIGHT_COLOR, 31), 
                                     (chess.square_file(to_square) * SQ_SIZE, (DIMENSION - 1 - chess.square_rank(to_square)) * SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def draw_promotion_overlay(self) -> None:
        """
        Draws an overlay on the GUI to allow human players 
        to select which piece to promote a pawn to.
        """
        # Draw a transparent overlay with clickable areas for promotion choices
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # Transparent black overlay

        # Define clickable areas for each promotion choice
        button_size = (SQ_SIZE, SQ_SIZE)
        button_positions = {
            'q': (2 * SQ_SIZE, 4 * SQ_SIZE),
            'r': (3 * SQ_SIZE, 4 * SQ_SIZE),
            'n': (4 * SQ_SIZE, 4 * SQ_SIZE),
            'b': (5 * SQ_SIZE, 4 * SQ_SIZE),
        }

        for choice, pos in button_positions.items():
            pygame.draw.rect(overlay, (255, 255, 255), (*pos, *button_size))
            font = pygame.font.SysFont(FONT, FONT_SIZE, True, False)
            # text_object = font.render(text, False, FONT_SHADOW)
            text = font.render(choice.upper(), True, (0, 0, 0))
            overlay.blit(text, (pos[0] + SQ_SIZE // 4, pos[1] + SQ_SIZE // 4))

        self.screen.blit(overlay, (0, 0))
        pygame.display.flip()

    def get_promotion_choice(self, event: pygame.event) -> Optional[chess.PieceType]:
        """
        Gets the piece that a player wants to promote a pawn to.

        Arguments:
            event (event): A pygame mouse cursor event to indicate which piece to select.

        Returns:
            (Optional[chess.PieceType]): The chess piece to promote to
        """
        # Check if the player clicked on a promotion choice
        button_positions = {
            'q': (2 * SQ_SIZE, 4 * SQ_SIZE, SQ_SIZE, SQ_SIZE),
            'r': (3 * SQ_SIZE, 4 * SQ_SIZE, SQ_SIZE, SQ_SIZE),
            'n': (4 * SQ_SIZE, 4 * SQ_SIZE, SQ_SIZE, SQ_SIZE),
            'b': (5 * SQ_SIZE, 4 * SQ_SIZE, SQ_SIZE, SQ_SIZE),
        }

        for choice, pos in button_positions.items():
            if pos[0] <= event.pos[0] <= pos[0] + SQ_SIZE and pos[1] <= event.pos[1] <= pos[1] + SQ_SIZE:
                return chess.Piece.from_symbol(choice.upper()).piece_type

        return None

    def promote_pawn(self, move: chess.Move) -> chess.Move:
        """
        Promotes a pawn to another piece.

        Arguments:
            move (Move): A chess move with no promotion information (promotes to queen by default)
        
        Returns:
            (Move): A new move containing a particular promotion piece (e.g. knight)
        """
        piece_at_square: chess.Piece = self.board.piece_at(move.from_square)

        if piece_at_square and piece_at_square.piece_type == chess.PAWN and \
            chess.square_rank(move.to_square) in [0, DIMENSION - 1]:
            # Draw the promotion overlay
            self.draw_promotion_overlay()

            promotion_choice = None
            while promotion_choice is None:
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        promotion_choice = self.get_promotion_choice(event)

            # Return the move with the specified promotion
            return chess.Move(move.from_square, move.to_square, promotion=promotion_choice)
        return move

    def animate_move(self, move: chess.Move) -> None:
        """
        Animates a chess move.

        Arguments:
            move (Move): The move to animate.

        Returns:
            None
        """
        start_square: chess.Square = move.from_square
        end_square: chess.Square = move.to_square
        start_row: int = chess.square_rank(start_square)
        start_col: int = chess.square_file(start_square)
        end_row: int = chess.square_rank(end_square)
        end_col: int = chess.square_file(end_square)
        piece: chess.Piece = self.board.piece_at(start_square)
        delta_row: int = end_row - start_row
        delta_col: int = end_col - start_col
        frame_count: int = (abs(delta_row) + abs(delta_col)) * FRAMES_PER_SQUARE

        for frame in range(frame_count + 1):
            frame_row: float = (start_row) + delta_row * frame / frame_count
            frame_col: float = start_col + delta_col * frame / frame_count
            self.__draw_board()
            # Python's chess library and pygame have coordinate systems that start at the bottom left
            # and top right respectively, so the rows must be flipped to correctly animate the movement
            self.__draw_pieces(exclude=[(DIMENSION - 1 - start_row, start_col)])
            self.screen.blit(IMAGES[piece.symbol()], pygame.Rect(frame_col * SQ_SIZE, (DIMENSION - 1 - frame_row) * SQ_SIZE, SQ_SIZE, SQ_SIZE))
            pygame.display.flip()
            self.clock.tick(MAX_FPS)

    def get_human_move(self, event) -> Optional[chess.Move]:
        """
        Handles a human move made through the GUI.

        Arguments:
            event (event): A pygame mouse cursor event to indicate 
                           which piece the player has selected.
        
        Returns:
            (Optional[chess.Move]): A move the player made, or none at all if they decide to no longer play the move
                                    (e.g. they select a piece and then deselect it)
        """
        col: int = int(event.pos[0] // SQ_SIZE)
        row: int = int(DIMENSION - 1 - (event.pos[1] // SQ_SIZE))
        square: chess.Square = chess.square(col, row)
        piece: Optional[chess.Piece] = self.board.piece_at(square)

        if self.selected_piece is None:
            # No piece is currently selected
            if piece and piece.color == self.board.turn:
                # Select a piece
                self.selected_piece = square
        elif self.selected_piece == square:
            # Unselect the current piece if clicked again
            self.selected_piece = None
        elif self.selected_piece is not None and square in [move.to_square for move in self.board.legal_moves if move.from_square == self.selected_piece]:
            return chess.Move(self.selected_piece, square)
        elif piece and piece.color == self.board.turn:
            # Change the selection to a new piece
            self.selected_piece = square
        else:
            # Clicked on an empty square or an opponent's piece
            self.selected_piece = None

    # Main game loop
    def run_game_gui(self) -> None:
        """
        Launches a GUI to play an interactive game of chess.
        """
        self.__load_images()
        running: bool = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.player_one_human or self.player_two_human:
                        self.move_to_play = self.get_human_move(event)
                # Check for the "Z" key press to undo the move
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_z:
                        if self.board.move_stack:
                            # Undo the last move
                            self.board.pop()
                            self.moves_made.pop()
                            # If a human is playing against a computer, then undoing should
                            # undo both the player's and computer's last moves
                            if self.player_one_human ^ self.player_two_human:
                                self.board.pop()
                                self.moves_made.pop()
                            self.selected_piece = None
                    elif event.key == pygame.K_q:
                        pygame.quit()
                    elif event.key == pygame.K_r:
                        # Reset the game
                        self.board = chess.Board()
                        self.moves_made = []
                        self.selected_piece = None
                        self.move_to_play = None

            if not self.board.is_game_over():
                if not self.player_one_human and self.board.turn == chess.WHITE:
                    self.move_to_play = self.computer_one_ai.find_move(self.board)
                elif not self.player_two_human and self.board.turn == chess.BLACK:
                    self.move_to_play = self.computer_two_ai.find_move(self.board)

            if self.move_to_play:
                if self.animated:
                    self.animate_move(self.move_to_play)
                if self.player_one_human and self.board.turn == chess.WHITE or \
                    self.player_two_human and self.board.turn == chess.BLACK:
                    self.move_to_play = self.promote_pawn(self.move_to_play)
                self.board.push(self.move_to_play)
                self.moves_made.append(copy(self.move_to_play))
                

                self.selected_piece = None
                self.move_to_play = None

            self.__draw_board()
            self.__highlight_tiles()
            self.__draw_pieces()

            if self.board.is_game_over():
                winner_text: str
                if self.board.is_checkmate():
                    winner: str =  "White" if self.board.turn == chess.BLACK else "Black"
                    winner_text = f"{winner} Wins By Checkmate"
                else:
                    winner_text = "Draw"

                text_position = (WIDTH // 2 - 50, HEIGHT // 2 - 20)
                # Render the text with a black outline
                font = pygame.font.SysFont(FONT, FONT_SIZE)
                text_surface = font.render(winner_text, True, (0, 0, 0))
                text_surface_outline = font.render(winner_text, True, (255, 255, 255))

                # Calculate the position to center the text
                text_width, text_height = text_surface.get_size()
                text_position = ((WIDTH - text_width) // 2, (HEIGHT - text_height) // 2)

                # Draw the text with a black outline
                outline_offset = 2
                self.screen.blit(text_surface_outline, (text_position[0] - outline_offset, text_position[1] - outline_offset))
                self.screen.blit(text_surface_outline, (text_position[0] + outline_offset, text_position[1] + outline_offset))
                self.screen.blit(text_surface_outline, (text_position[0] - outline_offset, text_position[1] + outline_offset))
                self.screen.blit(text_surface_outline, (text_position[0] + outline_offset, text_position[1] - outline_offset))

                # Draw the main text
                self.screen.blit(text_surface, text_position)

            pygame.display.flip()
        
        pygame.quit()

    def simulate_games_parallel(self, file_path: str, num_games: int=104, num_processes: int=8) -> None:
        """
        Run a simulation between two chess AIs.

        Arguments:
            file_path (str): A file path to write the results to.
            num_games (int): How many simulations to run.
            num_processes (int): How many processes to use to parallelize the workload.
        """
        self.screen = None
        self.clock = None
        pool = multiprocessing.Pool(processes=num_processes)
        games_per_process = num_games // num_processes

        results = pool.starmap(self.run_simulation_batch, [(file_path, games_per_process)] * num_processes)

        pool.close()
        pool.join()

        wins_player_one = sum(result[0] for result in results)
        wins_player_two = sum(result[1] for result in results)
        draws = sum(result[2] for result in results)

        print(f"Player One Wins: {wins_player_one}")
        print(f"Player Two Wins: {wins_player_two}")
        print(f"Draws: {draws}")
        file = open(file_path, "a+")
        file.write(f"Player One Wins: {wins_player_one}\n")
        file.write(f"Player Two Wins: {wins_player_two}\n")
        file.write(f"Game Ended in Draw: {draws}\n")

    def run_simulation_batch(self, file_path: str, num_games: int) -> Tuple[int, int, int]:
        """
        Runs a batch of simulations.

        Arguments:
            file_path (str): A file path to write the results to.
            num_games (int): How many simulations to run.
        
        Returns:
            (Tuple[int, int, int]): The number of games player one won, 
                                    the number of games player two won,
                                    the number of draws
        """
        file = open(file_path, "a+")

        wins_player_one: int = 0
        wins_player_two: int = 0
        draws: int = 0

        for _ in range(num_games):
            self.board = chess.Board()
            self.moves_made = []
            self.selected_piece = None
            self.move_to_play = None

            while not self.board.is_game_over():
                if self.board.turn == chess.WHITE:
                    move = self.computer_one_ai.find_move(self.board)
                else:
                    move = self.computer_two_ai.find_move(self.board)

                self.board.push(move)
                self.moves_made.append(copy(move))

            if self.board.is_checkmate():
                winner = "White" if self.board.turn == chess.BLACK else "Black"
                if winner == "White":
                    wins_player_one += 1
                    print("Player One Won")
                else:
                    wins_player_two += 1
                    print("Player Two Won")
            else:
                draws += 1
                print("Game Ended in Draw")

        file.close()
        return wins_player_one, wins_player_two, draws


if __name__ == "__main__":
    chess_gui = ChessGUI()
    chess_gui.player_one_human = False
    chess_gui.player_two_human = False
    chess_gui.animated = False
    chess_gui.computer_one_ai = AlphaBetaAI(evaluation_function=get_piece_square_material_score)
    chess_gui.computer_two_ai = RandomAI()

    chess_gui.run_game_gui()

    # Run simulations and record the results in a text file
    # current_path: str = os.path.abspath(__file__)
    # file_path: str = os.path.join(os.path.dirname(current_path), 'simulation', "alphabeta_vs_random.txt")
    # chess_gui.simulate_games_parallel(file_path=file_path, num_games=104, num_processes=8)
