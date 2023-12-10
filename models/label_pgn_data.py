import os
import chess
import chess.pgn
import chess.engine
from concurrent.futures import ThreadPoolExecutor

BATCH_SIZE = 100
NUM_GAMES = 3000
MAX_WORKERS = 8

script_path = os.path.abspath(__file__)
stockfish_path = os.path.join(os.path.dirname(script_path), 'engines', 'stockfish', '16', 'bin', 'stockfish')

def get_stockfish_evaluation(board: chess.Board, time_limit_seconds: float) -> int:
    """
    Gets the stockfish evaluation for a board as a centipawn score.
    A centipawn score measures how good a position is for a player in
    terms of pawns. A score of 50 means White is up half a pawn, while
    a score of -200 means Black is up two pawns.

    Arguments:
        board (Board): A chess board to evaluate
        time (float): a maximum time limit for stockfish to run in seconds
    
    Returns:
        (int): The stockfish evaluation for a board as a centipawn score.
    """
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        # Analyze the position
        info: chess.engine.InfoDict = engine.analyse(board=board, limit=chess.engine.Limit(time=time_limit_seconds))
        # Get the evaluation score (in centipawns)
        score: int = info["score"].relative.score()

    return score
def process_game(game):
    board = chess.Board()
    fen_stockfish_evaluations = []

    for move in game.mainline_moves():
        board.push(move)
        fen = board.fen()

        stockfish_evaluation = get_stockfish_evaluation(board, time_limit_seconds=0.5)
        fen_stockfish_evaluations.append((fen, stockfish_evaluation))

    process_game.num_games += 1
    print("Games Processed:", process_game.num_games)
    return fen_stockfish_evaluations

def main():
    script_path = os.path.abspath(__file__)
    dataset_path = os.path.join(os.path.dirname(script_path), 'data', 'lichess_db_standard_rated_2014-08.pgn')
    pgn_file = open(dataset_path)

    output_file_path = os.path.join(os.path.dirname(script_path), 'data', 'fen_evaluations_lichess_db_standard_rated_2014-08.txt')
    output_file = open(output_file_path, 'w')

    process_game.num_games = 0
    num_games: int = 0
    no_more_games: bool = False
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while num_games < NUM_GAMES:
            if no_more_games:
                break
            futures = []
            i: int = 0
            while i < BATCH_SIZE:
                game = chess.pgn.read_game(pgn_file)
                if not (
                    "WhiteElo" in game.headers
                    and "BlackElo" in game.headers
                    and game.headers["WhiteElo"] != "?"
                    and game.headers["BlackElo"] != "?"
                    and int(game.headers["WhiteElo"]) > 2000
                    and int(game.headers["BlackElo"]) > 2000
                ):
                    continue
                if not game:
                    no_more_games = True
                    break
                future = executor.submit(process_game, game)
                futures.append(future)
                i += 1
            for future in futures:
                result = future.result()
                if result:
                    for fen, stockfish_evaluation in result:
                        output_file.write(f"{fen}\n{stockfish_evaluation}\n")
            
            num_games += len(futures)

    output_file.close()
    pgn_file.close()

if __name__ == "__main__":
    main()
