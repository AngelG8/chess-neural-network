import chess
import random
import math

from typing import List
from models.chess_ai import ChessAI

class MCTSTreeNode:
    def __init__(self, move, parent=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

class MCTSAI(ChessAI):
    """
    Represents a chess AI that makes moves based on the Monte Carlo Tree Search algorithm.
    """
    def __init__(self, iterations=10):
        self.iterations = iterations

    def find_move(self, board: chess.Board) -> chess.Move:
        root = MCTSTreeNode(None)

        for _ in range(self.iterations):
            new_node = MCTSTreeNode(None)

            node = self.selection(root)
            outcome = self.simulation(node, board.copy())

            new_node.parent = node

            self.backpropagation(new_node, outcome)

            node.children.append(new_node)

        best_child = self.best_child(root)
        return best_child.move

    def selection(self, node: MCTSTreeNode) -> MCTSTreeNode:
        while node.children:
            node = self.ucb_select_child(node)
        return node

    def ucb_select_child(self, node: MCTSTreeNode) -> MCTSTreeNode:
        exploration_weight = 1.44  # Tunable parameter
        return max(node.children, key=lambda child: (child.wins / child.visits) +
                                                    exploration_weight * math.sqrt(math.log(node.visits) / child.visits))

    def simulation(self, node: MCTSTreeNode, board: chess.Board) -> int:
        list_of_moves = []

        while not board.is_game_over():
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
            list_of_moves.append(move)
            board.push(move)
        node.move = list_of_moves[0]
        return self.get_outcome(board)

    def backpropagation(self, node: MCTSTreeNode, outcome: int):
        while node:
            node.visits += 1
            node.wins += outcome
            node = node.parent

    def best_child(self, node: MCTSTreeNode) -> MCTSTreeNode:
        return max(node.children, key=lambda child: child.visits)

    def get_outcome(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return 1 if board.turn == chess.WHITE else -1
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
            return 0
        else:
            return 0
