import time
import random
import copy # Keep standard copy import
import pygame
import math
import os

# --- Removed deepcopy_ignore_surfaces helper function ---
# Relies on __deepcopy__ implemented in Board, Square, Piece classes above

# --- Piece Material Scores ---
SCORES_DICT = {
    " ": 100,  # pawn
    "N": 320,  # knight
    "B": 330,  # bishop
    "R": 500,  # rook
    "S": 500,  # star
    "Q": 900,  # queen
    "J": 900,  # joker
    "K": 20000 # king
}

# --- Piece-Square Tables (Example Heuristics - Need Tuning!) ---
# (Tables remain the same as previous version)
PAWN_TABLE = [
    [ 0,  0,  0,  0,  0,  0],[10, 10,  0,  0, 10, 10],[ 5,  5, 10, 10,  5,  5],
    [ 0,  0, 20, 20,  0,  0],[ 5, 10, 25, 25, 10,  5],[ 0,  0,  0,  0,  0,  0]
]
KNIGHT_TABLE = [
    [-50,-40,-30,-30,-40,-50],[-40,-20,  0,  0,-20,-40],[-30,  0, 10, 10,  0,-30],
    [-30,  5, 15, 15,  5,-30],[-40,-20,  0,  0,-20,-40],[-50,-40,-30,-30,-40,-50]
]
BISHOP_TABLE = [
    [-20,-10,-10,-10,-10,-20],[-10,  0,  0,  0,  0,-10],[-10,  0,  5,  5,  0,-10],
    [-10,  5,  5,  5,  5,-10],[-10,  0,  5,  5,  0,-10],[-20,-10,-10,-10,-10,-20]
]
ROOK_TABLE = [
    [ 0,  0,  0,  0,  0,  0],[ 5, 10, 10, 10, 10,  5],[-5,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0, -5],[-5,  0,  0,  0,  0, -5],[ 0,  0,  0,  5,  5,  0]
]
QUEEN_TABLE = [
    [-20,-10,-10, -5, -5,-10,-20],[-10,  0,  0,  0,  0,  0,-10],[-10,  0,  5,  5,  5,  5,-10],
    [ -5,  0,  5,  5,  5,  5, -5],[-10,  5,  5,  5,  5,  0,-10],[-20,-10,-10, -5, -5,-10,-20]
]
KING_TABLE_EARLY = [
    [-30,-40,-40,-50,-50,-40,-30],[-30,-40,-40,-50,-50,-40,-30],[-30,-40,-40,-50,-50,-40,-30],
    [-30,-40,-40,-50,-50,-40,-30],[-10,-20,-20,-20,-20,-20,-10],[ 20, 30, 10,  0,  0, 10, 30, 20]
]
STAR_TABLE = [
    [-40,-30,-20,-20,-30,-40],[-30,-10,  5,  5,-10,-30],[-20,  5, 15, 15,  5,-20],
    [-20, 10, 15, 15, 10,-20],[-30,-10,  5,  5,-10,-30],[-40,-30,-20,-20,-30,-40]
]
JOKER_TABLE = QUEEN_TABLE

PIECE_TABLES = {
    " ": PAWN_TABLE, "N": KNIGHT_TABLE, "B": BISHOP_TABLE, "R": ROOK_TABLE,
    "Q": QUEEN_TABLE, "K": KING_TABLE_EARLY, "S": STAR_TABLE, "J": JOKER_TABLE
}


class Bot:
    """
    Optimized Minimax bot using Iterative Deepening and Alpha-Beta pruning.
    Single-threaded. Attempts optimized copying via custom __deepcopy__ methods
    in Board, Square, Piece classes (must be implemented in those classes).
    """
    def __init__(self, max_depth=7, time_limit=0.090, scores_dict=SCORES_DICT, piece_tables=PIECE_TABLES):
        """ Initializes the bot. """
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.calculation_time = 0
        self.nodes_visited = 0

        self.SCORES_DICT = scores_dict
        self.PIECE_TABLES = piece_tables

    # --- Core Search Logic ---

    def get_possible_moves(self, side, board):
        """Gets all valid moves for the given side."""
        return board.get_all_valid_moves(side)

    def evaluate_board(self, board, player_side):
        """ Evaluates the board state: material + positional bonuses. (Unchanged) """
        self.nodes_visited += 1
        evaluation = 0
        opponent_side = 'black' if player_side == 'white' else 'white'

        if board.is_in_checkmate(opponent_side): return 30000
        if board.is_in_checkmate(player_side): return -30000
        if board.is_in_draw(): return 0

        for square in board.squares:
            piece = square.occupying_piece
            if piece:
                piece_color = piece.color
                notation_key = piece.notation
                piece_value = self.SCORES_DICT.get(notation_key, 0)
                positional_bonus = 0
                table = self.PIECE_TABLES.get(notation_key)
                if table:
                    y, x = square.y, square.x
                    mirrored_y = 5 - y
                    positional_bonus = table[y][x] if piece_color == 'white' else table[mirrored_y][x]
                total_value = piece_value + positional_bonus
                evaluation += total_value if piece_color == player_side else -total_value
        return evaluation

    def simulate_move(self, board, start_pos, end_pos):
        """
        Creates a deep copy using standard copy.deepcopy, relying on custom
        __deepcopy__ methods in Board/Square/Piece classes, and simulates a move.
        """
        try:
            # Use standard deepcopy, assuming custom __deepcopy__ are implemented
            # The memo dictionary is handled internally by copy.deepcopy
            new_board = copy.deepcopy(board)
            # Perform the move on the copied board
            success = new_board.handle_move(start_pos, end_pos)
            if not success:
                # print(f"Warning: Simulation of move {start_pos}->{end_pos} failed on copied board.")
                return None
            return new_board
        except Exception as e:
            # import traceback
            # print(f"Error during deepcopy or simulation: {e}")
            # traceback.print_exc()
            return None # Indicate simulation failure

    def minimax(self, board, side, depth, alpha, beta, maximizing_player):
        """ Minimax algorithm with Alpha-Beta pruning. (Unchanged) """
        self.nodes_visited += 1
        is_terminal = board.is_in_checkmate('white') or \
                      board.is_in_checkmate('black') or \
                      board.is_in_draw()

        if depth == 0 or is_terminal:
            return self.evaluate_board(board, side)

        moves = self._get_ordered_moves(board, side)
        if not moves:
             return self.evaluate_board(board, side)

        if maximizing_player:
            max_eval = float('-inf')
            for init_pos, end_pos in moves:
                simulated_board = self.simulate_move(board, init_pos, end_pos)
                if simulated_board is None: continue
                opponent_side = simulated_board.turn
                eval_score = self.minimax(simulated_board, opponent_side, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha: break
            return max_eval
        else: # Minimizing player
            min_eval = float('inf')
            for init_pos, end_pos in moves:
                simulated_board = self.simulate_move(board, init_pos, end_pos)
                if simulated_board is None: continue
                opponent_side = simulated_board.turn
                eval_score = self.minimax(simulated_board, opponent_side, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha: break
            return min_eval

    def _get_ordered_moves(self, board, side):
        """ Basic move ordering: captures first, then others. (Unchanged) """
        moves = board.get_all_valid_moves(side)
        capture_moves = []
        other_moves = []
        for move in moves:
            start_pos, end_pos = move
            end_square = board.get_square_from_pos(end_pos)
            if end_square.occupying_piece is not None and end_square.occupying_piece.color != side:
                capture_moves.append(move)
            else:
                other_moves.append(move)
        random.shuffle(other_moves)
        return capture_moves + other_moves

    def get_best_move_at_depth(self, board, side, depth):
        """ Finds best move for a specific depth (Single-Threaded). (Unchanged logic) """
        best_moves = []
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        moves = self._get_ordered_moves(board, side)

        if not moves: return None, float('-inf')
        # Handle single move case slightly differently to get score
        if len(moves) == 1:
             sim_board = self.simulate_move(board, *moves[0])
             score = float('-inf')
             if sim_board:
                 opponent_side = sim_board.turn
                 score = self.minimax(sim_board, opponent_side, depth - 1, alpha, beta, False)
             return moves[0], score

        # Sequential Evaluation Loop
        for move in moves:
            simulated_board = self.simulate_move(board, *move) # Unpack move tuple
            if simulated_board is None: continue

            opponent_side = simulated_board.turn
            move_value = self.minimax(simulated_board, opponent_side, depth - 1, alpha, beta, False)

            if move_value > best_value:
                best_value = move_value
                best_moves = [move]
                alpha = max(alpha, best_value) # Update alpha for subsequent searches at this level
            elif move_value == best_value:
                best_moves.append(move)

        selected_move = random.choice(best_moves) if best_moves else (random.choice(moves) if moves else None)
        # Return best_value which is from the perspective of 'side'
        return selected_move, best_value


    # --- Main move function implementing Iterative Deepening --- (Unchanged)

    def move(self, side, board):
        """ Calculates the best move using Iterative Deepening (Single-Threaded). """
        self.start_time_for_move = time.time()
        self.calculation_time = 0
        self.nodes_visited = 0
        best_move_overall = None
        initial_moves = self.get_possible_moves(side, board)
        if not initial_moves: return None

        is_terminal = board.is_in_checkmate('white') or \
                      board.is_in_checkmate('black') or \
                      board.is_in_draw()
        if is_terminal: return None

        last_completed_depth = 0
        last_score = float('-inf')

        for depth in range(1, self.max_depth + 1):
            time_elapsed = time.time() - self.start_time_for_move
            time_remaining = self.time_limit - time_elapsed
            if time_remaining < (self.time_limit * 0.10) or time_elapsed > self.time_limit :
                 break
            try:
                current_best_move, current_best_score = self.get_best_move_at_depth(board, side, depth)
                if current_best_move is not None:
                    best_move_overall = current_best_move
                    last_completed_depth = depth
                    last_score = current_best_score
                if current_best_score >= 30000: break # Checkmate found
            except Exception as e:
                # import traceback; traceback.print_exc()
                break
            time_elapsed_after = time.time() - self.start_time_for_move
            if time_elapsed_after >= self.time_limit: break

        self.calculation_time = time.time() - self.start_time_for_move
        if best_move_overall is None:
            best_move_overall = random.choice(initial_moves) if initial_moves else None

        # print(f"Bot ({side}): Chose {best_move_overall}. Depth: {last_completed_depth}. Score: {last_score:.0f}. Nodes: {self.nodes_visited}. Time: {self.calculation_time:.4f}s")
        # if self.calculation_time > self.time_limit: print(f"Warning: Time limit exceeded")
        return best_move_overall

