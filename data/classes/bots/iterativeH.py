import time
import random
import copy
import pygame
import concurrent.futures
import math
import os  # To get CPU count

# --- Helper function (Ensure this is available or included) ---
def deepcopy_ignore_surfaces(obj, memo=None):
    if memo is None:
        memo = {}
    if id(obj) in memo:
        return memo[id(obj)]
    if isinstance(obj, pygame.Surface):
        return obj
    if isinstance(obj, dict):
        copied = {}
        memo[id(obj)] = copied
        for k, v in obj.items():
            copied[deepcopy_ignore_surfaces(k, memo)] = deepcopy_ignore_surfaces(v, memo)
        return copied
    elif isinstance(obj, list):
        copied = []
        memo[id(obj)] = copied
        for item in obj:
            copied.append(deepcopy_ignore_surfaces(item, memo))
        return copied
    elif hasattr(obj, '__dict__'):
        try:
            copied = obj.__class__.__new__(obj.__class__)
            memo[id(obj)] = copied
            for k, v in obj.__dict__.items():
                setattr(copied, k, deepcopy_ignore_surfaces(v, memo))
            return copied
        except TypeError:
            return obj
    else:
        try:
            return copy.deepcopy(obj, memo)
        except TypeError:
            return obj
# --- End Helper Function ---

class Bot:
    def __init__(self, max_depth=10, time_limit=0.095):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.calculation_time = 0

        cpu_cores = os.cpu_count()
        self.max_threads = min(cpu_cores if cpu_cores else 4, 8)

        self.SCORES_DICT = {
            " ": 1,   # pawn
            "N": 5,   # knight
            "B": 3,   # bishop
            "R": 5,   # rook
            "S": 5,   # star
            "Q": 15,  # queen
            "J": 9,   # joker
            "K": 10000  # king
        }

        self.PAWN_TABLE = [
            [0,   0,   0,   0,   0,   0],
            [5,   5,   5,   5,   5,   5],
            [1,   1,   2,   2,   1,   1],
            [0.5, 0.5, 1,   1,   0.5, 0.5],
            [0,   0,   0,   0,   0,   0],
            [-1, -1, -1, -1, -1, -1],
        ]

        self.best_move_from_last_iter = None

    def get_possible_moves(self, side, board):
        return board.get_all_valid_moves(side)

    def evaluate_board(self, board, player_side):
        evaluation = 0
        opponent_side = 'black' if player_side == 'white' else 'white'

        if board.is_in_checkmate(opponent_side):
            return float('inf')
        if board.is_in_checkmate(player_side):
            return float('-inf')
        if board.is_in_draw():
            return 0

        board_state = board.get_board_state()
        for y in range(len(board_state)):
            for x in range(len(board_state[y])):
                piece_code = board_state[y][x]
                if piece_code != "":
                    piece_color_char = piece_code[0]
                    piece_type_char = piece_code[1]

                    notation_key = piece_type_char if piece_type_char != "P" else " "
                    if piece_type_char == "J":
                        notation_key = "J"

                    piece_value = self.SCORES_DICT.get(notation_key, 0)

                    positional_bonus = 0
                    if piece_type_char == "P":
                        if piece_color_char == "w":
                            positional_bonus = self.PAWN_TABLE[y][x]
                        else:
                            mirrored_y = 5 - y
                            positional_bonus = self.PAWN_TABLE[mirrored_y][x]

                    total_value = piece_value + positional_bonus

                    if (piece_color_char == "w" and player_side == "white") or \
                       (piece_color_char == "b" and player_side == "black"):
                        evaluation += total_value
                    else:
                        evaluation -= total_value
        return evaluation

    def simulate_move(self, board, start_pos, end_pos):
        try:
            new_board = deepcopy_ignore_surfaces(board)
            success = new_board.handle_move(start_pos, end_pos)
            if not success:
                return None
            return new_board
        except Exception:
            return None

    def minimax(self, board, side, depth, alpha, beta, maximizing_player):
        if board.is_in_checkmate('white') or board.is_in_checkmate('black') or board.is_in_draw():
            return self.evaluate_board(board, side)
        if depth == 0:
            return self.evaluate_board(board, side)

        moves = board.get_all_valid_moves(side)
        if not moves:
            return self.evaluate_board(board, side)

        if maximizing_player:
            max_eval = float('-inf')
            for init_pos, end_pos in moves:
                simulated_board = self.simulate_move(board, init_pos, end_pos)
                if simulated_board is None:
                    continue
                opponent_side = simulated_board.turn
                eval_score = self.minimax(simulated_board, opponent_side, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for init_pos, end_pos in moves:
                simulated_board = self.simulate_move(board, init_pos, end_pos)
                if simulated_board is None:
                    continue
                opponent_side = simulated_board.turn
                eval_score = self.minimax(simulated_board, opponent_side, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def _evaluate_move_task(self, move, board, side, depth):
        start_pos, end_pos = move
        simulated_board = self.simulate_move(board, start_pos, end_pos)
        if simulated_board is None:
            return (move, float('-inf'))
        opponent_side = simulated_board.turn
        score = self.minimax(simulated_board, opponent_side, depth - 1, float('-inf'), float('inf'), False)
        return (move, score)

    def get_best_move_at_depth(self, board, side, depth):
        best_moves = []
        best_value = float('-inf')
        moves = board.get_all_valid_moves(side)
        if not moves:
            return None, float('-inf')
        if len(moves) == 1:
            return moves[0], self.evaluate_board(self.simulate_move(board, *moves[0]), side)

        results = []
        num_workers = min(self.max_threads, len(moves))
        if num_workers <= 0:
            num_workers = 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._evaluate_move_task, move, board, side, depth): move for move in moves}
            for future in concurrent.futures.as_completed(futures):
                move = futures[future]
                try:
                    _move, move_value = future.result()
                    results.append((move, move_value))
                    if move_value > best_value:
                        best_value = move_value
                        best_moves = [move]
                    elif move_value == best_value:
                        best_moves.append(move)
                except Exception:
                    results.append((move, float('-inf')))

        selected_move = None
        if not best_moves and results:
            results.sort(key=lambda item: item[1], reverse=True)
            if results and results[0][1] > float('-inf'):
                top_score = results[0][1]
                best_moves = [m for m, score in results if score == top_score]

        if best_moves:
            selected_move = random.choice(best_moves)
        elif moves:
            selected_move = random.choice(moves)
            best_value = float('-inf')

        return selected_move, best_value

    def move(self, side, board):
        start_time = time.time()
        self.calculation_time = 0
        best_move_overall = None
        initial_moves = board.get_all_valid_moves(side)
        if not initial_moves:
            return None

        if board.is_in_checkmate('white') or board.is_in_checkmate('black') or board.is_in_draw():
            return None

        for depth in range(1, self.max_depth + 1):
            time_elapsed = time.time() - start_time
            time_remaining = self.time_limit - time_elapsed
            if time_remaining < (self.time_limit * 0.1) or time_elapsed > self.time_limit * 0.95:
                break

            try:
                current_best_move, current_best_score = self.get_best_move_at_depth(board, side, depth)
                if current_best_move is not None:
                    best_move_overall = current_best_move
                    self.best_move_from_last_iter = current_best_move
                if current_best_score == float('inf'):
                    break
            except Exception:
                break

            time_elapsed_after = time.time() - start_time
            if time_elapsed_after >= self.time_limit:
                break

        self.calculation_time = time.time() - start_time
        if best_move_overall is None:
            best_move_overall = random.choice(initial_moves) if initial_moves else None

        self.best_move_from_last_iter = None
        return best_move_overall
