import time
import random
import copy
import pygame
import concurrent.futures
import math
import os # To get CPU count

# --- Helper function (Ensure this is available or included) ---
def deepcopy_ignore_surfaces(obj, memo=None):
    """
    Performs a deep copy of an object, skipping Pygame surfaces.
    Crucial for copying game states without duplicating large graphical assets.
    """
    if memo is None:
        memo = {}

    if id(obj) in memo:
        return memo[id(obj)]

    # Skip Pygame surfaces
    if isinstance(obj, pygame.Surface):
        return obj # Return the original surface, don't copy

    # Handle dictionaries
    if isinstance(obj, dict):
        copied = {}
        memo[id(obj)] = copied
        for k, v in obj.items():
            copied[deepcopy_ignore_surfaces(k, memo)] = deepcopy_ignore_surfaces(v, memo)
        return copied

    # Handle lists
    elif isinstance(obj, list):
        copied = []
        memo[id(obj)] = copied
        # Recursively copy items
        for item in obj:
            copied.append(deepcopy_ignore_surfaces(item, memo))
        return copied

    # Handle objects with __dict__ (custom classes like Board, Piece, Square)
    elif hasattr(obj, '__dict__'):
        # Create a new instance without calling __init__
        try:
            copied = obj.__class__.__new__(obj.__class__)
            memo[id(obj)] = copied
            # Recursively copy attributes stored in __dict__
            for k, v in obj.__dict__.items():
                setattr(copied, k, deepcopy_ignore_surfaces(v, memo))
            return copied
        except TypeError as e:
            return obj # Fallback: return original (use with caution)

    # Handle other types using standard deepcopy (like tuples, basic types)
    else:
        try:
            return copy.deepcopy(obj, memo)
        except TypeError:
            # print(f"Warning: Could not deepcopy object of type {type(obj)}. Falling back.")
            return obj # Fallback: return original (use with caution)
# --- End Helper Function ---

SCORES_DICT = {
    " ": 1,   # pawn
    "N": 3,   # knight
    "B": 3,   # bishop
    "R": 5,   # rook
    "S": 5,   # star
    "Q": 9,   # queen
    "J": 9,   # joker
    "K": 1000 # king (Increased value for safety)
}

class Bot:
    """
    Minimax bot using Iterative Deepening, Alpha-Beta pruning, and
    multi-threading for the first level of moves within each depth.

    Attributes:
        max_depth (int): The maximum depth the iterative deepening will attempt.
        max_threads (int): Max threads for parallelization within each depth search.
        time_limit (float): Target time limit per move in seconds.
    """
    def __init__(self, max_depth=10, time_limit=0.095, SCORES_DICT=SCORES_DICT):
        """
        Initializes the bot.

        Args:
            max_depth (int): Max depth for iterative deepening. Defaults to 10.
            time_limit (float): Time limit per move. Defaults to 0.095s.
        """
        # self.depth is no longer fixed, iterative deepening controls it.
        self.max_depth = max_depth # Max depth iterative deepening will attempt
        self.time_limit = time_limit
        self.calculation_time = 0 # Stores time for the last move calculation

        # Determine max threads
        cpu_cores = os.cpu_count()
        self.max_threads = cpu_cores if cpu_cores else 4
        self.max_threads = min(self.max_threads, 8) # Cap threads
        
        # Internal state for iterative deepening (optional, e.g., for move ordering)
        self.best_move_from_last_iter = None
        self.SCORES_DICT = SCORES_DICT 

    
    # --- Core Search Logic (largely unchanged) ---

    def get_possible_moves(self, side, board):
        """Gets all valid moves for the given side."""
        return board.get_all_valid_moves(side)

    def evaluate_board(self, board, player_side):
        """
        Evaluates the board state based on material count.
        Checks terminal states first.
        """
        evaluation = 0
        opponent_side = 'black' if player_side == 'white' else 'white'

        # Check for terminal states first for potentially infinite scores
        if board.is_in_checkmate(opponent_side):
            return float('inf') # Current player wins
        if board.is_in_checkmate(player_side):
            return float('-inf') # Current player loses
        if board.is_in_draw():
            return 0 # Draw

        # Material counting logic
        board_state = board.get_board_state() # Gets the 6x6 array representation
        for y in range(len(board_state)):
            for x in range(len(board_state[y])):
                piece_code = board_state[y][x] # e.g., "wP", "bK", ""
                if piece_code != "":
                    piece_color_char = piece_code[0] # 'w' or 'b'
                    piece_type_char = piece_code[1]  # 'P', 'N', 'K', etc.
                    notation_key = piece_type_char if piece_type_char != 'P' else ' '
                    if piece_type_char == 'J': notation_key = 'J'
                    piece_value = self.SCORES_DICT.get(notation_key, 0)

                    if (piece_color_char == 'w' and player_side == 'white') or \
                       (piece_color_char == 'b' and player_side == 'black'):
                        evaluation += piece_value
                    else:
                        evaluation -= piece_value
        return evaluation

    def simulate_move(self, board, start_pos, end_pos):
        """Creates a deep copy (ignoring surfaces) and simulates a move."""
        try:
            new_board = deepcopy_ignore_surfaces(board)
            success = new_board.handle_move(start_pos, end_pos)
            if not success: return None
            return new_board
        except Exception as e:
            # print(f"Error in simulate_move: {e}")
            return None

    def minimax(self, board, side, depth, alpha, beta, maximizing_player):
        """ Minimax algorithm with Alpha-Beta pruning. (Unchanged) """
        is_terminal = board.is_in_checkmate('white') or \
                      board.is_in_checkmate('black') or \
                      board.is_in_draw()

        if depth == 0 or is_terminal:
            return self.evaluate_board(board, side)

        moves = board.get_all_valid_moves(side)
        if not moves: # Handle stalemate/checkmate
             return self.evaluate_board(board, side)

        # --- Add Move Ordering Here (Optional Enhancement) ---
        # If self.best_move_from_last_iter exists and is in moves, try it first.
        # Sort moves based on heuristics (captures > checks > quiet moves)
        # ----------------------------------------------------

        if maximizing_player:
            max_eval = float('-inf')
            for init_pos, end_pos in moves:
                simulated_board = self.simulate_move(board, init_pos, end_pos)
                if simulated_board is None: continue
                opponent_side = simulated_board.turn
                eval_score = self.minimax(simulated_board, opponent_side, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break # Beta cut-off
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
                if beta <= alpha:
                    break # Alpha cut-off
            return min_eval

    def _evaluate_move_task(self, move, board, side, depth):
        """ Helper task for threading: simulates move, calls minimax. (Unchanged) """
        start_pos, end_pos = move
        simulated_board = self.simulate_move(board, start_pos, end_pos)
        if simulated_board is None:
            return (move, float('-inf')) # Failed simulation
        opponent_side = simulated_board.turn
        score = self.minimax(simulated_board, opponent_side, depth - 1, float('-inf'), float('inf'), False)
        return (move, score)

    def _get_ordered_moves(self, board, side):
        """ Basic move ordering: captures first, then others. (Optional Enhancement) """
        moves = board.get_all_valid_moves(side)
        capture_moves = []
        other_moves = []
        for move in moves:
            start_pos, end_pos = move
            end_square = board.get_square_from_pos(end_pos)
            if end_square.occupying_piece is not None: # It's a capture
                capture_moves.append(move)
            else:
                other_moves.append(move)
        # Optionally prioritize the best move from the previous iteration
        # if self.best_move_from_last_iter in other_moves:
        #    other_moves.remove(self.best_move_from_last_iter)
        #    return capture_moves + [self.best_move_from_last_iter] + other_moves
        random.shuffle(other_moves) # Shuffle quiet moves
        return capture_moves + other_moves


    def get_best_move_at_depth(self, board, side, depth):
        """
        Finds the best move for a specific depth using threaded Minimax+AlphaBeta.
        (Formerly get_best_move_minimax_threaded).
        """
        best_moves = []
        best_value = float('-inf')
        # Use basic move ordering (optional but recommended)
        # moves = self._get_ordered_moves(board, side)
        moves = board.get_all_valid_moves(side) # Get unordered moves for now

        if not moves: return None, float('-inf') # Return move and score
        if len(moves) == 1: return moves[0], self.evaluate_board(self.simulate_move(board, *moves[0]), side) # Evaluate the single move

        results = []
        num_workers = min(self.max_threads, len(moves))
        if num_workers <= 0: num_workers = 1

        # --- Threaded Evaluation ---
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
                except Exception as exc:
                    # print(f'Move {move} generated exception: {exc}')
                    results.append((move, float('-inf')))

        # --- Final Selection for this Depth ---
        selected_move = None
        if not best_moves and results: # If initial tracking failed
            results.sort(key=lambda item: item[1], reverse=True)
            if results and results[0][1] > float('-inf'):
                top_score = results[0][1]
                best_moves = [m for m, score in results if score == top_score]

        if best_moves:
            selected_move = random.choice(best_moves)
        elif moves: # Fallback if all evaluations failed
            selected_move = random.choice(moves)
            best_value = float('-inf') # Indicate failure

        return selected_move, best_value # Return best move and its score for this depth


    # --- Main move function implementing Iterative Deepening ---

    def move(self, side, board):
        """
        Calculates the best move using Iterative Deepening.
        """
        start_time = time.time()
        self.calculation_time = 0
        best_move_overall = None
        # Get initial moves for fallback if time runs out early
        initial_moves = board.get_all_valid_moves(side)
        if not initial_moves: return None # No moves possible

        # Check if game already over
        is_terminal = board.is_in_checkmate('white') or \
                      board.is_in_checkmate('black') or \
                      board.is_in_draw()
        if is_terminal:
            return None

        # --- Iterative Deepening Loop ---
        for depth in range(1, self.max_depth + 1):
            time_elapsed = time.time() - start_time
            time_remaining = self.time_limit - time_elapsed

            # Basic time check: Stop if less than a small fraction of time remains
            # This is a simple heuristic and might stop too early or too late.
            # A better approach estimates time needed for the next depth.
            # For simplicity, let's stop if less than ~10-20% of time limit remains.
            # Or if elapsed time already exceeds limit significantly.
            if time_remaining < (self.time_limit * 0.1) or time_elapsed > self.time_limit * 0.95 :
                 # print(f"ID: Time limit approaching ({time_elapsed:.3f}s). Stopping at depth {depth-1}.")
                 break

            # print(f"ID: Starting search at depth {depth}...")
            try:
                current_best_move, current_best_score = self.get_best_move_at_depth(board, side, depth)

                # Store the best move found at this completed depth
                if current_best_move is not None:
                    best_move_overall = current_best_move
                    self.best_move_from_last_iter = current_best_move # Store for potential move ordering next iter
                    # print(f"ID: Depth {depth} completed. Best move: {best_move_overall} Score: {current_best_score:.2f}")

                # Optional: Check for checkmate score - can stop early if mate found
                if current_best_score == float('inf'):
                    # print(f"ID: Checkmate found at depth {depth}. Stopping early.")
                    break

            except Exception as e:
                # print(f"ID: Error during search at depth {depth}: {e}")
                # Stop iterative deepening if an error occurs
                break

            # Check time again *after* iteration completes
            time_elapsed_after = time.time() - start_time
            if time_elapsed_after >= self.time_limit:
                # print(f"ID: Iteration for depth {depth} completed, but time limit exceeded ({time_elapsed_after:.3f}s).")
                break # Stop even if iteration finished


        # --- End of Loop ---
        self.calculation_time = time.time() - start_time

        # If no iterations completed successfully, return a random move
        if best_move_overall is None:
            # print("ID: No depth completed in time or error occurred. Returning random move.")
            best_move_overall = random.choice(initial_moves) if initial_moves else None

        # print(f"Bot ({side}): Chose {best_move_overall}. Total Time: {self.calculation_time:.4f}s")
        if self.calculation_time > self.time_limit:
            print(f"Warning: Total time limit exceeded ({self.calculation_time:.4f}s > {self.time_limit}s)")
            pass

        self.best_move_from_last_iter = None # Reset for next turn
        return best_move_overall
