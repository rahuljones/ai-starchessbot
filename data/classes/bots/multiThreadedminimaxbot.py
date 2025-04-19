import time
import random
import copy
import pygame
import concurrent.futures
import math
import os 

def deepcopy_ignore_surfaces(obj, memo=None):
    """
    Performs a deep copy of an object, skipping Pygame surfaces.
    Crucial for copying game states without duplicating large graphical assets.
    """
    if memo is None:
        memo = {}

    if id(obj) in memo:
        return memo[id(obj)]

    if isinstance(obj, pygame.Surface):
        return obj # Return the original surface, don't copy

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
        for item in obj:
            copied.append(deepcopy_ignore_surfaces(item, memo))
        return copied

    elif hasattr(obj, '__dict__'):
        copied = obj.__class__.__new__(obj.__class__)
        memo[id(obj)] = copied
        for k, v in obj.__dict__.items():
            setattr(copied, k, deepcopy_ignore_surfaces(v, memo))
        return copied

    else:
        return copy.deepcopy(obj, memo)

class Bot:
    """
    Minimax bot modified to use Alpha-Beta pruning and multi-threading
    for the first level of moves, aiming for minimal changes to the original structure.

    Attributes:
        depth (int): Search depth. WARNING: Depths > 2 likely exceed 0.1s limit.
        max_threads (int): Max threads for parallelization.
        time_limit (float): Target time limit (for reporting).
    """
    def __init__(self, depth=2, time_limit=0.095):
        """
        Initializes the bot.

        Args:
            depth (int): Search depth. Defaults to 2.
            time_limit (float): Approximate time limit.
        """
        self.depth = depth ## Please set the depth <= 2 unless you are sure your bot runs within the time limit.
        self.time_limit = time_limit
        self.calculation_time = 0
        self.transposition_table = {}

        # Determine max threads, similar to the previous version
        cpu_cores = os.cpu_count()
        self.max_threads = cpu_cores if cpu_cores else 4
        self.max_threads = min(self.max_threads, 8) # Cap threads

        # Piece scores (kept from original)
        self.SCORES_DICT = {
            " ": 1,   # pawn
            "N": 3,   # knight
            "B": 3,   # bishop
            "R": 5,   # rook
            "S": 5,   # star
            "Q": 9,   # queen
            "J": 9,   # joker
            "K": 1000 # king (Increased value for safety)
        }

    # --- Functions largely unchanged from original minimax_bot.py ---

    def get_possible_moves(self, side, board):
        """Gets all valid moves for the given side."""
        return board.get_all_valid_moves(side)

    def evaluate_board(self, board, player_side):
        """
        Evaluates the board state based on material count.
        Modified slightly to check terminal states first.
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

        # Original material counting logic
        board_state = board.get_board_state() # Gets the 6x6 array representation
        for y in range(len(board_state)):
            for x in range(len(board_state[y])):
                piece_code = board_state[y][x] # e.g., "wP", "bK", ""
                if piece_code != "":
                    piece_color_char = piece_code[0] # 'w' or 'b'
                    piece_type_char = piece_code[1]  # 'P', 'N', 'K', etc. (original used ' ' for pawn)

                    # Adjust notation if needed (original used ' ' for pawn notation)
                    notation_key = piece_type_char if piece_type_char != 'P' else ' '
                    # Handle Joker ('J') if pawn promotion implemented correctly in Board
                    if piece_type_char == 'J': notation_key = 'J'


                    piece_value = self.SCORES_DICT.get(notation_key, 0) # Use .get for safety

                    # Add score if the piece belongs to the player_side, subtract otherwise
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
            # handle_move changes the turn internally in the board object
            if not success: return None
            return new_board
        except Exception as e:
            # print(f"Error in simulate_move: {e}")
            return None

    # --- Minimax function modified for Alpha-Beta ---

    def minimax(self, board, side, depth, alpha, beta, maximizing_player):
        """
        Minimax algorithm with Alpha-Beta pruning.

        Args:
            board: Current board state.
            side: The player whose turn it is on this board.
            depth: Remaining search depth.
            alpha: Alpha value for pruning.
            beta: Beta value for pruning.
            maximizing_player: Boolean, True if current node is maximizing.

        Returns:
            Evaluation score for the board state.
        """

        # Check if terminal state or depth limit reached
        is_terminal = board.is_in_checkmate('white') or \
                      board.is_in_checkmate('black') or \
                      board.is_in_draw()

        if depth == 0 or is_terminal:
            # Evaluate from the perspective of the player whose turn it is *at this node*
            return self.evaluate_board(board, side)

        moves = board.get_all_valid_moves(side)
        if not moves: # Handle case where no moves are possible (stalemate/checkmate)
             return self.evaluate_board(board, side)
        board_key = self.get_board_hash(board)

        # Transposition table lookup
        if board_key in self.transposition_table:
            cached_depth, cached_score = self.transposition_table[board_key]
            if cached_depth >= depth:
                return cached_score

        if maximizing_player:
            max_eval = float('-inf')
            for init_pos, end_pos in moves:
                simulated_board = self.simulate_move(board, init_pos, end_pos)
                if simulated_board is None: continue # Skip if simulation failed

                opponent_side = simulated_board.turn # Get the side whose turn it is now
                eval_score = self.minimax(simulated_board, opponent_side, depth - 1, alpha, beta, False) # Opponent minimizes
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score) # Update alpha
                if beta <= alpha:
                    break # Beta cut-off
            self.transposition_table[board_key] = (depth, max_eval)
            return max_eval
        else: # Minimizing player
            min_eval = float('inf')
            for init_pos, end_pos in moves:
                simulated_board = self.simulate_move(board, init_pos, end_pos)
                if simulated_board is None: continue # Skip if simulation failed

                opponent_side = simulated_board.turn # Get the side whose turn it is now
                eval_score = self.minimax(simulated_board, opponent_side, depth - 1, alpha, beta, True) # Opponent maximizes
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score) # Update beta
                if beta <= alpha:
                    break # Alpha cut-off
            self.transposition_table[board_key] = (depth, min_eval)
            return min_eval


    # --- Function to orchestrate threaded evaluation ---
    def _evaluate_move_task(self, move, board, side, depth):
        """Helper task for threading: simulates move, calls minimax."""
        start_pos, end_pos = move
        simulated_board = self.simulate_move(board, start_pos, end_pos)
        if simulated_board is None:
            return (move, float('-inf')) # Failed simulation

        # After the first move, it's opponent's turn (minimizing). Call minimax.
        opponent_side = simulated_board.turn
        # Initial alpha/beta for the recursive call
        score = self.minimax(simulated_board, opponent_side, depth - 1, float('-inf'), float('inf'), False) # Opponent is minimizing player

        return (move, score)

    def get_best_move_minimax_threaded(self, board, side, depth):
        """
        Finds the best move using Minimax+AlphaBeta, parallelizing the first level.
        (Replaces original get_best_move_minimax).
        """
        best_moves = [] # Store potentially multiple best moves
        best_value = float('-inf')
        moves = board.get_all_valid_moves(side)

        if not moves: return None
        if len(moves) == 1: return moves[0] # Only one choice

        results = []
        num_workers = min(self.max_threads, len(moves))
        if num_workers <= 0: num_workers = 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._evaluate_move_task, move, board, side, depth): move for move in moves}

            for future in concurrent.futures.as_completed(futures):
                move = futures[future]
                try:
                    _move, move_value = future.result()
                    results.append((move, move_value)) # Store result

                    # Update best score and move(s) found so far
                    if move_value > best_value:
                        best_value = move_value
                        best_moves = [move] # New best move found
                    elif move_value == best_value:
                        best_moves.append(move) # Add to list of equally good moves

                except Exception as exc:
                    print(f'Move {move} generated exception: {exc}')
                    results.append((move, float('-inf'))) # Penalize errors

        # Select the best move from results if the initial tracking failed or for tie-breaking
        if not best_moves and results:
            # Sort results and find best score again just in case
            results.sort(key=lambda item: item[1], reverse=True)
            if results and results[0][1] > float('-inf'):
                top_score = results[0][1]
                best_moves = [m for m, score in results if score == top_score]

        # Final selection: Choose randomly among the best moves found
        if best_moves:
            return random.choice(best_moves)
        elif moves: # Fallback if all evaluations failed
            # print("Warning: All evaluations failed or returned -inf. Choosing random move.")
            return random.choice(moves)
        else:
            return None # No moves possible

    def move(self, side, board):
        """Calculates the best move using threaded Minimax+AlphaBeta."""
        self.transposition_table.clear()
        start_time = time.time()

        best_move = self.get_best_move_minimax_threaded(board, side, self.depth)

        self.calculation_time = time.time() - start_time
        print(f"Bot ({side}): Chose {best_move}. Time: {self.calculation_time:.4f}s")

        if self.calculation_time > self.time_limit:
            print(f"Warning: Time limit exceeded ({self.calculation_time:.4f}s > {self.time_limit}s)")
            pass

        if best_move is None:
            possible_moves = board.get_all_valid_moves(side)
            if possible_moves:
                return random.choice(possible_moves)
            else:
                return None 

        return best_move
    def get_board_hash(self, board):
        board_tuple = tuple(tuple(row) for row in board.get_board_state())
        return (board_tuple, board.turn)
