import copy
import random
import concurrent.futures
import os # Needed for os.cpu_count()

class Bot:
    """
    This is a sample minimax bot that uses the minimax algorithm to choose the best move.
    It evaluates the board state and simulates moves to find the optimal one.
    This is a basic implementation and may not be optimal for all scenarios.
    You are responsible for testing and improving the bot's performance. We recommend using depth 1 at first.
    We also recommend using a more advanced evaluation function for better performance.
    Warning: we have set a hard time limit of 0.1 second for the bot to make a move. If your bot takes
    longer than that, it will be terminated and our evaluation server will choose random moves. We have tested
    the bot with different depths and it nearly always exceeds the time limit if the depth is greater than 2.

    Multithreaded/Multiprocessed Notes:
    - Uses concurrent.futures.ProcessPoolExecutor to parallelize the evaluation of top-level moves.
    - minimax and evaluate_board are now static methods to be easily callable by child processes.
    - Assumes the board object returned by board.copy_logical_state() is pickleable.
    - Overhead of process creation means this might only be faster for depth >= 2 and sufficient moves.
    """
    SCORES_DICT = {
        " ": 1, # pawn
        "N": 3, # knight
        "B": 3, # bishop
        "R": 5, # rook
        "S": 5, # star
        "Q": 9, # queen
        "J": 9, # joker
        "K": 10000 # king
    }

    def __init__(self):
        # Set depth carefully. Multiprocessing adds overhead.
        # Depth 2 might be feasible now, but test against the 0.1s limit.
        self.depth = 2 # Example: increased depth, test performance!

    def get_possible_moves(self, side, board):
        return board.get_all_valid_moves(side)

    @staticmethod
    def evaluate_board(side, board):
        """Static method to evaluate the board state."""
        evaluation = 0
        board_state = board.get_board_state()
        for x in board_state:
            for y in x:
                if y != "":
                    piece = y
                    # Use Bot.SCORES_DICT as it's now a class variable
                    piece_value = Bot.SCORES_DICT[piece[1]]
                    if piece[0] == 'b' and side == 'black':
                        evaluation += piece_value
                    elif piece[0] == 'w' and side == 'white':
                        evaluation += piece_value
                    else:
                        evaluation -= piece_value
        return evaluation

    def simulate_move(self, board, start_pos, end_pos):
        """Simulates a move and returns the new board state. (Instance method is fine here)"""
        # Use copy_logical_state() which is likely more efficient and potentially pickle-friendly
        new_board = board.copy_logical_state()

        # handle_move should ideally not fail if the move came from get_all_valid_moves,
        # but check just in case. handle_move modifies new_board in place.
        success = new_board.handle_move(start_pos, end_pos)
        if not success:
             # This scenario might indicate a bug in move generation or handle_move
             print(f"WARNING: Simulated move failed! {start_pos} -> {end_pos} on board state:")
             print(board.get_board_state()) # Print state *before* failed move
             # Returning the original state copy might be safest here to avoid errors down the line
             return board.copy_logical_state()
        return new_board

    @staticmethod
    def minimax(board, side, depth, maximizing_player):
        """Static minimax method for use in multiprocessing."""
        if depth == 0 or board.is_in_checkmate(side):
            # Call static evaluate_board
            return Bot.evaluate_board(side, board)

        # Note: We need a way to simulate moves within the static method context.
        # Assuming board objects have the necessary methods directly.
        moves = board.get_all_valid_moves(side)

        # --- Helper function to simulate move within static context ---
        # This assumes board.copy_logical_state and board.handle_move work correctly
        # on the board object passed to this static method.
        def _static_simulate_move(current_board, start_pos, end_pos):
            new_board_state = current_board.copy_logical_state()
            success = new_board_state.handle_move(start_pos, end_pos)
            if not success:
                # Handle error - perhaps return None or raise an exception
                # For simplicity here, we might just rely on valid moves
                # but in production, robust handling is needed.
                print(f"Static Sim Error: {start_pos}->{end_pos}")
                return current_board # Return original board on failure
            return new_board_state
        # --- End Helper ---


        if maximizing_player:
            max_eval = float('-inf')
            for init_pos, end_pos in moves:
                # Use the helper for simulation
                simulated_board = _static_simulate_move(board, init_pos, end_pos)
                # Recursive call to static minimax
                eval_score = Bot.minimax(simulated_board, side, depth - 1, False)
                max_eval = max(max_eval, eval_score)
            return max_eval
        else: # Minimizing player
            min_eval = float('inf')
            for init_pos, end_pos in moves:
                 # Use the helper for simulation
                simulated_board = _static_simulate_move(board, init_pos, end_pos)
                 # Recursive call to static minimax
                eval_score = Bot.minimax(simulated_board, side, depth - 1, True)
                min_eval = min(min_eval, eval_score)
            return min_eval

    def get_best_move_minimax(self, board, side, depth):
        """Finds the best move using minimax, parallelizing the first level."""
        if depth == 0: # Should not happen if self.depth > 0, but safe check
             moves = self.get_possible_moves(side, board)
             return random.choice(moves) if moves else None

        best_move_candidates = []
        best_value = float('-inf')
        moves = self.get_possible_moves(side, board)

        # Use ProcessPoolExecutor for CPU-bound tasks
        # Defaults to os.cpu_count() workers, adjust max_workers if needed
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures_map = {}
            for init_pos, end_pos in moves:
                # Simulate the move in the main process
                simulated_board = self.simulate_move(board, init_pos, end_pos)

                # Check if the simulation produced a valid board before submitting
                if simulated_board:
                    # Submit the static minimax function to the executor
                    # It will run minimax for the next level (depth-1) as the minimizing player (False)
                    future = executor.submit(Bot.minimax, simulated_board, side, depth - 1, False)
                    futures_map[(init_pos, end_pos)] = future
                else:
                    # Handle simulation failure - perhaps assign a very bad score
                    print(f"Skipping evaluation for failed simulation: {init_pos} -> {end_pos}")


            # Collect results as they complete
            move_results = {}
            for move, future in futures_map.items():
                try:
                    # Get the evaluation result from the future
                    move_results[move] = future.result()
                except Exception as e:
                    print(f"Error getting result for move {move}: {e}")
                    # Assign a very bad score if a process fails
                    move_results[move] = float('-inf')

        # Find the best move(s) based on the collected results
        for move, move_value in move_results.items():
            if move_value > best_value:
                best_value = move_value
                best_move_candidates = [move]
            elif move_value == best_value:
                best_move_candidates.append(move)

        # Choose the best move (randomly among equals)
        return random.choice(best_move_candidates) if best_move_candidates else (None, None) # Return None if no valid moves found/evaluated


    def move(self, side, board):
        """Selects the best move using the parallelized minimax."""
        # Make sure depth is at least 1 for the parallel logic to work well
        if self.depth < 1:
             print("Warning: Depth is less than 1, minimax requires at least depth 1.")
             # Fallback or default behavior if depth is 0
             moves = self.get_possible_moves(side, board)
             return random.choice(moves) if moves else (None, None)

        best_move = self.get_best_move_minimax(board, side, self.depth)
        return best_move