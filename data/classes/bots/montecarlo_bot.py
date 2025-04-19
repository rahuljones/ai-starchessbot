import random
import math
import time
import copy

# --- Assume the existence of the game's Board class ---
# It should have methods like:
# board.get_all_valid_moves(side) -> list of ((r1, c1), (r2, c2)) tuples
# board.make_move(move) -> Applies the move
# board_copy = copy.deepcopy(board) -> Creates a deep copy
# board.is_game_over() -> bool
# board.get_winner() -> 'white', 'black', or None (for draw/ongoing)
# board.get_current_player() -> 'white' or 'black'
# board.move_count -> number of moves made (for 50-move rule) - Check if available
# --------------------------------------------------------

class MCTSNode:
    """ Represents a node in the Monte Carlo Search Tree. """
    def __init__(self, board_state, side, parent=None, move=None):
        self.board_state = board_state  # The board state this node represents
        self.side = side                # The player whose turn it is in this state
        self.parent = parent            # Parent node
        self.move = move                # The move that led to this node from the parent

        self.children = []              # Child nodes
        # Handle potential case where board state might be terminal initially
        if not board_state.is_game_over():
            self.untried_moves = board_state.get_all_valid_moves(side) # Moves not yet explored
            random.shuffle(self.untried_moves) # Shuffle for randomness
        else:
            self.untried_moves = []

        self.wins = 0                   # Number of simulations resulting in a win for self.side
        self.visits = 0                 # Number of times this node was visited

    def uct_select_child(self, exploration_constant=math.sqrt(2)):
        """ Selects a child node using the UCT formula. """
        # Ensure all children have been visited at least once to avoid division by zero
        # and to prioritize exploring unvisited children if any exist during selection phase
        if any(child.visits == 0 for child in self.children):
             zero_visit_children = [c for c in self.children if c.visits == 0]
             if zero_visit_children:
                 return random.choice(zero_visit_children)

        # If all children visited at least once, or no unvisited children left, use UCT
        log_parent_visits = math.log(self.visits) if self.visits > 0 else 0 # Avoid log(0)
        selected_child = max(self.children,
                             key=lambda c: (c.wins / c.visits) +
                                           exploration_constant * math.sqrt(log_parent_visits / c.visits)
                                           if c.visits > 0 else float('inf')) # Prioritize unvisited if any slip through
        return selected_child

    def add_child(self, move, board_state, child_side):
        """ Adds a new child node for the given move and resulting board state. """
        node = MCTSNode(board_state=board_state, side=child_side, parent=self, move=move)
        if move in self.untried_moves: # Ensure move is valid before removing
            self.untried_moves.remove(move)
        self.children.append(node)
        return node

    def update(self, result):
        """ Updates the node's statistics based on the simulation result. """
        self.visits += 1
        # Result should be 1 if the player *whose turn it is at this node* won, 0.5 for draw, 0 for loss.
        self.wins += result


# Renamed class from MCTSBot to Bot
class Bot:
    """
    A bot that uses Monte Carlo Tree Search (MCTS) to choose the best move.
    It simulates games randomly to evaluate move possibilities.
    Warning: The 0.1 second time limit is very strict. Performance depends heavily
    on the efficiency of the Board class methods and the number of simulations possible.
    """
    def __init__(self, time_limit=0.095, exploration_constant=math.sqrt(2)):
        """
        Initialize the MCTS Bot.
        Args:
            time_limit (float): Time limit in seconds for making a move (slightly less than 0.1s).
            exploration_constant (float): The exploration constant (C) for UCT.
        """
        self.time_limit = time_limit
        self.exploration_constant = exploration_constant
        # Caching nodes is complex with changing board states, usually omitted unless proven necessary
        # self.nodes = {}

    def move(self, side, board):
        """
        Chooses the best move for the given side and board state using MCTS.
        """
        start_time = time.time()
        # Ensure the initial board isn't already game over
        if board.is_game_over():
            print("Bot: Game is already over.")
            return None # No move to make

        root_node = MCTSNode(board_state=copy.deepcopy(board), side=side)

        # If no moves available initially, return None or a failsafe
        if not root_node.untried_moves and not root_node.children:
            print("Bot: No valid moves available at the start!")
            # Failsafe: Maybe the board's method failed? Try again directly.
            moves = board.get_all_valid_moves(side)
            return random.choice(moves) if moves else None

        simulations_run = 0
        while time.time() - start_time < self.time_limit:
            node = root_node
            simulation_board = copy.deepcopy(root_node.board_state)

            # 1. Selection: Traverse the tree using UCT until a leaf node is reached
            while not node.untried_moves and node.children: # Node is fully expanded and non-terminal
                node = node.uct_select_child(self.exploration_constant)
                # Ensure move is valid before applying (should be, but defensive check)
                if node.move:
                    simulation_board.make_move(node.move)
                else:
                    # Should not happen if selection logic is correct
                    print("Bot Warning: Selected node has no associated move during Selection.")
                    break # Exit simulation loop if state is inconsistent

            # Check if selection led to an inconsistent state
            if node.move and simulation_board.is_game_over():
                # If the selected move ended the game, we don't need to expand/simulate further from here
                pass # Backpropagation will handle this terminal state


            # 2. Expansion: If the node isn't terminal and has untried moves, expand one
            elif node.untried_moves:
                move = node.untried_moves[0] # Get an untried move
                simulation_board.make_move(move)
                child_side = 'white' if node.side == 'black' else 'black'
                node = node.add_child(move, copy.deepcopy(simulation_board), child_side)

            # If node selection resulted in a terminal state before expansion,
            # the simulation phase starts from that terminal state.
            current_side = simulation_board.get_current_player() # Get side from the board state after selection/expansion

            # 3. Simulation (Rollout): Play randomly until the game ends
            move_count_start = simulation_board.move_count if hasattr(simulation_board, 'move_count') else 0
            moves_simulated = 0
            max_sim_moves = 100 # Limit simulation length (50 moves per side)

            while not simulation_board.is_game_over() and moves_simulated < max_sim_moves:
                possible_moves = simulation_board.get_all_valid_moves(current_side)
                if not possible_moves:
                    # Stalemate or unexpected end state
                    winner = None # Treat as draw
                    break

                # --- Fast Random Move Simulation ---
                random_move = random.choice(possible_moves)
                simulation_board.make_move(random_move)
                current_side = 'white' if current_side == 'black' else 'black'
                moves_simulated += 1
                # --- End Simulation ---
            else: # Loop finished (either game over or max moves reached)
                 if moves_simulated >= max_sim_moves:
                      winner = None # Draw by 50-move rule approximation
                 else:
                      winner = simulation_board.get_winner() # 'white', 'black', or None


            # 4. Backpropagation: Update nodes from the simulated node back to the root
            # Result is 1 for win for node.side, 0.5 for draw, 0 for loss for node.side
            temp_node = node # Start backpropagation from the node where simulation started
            while temp_node is not None:
                if winner is None:
                    result = 0.5
                elif winner == temp_node.side: # The player whose turn it was at this node won
                    result = 1.0
                else: # The player whose turn it was at this node lost
                    result = 0.0
                temp_node.update(result)
                temp_node = temp_node.parent

            simulations_run += 1
            # End of MCTS loop (time check)

        # print(f"Bot ({side}): Ran {simulations_run} simulations in {time.time() - start_time:.4f} seconds.")

        # After time limit, choose the best move from the root's children
        if not root_node.children:
             print("Bot Warning: No children nodes generated after simulations. Choosing random move.")
             # Fallback if simulation loop never ran or expanded
             moves = board.get_all_valid_moves(side)
             return random.choice(moves) if moves else None

        # Choose the move leading to the most visited child node (robustness)
        best_child = max(root_node.children, key=lambda c: c.visits)

        # Sanity check: ensure the best move is actually a move
        if best_child.move is None:
            print("Bot Error: Best child node has no associated move. Selecting random.")
            # This indicates a potential logic error, fallback to random from available children moves
            valid_child_moves = [c.move for c in root_node.children if c.move is not None]
            return random.choice(valid_child_moves) if valid_child_moves else None


        return best_child.move


# --- Example Usage (requires a Board class) ---
# Assuming you have a 'Board' class defined elsewhere according
# to the ACM Chess rules and the required interface.

# if __name__ == '__main__':
#     # Example: Create a board and bot
#     current_board = Board() # Initialize your 6x6 board
#     bot_player = Bot(time_limit=0.095) # Instantiate the Bot
#
#     # Get the bot's move (e.g., for white)
#     current_player_side = 'white' # Or get from board.get_current_player()
#     chosen_move = bot_player.move(current_player_side, current_board) # Call move on the bot instance
#
#     if chosen_move:
#         print(f"Bot chose move: {chosen_move}")
#         # Apply the move to the board
#         # current_board.make_move(chosen_move)
#     else:
#         print("Bot could not find a move.")