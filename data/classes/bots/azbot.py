import torch
import numpy as np
from acmchess_alphazero import ACMChessNet, ACMChessGame, MCTS, index_to_move
from data.classes.Board import Board

class AlphaZeroBot:
    def __init__(self, model_path, simulations=50):
        self.model = ACMChessNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.simulations = simulations

    def move(self, side, board):
        game = ACMChessGame(board)
        mcts = MCTS(lambda: game.clone(), self.model, simulations=self.simulations)
        probs = mcts.get_action_probs(game, temp=0)
        best_index = np.argmax(probs)
        return index_to_move(best_index)