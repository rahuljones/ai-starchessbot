import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from data.classes.Board import Board

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class ACMChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(4)])

        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 6 * 6, 1296)

        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(6 * 6, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res_blocks(x)

        p = F.relu(self.policy_conv(x)).view(x.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)

        v = F.relu(self.value_conv(x)).view(x.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

class ACMChessGame:
    def __init__(self, board_obj=None):
        self.board = board_obj or Board(600, 600)
        self.current_player = 'w'

    def reset(self):
        self.board.reset()
        self.current_player = 'w'

    def get_state(self):
        return board_to_tensor(self.board.get_board_state())

    def get_legal_moves(self):
        return self.board.get_all_valid_moves(self.board.turn)

    def make_move(self, move):
        start, end = move
        success = self.board.handle_move(start, end)
        if success:
            self.current_player = 'b' if self.current_player == 'w' else 'w'
        return success

    def is_terminal(self):
        board = self.board.get_board_state()
        white_king, black_king = False, False
        for row in board:
            for piece in row:
                if piece == 'wK': white_king = True
                if piece == 'bK': black_king = True
        if not white_king: return True, -1 if self.current_player == 'w' else 1
        if not black_king: return True, -1 if self.current_player == 'b' else 1
        if self.board.is_in_draw(): return True, 0
        return False, 0

    def get_current_player(self):
        return self.current_player

    def clone(self):
        new_game = ACMChessGame(self.board.copy_logical_state())
        new_game.current_player = self.current_player
        return new_game

# ========== UTILS ==========
def board_to_tensor(board_state):
    tensor = np.zeros((16, 6, 6), dtype=np.float32)
    piece_map = {'P': 0, 'R': 1, 'N': 2, 'B': 3, 'Q': 4, 'K': 5, 'S': 6, 'J': 7, ' ': 0}
    for i in range(6):
        for j in range(6):
            val = board_state[i][j]
            if val and len(val) == 2:
                color = val[0]
                piece = val[1]
                offset = 0 if color == 'w' else 8
                if piece in piece_map:
                    idx = piece_map[piece] + offset
                    tensor[idx][i][j] = 1
                else:
                    print(f"Unknown piece type: {val} at ({i},{j})")
    return tensor

def move_to_index(move):
    (x1, y1), (x2, y2) = move
    return x1 * 216 + y1 * 36 + x2 * 6 + y2

def index_to_move(index):
    x1, rem = divmod(index, 216)
    y1, rem = divmod(rem, 36)
    x2, y2 = divmod(rem, 6)
    return ((x1, y1), (x2, y2))

# ========== MCTS ==========
class MCTS:
    def __init__(self, game_cls, model, simulations=100, cpuct=1.0):
        self.model = model
        self.simulations = simulations
        self.cpuct = cpuct
        self.game_cls = game_cls
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}

    def get_action_probs(self, game, temp=1):
        for _ in range(self.simulations):
            self.search(game.clone(), depth=0)

        s = self.serialize(game.get_state())
        counts = [self.Nsa.get((s, a), 0) for a in range(1296)]

        if temp == 0:
            best = np.argmax(counts)
            probs = np.zeros(1296)
            probs[best] = 1
            return probs
        counts = np.array(counts)
        return counts ** (1. / temp) / np.sum(counts ** (1. / temp))

    def search(self, game, depth=0):
        if depth > 200:
            return 0

        s = self.serialize(game.get_state())

        if s not in self.Es:
            terminal, reward = game.is_terminal()
            self.Es[s] = terminal
            if terminal:
                return -reward

        if s not in self.Ps:
            state_tensor = torch.tensor(game.get_state()).unsqueeze(0)
            with torch.no_grad():
                policy, value = self.model(state_tensor)
            policy = policy.squeeze().cpu().numpy()

            valid_moves = game.get_legal_moves()
            mask = np.zeros(1296)
            for move in valid_moves:
                mask[move_to_index(move)] = 1
            policy *= mask
            policy /= np.sum(policy)

            self.Ps[s] = policy
            self.Vs[s] = mask
            self.Ns[s] = 0
            return -value.item()

        valid_moves = self.Vs[s]
        best_u, best_a = -float('inf'), -1
        for a in np.nonzero(valid_moves)[0]:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)
            if u > best_u:
                best_u, best_a = u, a

        a = best_a
        move = index_to_move(a)
        if not game.make_move(move):
            return 0
        v = self.search(game, depth + 1)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        self.Ns[s] += 1
        return -v

    def serialize(self, tensor_state):
        return tensor_state.tobytes()

# ========== SELF-PLAY & TRAINING ==========
def play_game(game_cls, model, mcts_simulations=100):
    game = game_cls()
    memory = []
    mcts = MCTS(game_cls, model, simulations=mcts_simulations)
    while True:
        state_tensor = game.get_state()
        action_probs = mcts.get_action_probs(game, temp=1)
        memory.append((state_tensor, action_probs, game.get_current_player()))

        move = np.random.choice(1296, p=action_probs)
        if not game.make_move(index_to_move(move)):
            continue
        done, result = game.is_terminal()
        if done:
            break
    return [(s, p, result if pl == game.get_current_player() else -result) for (s, p, pl) in memory]

def train_model(model, memory, optimizer, batch_size=32, epochs=5):
    model.train()
    for epoch in range(epochs):
        np.random.shuffle(memory)
        for i in range(0, len(memory), batch_size):
            batch = memory[i:i+batch_size]
            states, policies, values = zip(*batch)
            states = torch.tensor(states, dtype=torch.float32)
            policies = torch.tensor(policies, dtype=torch.float32)
            values = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

            pred_policies, pred_values = model(states)
            v_loss = F.mse_loss(pred_values, values)
            p_loss = -torch.mean(torch.sum(policies * pred_policies, dim=1))
            loss = v_loss + p_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def run_training_pipeline(game_cls, model, optimizer, num_iterations=100, games_per_iteration=20):
    game_counter = 0
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        memory = []
        for g in range(games_per_iteration):
            game_counter += 1
            print(f"  Game {g + 1}/{games_per_iteration} (Global Game #{game_counter})")
            memory += play_game(game_cls, model)
        train_model(model, memory, optimizer)
        torch.save(model.state_dict(), f"acm_model_iter{iteration+1}.pt")
        print("  Model saved.")

if __name__ == "__main__":
    model = ACMChessNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def game_cls():
        return ACMChessGame(Board(600, 600))

    run_training_pipeline(
        game_cls=game_cls,
        model=model,
        optimizer=optimizer,
        num_iterations=4,
        games_per_iteration=4
    )