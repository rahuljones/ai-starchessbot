import pygame

class Bot:
    def __init__(self, side, board):
        self.side = side
        self.board = board
    
    def get_possible_moves(self):
        '''
        Get all possible moves for the bot
        '''
        pass

    def evaluate_move(self):
        '''
        Evaluate each move for the bot
        '''
        pass

    def move(self):
        '''API for moving the bot. Should be called by Board class'''
        pass