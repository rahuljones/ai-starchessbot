class Piece:
    def __init__(self, pos, color, board):
        self.pos = pos
        self.x = pos[0]
        self.y = pos[1]
        self.color = color
        self.has_moved = False
        self.has_promoted = False
        self.notation = None
    def get_notation(self):
        return self.notation
    def get_moves(self, board):
        output = []
        for direction in self.get_possible_moves(board):
            for square in direction:
                if square.occupying_piece is not None:
                    if square.occupying_piece.color == self.color:
                        break
                    else:
                        output.append(square)
                        break
                else:
                    output.append(square)
        return output

    def get_valid_moves(self, board):
        output = []
        for square in self.get_moves(board):
            output.append(square)
        return output

    def move(self, board, square, force=False):
        for i in board.squares:
            i.highlight = False
        if square in self.get_valid_moves(board) or force:
            prev_square = board.get_square_from_pos(self.pos)
            self.pos, self.x, self.y = square.pos, square.x, square.y
            if(self.get_notation() == ' ' and self.y == 0 and self.color == "white") or (self.get_notation() == ' ' and self.y == 5 and self.color == "black") and not self.has_promoted:
                self.promote(self.color, board)
                #print("Pawn has been promoted")

            prev_square.occupying_piece = None
            if square.occupying_piece is not None:
                board.last_captured = 0
            else:
                board.last_captured += 1
            square.occupying_piece = self
            board.selected_piece = None
            self.has_moved = True

            return True
        else:
            board.selected_piece = None
            return False

    # True for all pieces except pawn
    def attacking_squares(self, board):
        return self.get_moves(board)
