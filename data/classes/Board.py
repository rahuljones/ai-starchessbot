from data.classes.Square import Square
from data.classes.pieces.Rook import Rook
from data.classes.pieces.Bishop import Bishop
from data.classes.pieces.Knight import Knight
from data.classes.pieces.Queen import Queen
from data.classes.pieces.King import King
from data.classes.pieces.Pawn import Pawn
from data.classes.pieces.Star import Star

# Game state checker
class Board:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.tile_width = width // 6
        self.tile_height = height // 6
        self.selected_piece = None
        self.turn = "white"
        self.config = [
            ["bR", "bN", "bQ", "bK", "bB", "bS"],
            ["bP", "bP", "bP", "bP", "bP", "bP"],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["wP", "wP", "wP", "wP", "wP", "wP"],
            ["wR", "wN", "wQ", "wK", "wB", "wS"],
        ]
        self.squares = self.generate_squares()
        self.last_captured = 0
        self.num_moves = 0
        self.setup_board()

    def generate_squares(self):
        output = []
        for y in range(6):
            for x in range(6):
                output.append(Square(x, y, self.tile_width, self.tile_height))
        return output

    def get_square_from_pos(self, pos):
        for square in self.squares:
            if (square.x, square.y) == (pos[0], pos[1]):
                return square

    def get_piece_from_pos(self, pos):
        return self.get_square_from_pos(pos).occupying_piece

    def setup_board(self):
        for y, row in enumerate(self.config):
            for x, piece in enumerate(row):
                if piece != "":
                    square = self.get_square_from_pos((x, y))
                    # looking inside contents, what piece does it have
                    if piece[1] == "R":
                        square.occupying_piece = Rook(
                            (x, y), "white" if piece[0] == "w" else "black", self
                        )
                    # as you notice above, we put `self` as argument, or means our class Board
                    elif piece[1] == "N":
                        square.occupying_piece = Knight(
                            (x, y), "white" if piece[0] == "w" else "black", self
                        )
                    elif piece[1] == "B":
                        square.occupying_piece = Bishop(
                            (x, y), "white" if piece[0] == "w" else "black", self
                        )
                    elif piece[1] == "Q":
                        square.occupying_piece = Queen(
                            (x, y), "white" if piece[0] == "w" else "black", self
                        )
                    elif piece[1] == "K":
                        square.occupying_piece = King(
                            (x, y), "white" if piece[0] == "w" else "black", self
                        )
                    elif piece[1] == "P":
                        square.occupying_piece = Pawn(
                            (x, y), "white" if piece[0] == "w" else "black", self
                        )
                    elif piece[1] == "S":
                        square.occupying_piece = Star(
                            (x, y), "white" if piece[0] == "w" else "black", self
                        )


    def is_in_checkmate(self, color):
        output = False
        pieces_left = [
            i.occupying_piece.color + i.occupying_piece.notation
            for i in self.squares
            if i.occupying_piece is not None
        ]
        return color + "K" not in pieces_left

    def is_in_check(self, color):
        return False

    def handle_click(self, mx, my):
        x = mx // self.tile_width
        y = my // self.tile_height
        clicked_square = self.get_square_from_pos((x, y))
        if self.selected_piece is None:
            if clicked_square.occupying_piece is not None:
                if clicked_square.occupying_piece.color == self.turn:
                    self.selected_piece = clicked_square.occupying_piece
        # successfully made a move
        elif self.selected_piece.move(self, clicked_square):
            self.num_moves += 1
            self.turn = "white" if self.turn == "black" else "black"
            print(self.get_board_state())

        elif clicked_square.occupying_piece is not None:
            if clicked_square.occupying_piece.color == self.turn:
                self.selected_piece = clicked_square.occupying_piece

    def draw(self, display):
        if self.selected_piece is not None:
            self.get_square_from_pos(self.selected_piece.pos).highlight = True
            for square in self.selected_piece.get_valid_moves(self):
                square.highlight = True
        for square in self.squares:
            square.draw(display)

    def get_board_state(self):
        # 2d 6x6 array
        output = [["" for _ in range(6)] for _ in range(6)]

        for square in self.squares:
            if square.occupying_piece is not None:
                output[square.y][square.x] = (
                    square.occupying_piece.color[0] + square.occupying_piece.notation
                )
            else:
                output[square.y][square.x] = ""
        return output

    def handle_move(self, start_pos, end_pos):
        start_square = self.get_square_from_pos(start_pos)
        end_square = self.get_square_from_pos(end_pos)

        if start_square.occupying_piece is None:
            return False

        if start_square.occupying_piece.color != self.turn:
            return False

        if start_square.occupying_piece.move(self, end_square):
            self.turn = "white" if self.turn == "black" else "black"
            print(self.get_board_state())
            self.num_moves += 1
            return True

    def alg_not_to_pos(self, alg_not):
        return (ord(alg_not[0]) - 65, int(alg_not[1]) - 1)

    def get_all_valid_moves(self, color):
        output = []
        for square in self.squares:
            if (
                square.occupying_piece is not None
                and square.occupying_piece.color == color
            ):
                for move in square.occupying_piece.get_valid_moves(self):
                    output.append((square.pos, move.pos))
        return output
    def is_in_draw(self):
        return self.num_moves >= 100