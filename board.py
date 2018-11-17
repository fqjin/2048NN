import numpy as np
from random import randint, randrange
# Do not use scientific notation
np.set_printoptions(suppress=True)


# Board Dimensions
SIZE = 4
DIMENSIONS = (SIZE, SIZE)  # (4,4)
SIZE_SQRD = SIZE*SIZE  # 16


class Board:
    """Board object stores 2048 board state
    Numbers are stored as their log-base-2

    Args:
        gen (bool): Whether to generate two initial tiles.
            Defaults to True.

    Attributes:
        board: numpy array of board tiles, stored as log-base-2
        score: int score, the sum of all combination values
        moves: a list of the 4 move functions in order L,U,R,D

    """

    def __init__(self, gen=True):
        self.board = np.zeros(DIMENSIONS)
        self.score = 0
        # self.moves = [self.move_left,
        #               self.move_up,
        #               self.move_right,
        #               self.move_down]

        if gen:
            self.generate_tile()
            self.generate_tile()
            self.draw()

    def generate_tile(self):
        """Places a 2 or 4 in a random empty tile
        Unhandled error if board is full
        Chance of 2 is 90%
        """
        empty = np.transpose(np.where(self.board == 0))
        position = empty[randrange(len(empty))]
        if randint(0, 9):
            self.board[position[0], position[1]] = 1
        else:
            self.board[position[0], position[1]] = 0
        #   self.board[tuple(position)] is 3 times slower

    def draw(self):
        """Prints board state"""
        print(str(2**self.board).replace('1.', ' .', SIZE_SQRD))
        print(' Score : {}'.format(self.score))

    def check_full(self):
        """Checks if board is full and has no empty tiles"""
        # TODO: Do I actually need Board.check_full() function?
        return np.count_nonzero(self.board) == SIZE_SQRD

    def restore(self, board, score):
        """Sets board and score state to the input values"""
        self.board = np.copy(board)  # need to copy
        self.score = score  # immutable does not need copying

    def copy(self):
        """Returns a copy as a new Board object"""
        temp = Board()
        temp.board = np.copy(self.board)
        temp.score = self.score
        # Explicitly saves time rather than calling restore
        return temp

    def merge_row(self, row):
        """Merges input row and shifts tiles to the left side
        Score is updated if any new tiles are made

        Args:
            row: numpy array of row to merge

        """
        final = []
        base = 0
        for tile in row:
            if tile == 0:
                continue  # Skips zeros
            if base == tile:
                final.append(tile+1)
                self.score += 2**(int(tile)+1)
                base = 0
            else:
                if base:
                    final.append(base)  # Don't append zeros
                base = tile
        if base:
            final.append(base)
        # TODO: put moved=True into merge_row
        # if len(final) is not SIZE, then moved = True ?
        final += [0]*(SIZE-len(final))  # Pad with zeros
        return np.array(final)

    def move_left(self):
        """Execute left move or returns False if unable"""
        # Row by Row
        moved = False
        for i in range(SIZE):
            row = self.board[i]
            new_row = self.merge_row(row)
            if any(new_row != row):
                moved = True
                self.board[i] = new_row
        return moved

    def move_up(self):
        """Execute up move or returns False if unable"""
        # Column by Column
        moved = False
        for i in range(SIZE):
            row = self.board[:, i]
            new_row = self.merge_row(row)
            if any(new_row != row):
                moved = True
                self.board[:, i] = new_row
        return moved

    def move_right(self):
        """Execute right move or returns False if unable"""
        moved = False
        for i in range(SIZE):
            row = self.board[i, ::-1]
            new_row = self.merge_row(row)
            if any(new_row != row):
                moved = True
                self.board[i, ::-1] = new_row
        return moved

    def move_down(self):
        """Execute down move or returns False if unable"""
        moved = False
        for i in range(SIZE):
            row = self.board[::-1, i]
            new_row = self.merge_row(row)
            if any(new_row != row):
                moved = True
                self.board[::-1, i] = new_row
        return moved

