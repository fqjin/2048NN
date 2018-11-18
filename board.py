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

    """
    def __init__(self, gen=True):
        self.board = np.zeros(DIMENSIONS)
        self.score = 0
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
            self.board[position[0], position[1]] = 2
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
        # Explicit saves time rather than calling restore
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
                if base:  # Don't append zeros
                    final.append(base)
                base = tile
        if base:
            final.append(base)
        # Cannot use len(final) to predict if moved
        final += [0] * (SIZE - len(final))  # Pad with zeros

        # `list(row) != final` is faster than `any(row != final)`
        # if-else avoids computing np.array(final) when not moved
        if list(row) != final:
            return np.array(final), True
        else:
            return row, False

    def move(self, direction):
        """Execute move in a direction. Returns False if unable

        Args:
            direction: index representing move direction
                0 : Left
                1 : Up
                2 : Right
                3 : Down

        Returns:
            bool: True if able to move, False if unable

        Raises:
            IndexError: if direction index is not 0 to 3

        """
        moved_any = False
        for i in range(SIZE):
            if direction == 0:
                self.board[i], moved = self.merge_row(self.board[i])
            elif direction == 1:
                self.board[:, i], moved = self.merge_row(self.board[:, i])
            elif direction == 2:
                self.board[i, ::-1], moved = self.merge_row(self.board[i, ::-1])
            elif direction == 3:
                self.board[::-1, i], moved = self.merge_row(self.board[::-1, i])
            else:
                raise IndexError('Only 0 to 3 accepted as directions')

            if moved:
                moved_any = True
        return moved_any

