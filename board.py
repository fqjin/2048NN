import torch
import random
from random import randint, randrange

s = 12345
random.seed(s)
torch.manual_seed(s)

# Board Dimensions
SIZE = 4
DIMENSIONS = (SIZE, SIZE)  # (4,4)
SIZE_SQRD = SIZE*SIZE  # 16


class Board:
    """Board object stores 2048 board state
    Numbers are stored as their log-base-2

    Args:
        device: torch device
        gen (bool): whether to generate two initial tiles.
            Defaults to True
        draw (bool): whether to draw the board, if gen is True.
            Defaults to False

    Attributes:
        board: torch tensor of board tiles, stored as log-base-2
        score: int score, the sum of all combination values

    """
    def __init__(self, device, gen=True, draw=False):
        self.device = device
        self.board = torch.zeros(DIMENSIONS, dtype=torch.uint8, device=device)
        # TODO: compare dtypes
        self.score = 0
        if gen:
            self.generate_tile()
            self.generate_tile()
            if draw:
                self.draw()

    def generate_tile(self):
        """Places a 2 or 4 in a random empty tile
        Unhandled error if board is full
        Chance of 2 is 90%
        """
        empty = (self.board == 0).nonzero()
        position = empty[randrange(len(empty))]
        if randint(0, 9):
            self.board[position[0], position[1]] = 1
        else:
            self.board[position[0], position[1]] = 2
        #   self.board[tuple(position)] is 3 times slower
        # self.board[position[0], position[1]] = (not randint(0, 9)) * 2 or 1

    def draw(self):
        """Prints board state"""
        # TODO: Needs pretty print for torch tensor
        # print(str(2**self.board).replace('1', ' ', SIZE_SQRD))
        print(2**self.board.int())
        # 2**uint8 overflows
        print(' Score : {}'.format(self.score))

    def restore(self, board, score):
        """Sets board and score state to the input values"""
        self.board = board.clone()  # need to copy
        self.score = score  # immutable does not need copying

    def copy(self):
        """Returns a copy as a new Board object"""
        temp = Board(device=self.device, gen=False)
        temp.board = self.board.clone()
        temp.score = self.score
        # Explicit saves time rather than calling restore
        return temp

    def merge_row(self, row):
        """Merges input row and shifts tiles to the left side
        Score is updated if any new tiles are made

        Args:
            row: numpy array of row to merge

        Returns:
            array: row after shift and merge
            bool: True if row changed, False if unchanged

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
            return torch.tensor(final), True
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
                x = self.board[i].flip(0)
                x, moved = self.merge_row(x)
                self.board[i] = x.flip(0)
                # self.board[i, ::-1], moved = self.merge_row(self.board[i, ::-1])
                # torch cannot use negative strides
            elif direction == 3:
                x = self.board[:, i].flip(0)
                x, moved = self.merge_row(x)
                self.board[:, i] = x.flip(0)
                # self.board[::-1, i], moved = self.merge_row(self.board[::-1, i])
            else:
                raise IndexError('Only 0 to 3 accepted as directions')

            if moved:
                moved_any = True
        return moved_any


def play_fixed(game=None, press_enter=False):
    """Run 2048 with the fixed move priority L,U,R,D.

    Args (optional):
        game (Board): the starting game state.
            Default will generate a new Board.
        press_enter (bool): Whether keyboard press is
            required for each step. Defaults to False.
            Type 'q' to quit when press_enter is True.

    """
    if not game:
        game = Board(gen=True)
    while True:
        if press_enter and input() == 'q':
            break
        for i in range(4):
            if game.move(i):
                game.generate_tile()
                game.draw()
                break
        else:
            print('Game Over')
            break
