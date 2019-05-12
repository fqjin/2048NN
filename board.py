import os
import numpy as np
import random
import torch
from random import randint, randrange

# Seed
s = 12345
random.seed(s)
np.random.seed(s)
torch.manual_seed(s)

# Board Dimensions
SIZE = 4
DIMENSIONS = (SIZE, SIZE)  # (4,4)
SIZE_SQRD = SIZE*SIZE  # 16

# Print options
np.set_printoptions(suppress=True)
CLEAR = 'clear' if os.name == 'posix' else 'cls'
ARROWS = {0: '        \u2b9c        .',
          1: '        \u2b9d        .',
          2: '        \u2b9e        .',
          3: '        \u2b9f        .'}


class Board:
    """Board object stores 2048 board state
    Numbers are stored as their log-base-2

    Args:
        device: torch device. Defaults to 'cpu'
        gen (bool): whether to generate two initial tiles.
            Defaults to True
        draw (bool): whether to draw the board, if gen is True.
            Defaults to False

    Attributes:
        board: torch tensor of board tiles, stored as log-base-2
        score: int score, the sum of all combination values

    """
    def __init__(self, device='cpu', gen=True, draw=False):
        self.device = device
        self.board = torch.zeros(DIMENSIONS, dtype=torch.uint8, device=device)
        self.score = 0
        self.dead = 0
        self.moved = 0
        if gen:
            self.generate_tile()
            self.generate_tile()
            if draw:
                self.draw()

    def generate_tile(self):
        """Places a 2 or 4 in a random empty tile
        Chance of 2 is 90%

        Raises:
            ValueError: if board is full
                (empty range for randrange)

        """
        empty = (self.board == 0).nonzero()
        # if len(empty) == 0:
        #     print('Full')
        #     self.draw()
        position = empty[randrange(len(empty))]
        self.board[position[0], position[1]] = 1 if randint(0, 9) else 2
        # self.board[tuple(position)] is 3 times slower

    def draw(self):
        """Prints board state"""
        expo = 2**self.board.float()
        print(str(expo.cpu().numpy()).replace('1.', ' .', SIZE_SQRD))
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
        # TODO: Switch from merge_row to merge_row_batch.
        #       This deprecates merge_row
        moved_any = 0
        if direction == 0:
            for i in range(SIZE):
                self.board[i], moved = self.merge_row(self.board[i])
                moved_any += moved
        elif direction == 1:
            for i in range(SIZE):
                self.board[:, i], moved = self.merge_row(self.board[:, i])
                moved_any += moved
        elif direction == 2:
            for i in range(SIZE):
                # torch cannot use negative strides
                x = self.board[i].flip(0)
                x, moved = self.merge_row(x)
                self.board[i] = x.flip(0)
                moved_any += moved
        elif direction == 3:
            for i in range(SIZE):
                x = self.board[:, i].flip(0)
                x, moved = self.merge_row(x)
                self.board[:, i] = x.flip(0)
                moved_any += moved
        else:
            raise IndexError('''Only 0 to 3 accepted as directions,
                {} given'''.format(direction))
        return bool(moved_any)

    # TODO: it will be more efficient to use a BatchBoard object

    @staticmethod
    def merge_row_batch(rows):
        """Merge a batch of rows

        Args:
            rows: tensor of rows, should be shape (_, SIZE)

        Returns:
            newrows: new tensor after performing move left to all rows
            score: score generated from combinations, per row

        """
        newrows = rows.clone().t()  # transpose to index columns first
        scores = torch.zeros(len(rows), dtype=torch.int, device=rows.device)
        # Shift nonzeros to the left
        for i in reversed(range(SIZE - 1)):
            temp = newrows[i] == 0  # column is zero
            for j in range(SIZE - 1 - i):
                newrows[i+j] += newrows[i+j+1] * temp  # shift over if zero
                newrows[i+j+1] *= (1 - temp)  # clear after shift
        # Merge tiles
        for i in range(SIZE - 1):
            temp = (newrows[i] == newrows[i+1]) * (newrows[i] != 0)
            newrows[i] += temp
            scores += 2 ** newrows[i].int() * temp.int()
            newrows[i+1] *= (1 - temp)
            for j in range(1, SIZE - 1 - i):
                newrows[i+j] += newrows[i+j+1] * temp
                newrows[i+j+1] *= (1 - temp)
        # Check if moved
        newrows = newrows.t()
        moved = torch.sum((rows != newrows), dim=1)
        return newrows, scores, moved

    @staticmethod
    def move_batch(games, moves):
        """Perform moves on a batch of games

        Args:
            games: a list of Board objects
            moves: a list of direction indices (0 to 3)
                if moves in an integer, it is broadcasted

        Returns:
            None
            - board, moved, and score attributes of games are modified

        """
        if len(games) == 0:
            return None
        if isinstance(moves, int):
            moves = [moves] * len(games)
        rows = []
        for game, move in zip(games, moves):
            if move == 0:
                rows.append(game.board)
            elif move == 1:
                rows.append(game.board.t())
            elif move == 2:
                rows.append(game.board.flip(1))
            elif move == 3:
                rows.append(game.board.t().flip(1))
            else:
                raise IndexError('''Only 0 to 3 accepted as directions, 
                    {} given'''.format(move))
        rows = torch.cat(rows)
        newrows, scores, moved = Board.merge_row_batch(rows)
        newrows = newrows.split(SIZE)
        scores = torch.sum(scores.view(-1, SIZE), dim=1)
        moved = torch.sum(moved.view(-1, SIZE), dim=1)
        for game, board, score, move, m in \
                zip(games, newrows, scores, moves, moved):
            if m:
                game.moved += m.item()
                game.score += score.item()
                if move == 0:
                    game.board = board
                elif move == 1:
                    game.board = board.t()
                elif move == 2:
                    game.board = board.flip(1)
                else:
                    game.board = board.flip(1).t()


def play_fixed(game=None, device='cpu'):
    """Run 2048 with the fixed move priority L,U,R,D.

    Args (optional):
        game (Board): the starting game state.
            Default will generate a new Board.
        press_enter (bool): Whether keyboard press is
            required for each step. Defaults to False.
            Type 'q' to quit when press_enter is True.
        device: torch device. Defaults to 'cpu'

    """
    if not game:
        game = Board(device=device, gen=True)
    while True:
        # if press_enter and input() == 'q':
        #     break
        for i in range(4):
            if game.move(i):
                game.generate_tile()
                # game.draw()
                break
        else:
            # game.draw()
            print(game.score)
            # print('Game Over')
            return game


def play_fixed_batch(games=None, number=None, device='cpu'):
    """Run 2048 with the fixed move priority L,U,R,D.

    Args (optional):
        games: a list of games to play. Defaults to None
        number: if no games provided, generate this number
        device: torch device. Defaults to 'cpu'

    """
    if not games:
        # if not number:
        #     raise ValueError('games and number both None')
        games = [Board(device=device, gen=True) for _ in range(number)]
    # fixed_moves = torch.arange(4).repeat((len(games), 1))
    while True:
        for i in range(4):
            subgames = [
                g for g in games if not g.dead and not g.moved
            ]
            Board.move_batch(subgames, i)
        for g in games:
            if g.moved:
                g.moved = 0
                g.generate_tile()
            else:
                g.dead = 1
        if 0 not in [g.dead for g in games]:
            break
    for g in games:
        # g.draw()
        print(g.score)
    # print('Game Over')
    # return games
