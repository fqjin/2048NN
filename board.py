import os
import numpy as np
import pickle
import random
import torch
from random import randrange

# Seed
seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Print options
np.set_printoptions(precision=3, suppress=True)
CLEAR = 'clear' if os.name == 'posix' else 'cls'
ARROWS = {0: '  \u2b9c                   .',
          1: '       \u2b9d              .',
          2: '            \u2b9e         .',
          3: '                 \u2b9f    .'}


def get_tiles(x):
    """Returns list of tiles

    Using for and append is about 9% faster than
    using list comprehension and >> (4*i)
    """
    tiles = []
    for _ in range(16):
        tiles.append(x & 0xF)
        x >>= 4
    return tiles


def draw(x, score=None):
    """Prints board state"""
    tiles = get_tiles(x)
    expo = np.power(2, tiles)
    expo = expo.reshape(4, 4)
    expo = str(expo)
    print(expo.replace('1 ', '. ', 12).replace('1]', '.]', 4))
    print(' Score : {}'.format(score))


def transpose(x):
    a1 = x & 0xF0F00F0FF0F00F0F
    a2 = x & 0x0000F0F00000F0F0
    a3 = x & 0x0F0F00000F0F0000
    a = a1 | (a2 << 12) | (a3 >> 12)
    b1 = a & 0xFF00FF0000FF00FF
    b2 = a & 0x00FF00FF00000000
    b3 = a & 0x00000000FF00FF00
    return b1 | (b2 >> 24) | (b3 << 24)


def countzero(x):
    x |= (x >> 2) & 0x3333333333333333
    x |= (x >> 1)
    x = ~x & 0x1111111111111111
    x += x >> 32
    x += x >> 16
    x += x >> 8
    x += x >> 4
    return x & 0xf


def generate_init_tiles():
    """Returns a board (int64) with 2 tiles"""
    pos1 = randrange(16)
    pos2 = randrange(16)
    while pos1 == pos2:
        pos2 = randrange(16)
    if randrange(10):
        tile1 = 1 << (4*pos1)
    else:
        tile1 = 1 << (4*pos1 + 1)
    if randrange(10):
        tile2 = 1 << (4*pos2)
    else:
        tile2 = 1 << (4*pos2 + 1)
    return tile1 | tile2


def generate_tile(board):
    """Places a 2 or 4 in a random empty tile"""
    position = randrange(countzero(board))
    x = board
    tile = 1
    while True:
        if (x & 0xf) == 0:
            if position == 0:
                break
            else:
                position -= 1
        x >>= 4
        tile <<= 4
    if randrange(10):  # 90% put 2-tile, 10% put 4-tile
        return board | tile
    else:
        return board | (tile << 1)


def move_row(x, rev):
    row = [(x >> i) & 0xF for i in (0, 4, 8, 12)]
    if rev:
        row.reverse()
    final = []
    score = 0
    base = 0
    for tile in row:
        if tile == 0:
            continue  # Skips zeros
        if base == tile:
            newtile = tile + 1
            score += 2 ** newtile
            if newtile == 16:
                newtile = 0  # When two 32768 tiles merge, they annihilate
            final.append(newtile)
            base = 0
        else:
            if base:  # Don't append zeros
                final.append(base)
            base = tile
    if base:
        final.append(base)
    final += [0] * (4 - len(final))  # Pad with zeros
    if rev:
        final.reverse()
    final = (final[0] << 0) | (final[1] << 4) | (final[2] << 8) | (final[3] << 12)
    moved = (x != final)
    return final, score, moved


# Generate merge tables
# Making a new table takes ~ 0.32 seconds
# Loading the pickle takes ~ 0.08 seconds
try:
    with open('merge_tables.pickle', 'rb') as pkl:
        merge_table, merge_table_rev = pickle.load(pkl)
except FileNotFoundError:
    merge_table = []
    merge_table_rev = []
    for brd in range(2**16):
        merge_table.append(move_row(brd, False))
        merge_table_rev.append(move_row(brd, True))
    with open('merge_tables.pickle', 'wb') as pkl:
        pickle.dump([merge_table, merge_table_rev], pkl)


def move(x, direction):
    """Execute move in a direction

    Args:
        x: input board
        direction: move direction index
            0 : Left
            1 : Up
            2 : Right
            3 : Down
            Note: other indices still work modulo 4

    Returns:
        board, score, moved (bool)
    """
    final = 0
    score = 0
    moved = False
    if direction & 0x2:
        table = merge_table_rev
    else:
        table = merge_table

    if direction & 0x1:
        x = transpose(x)
        for i in (0, 16, 32, 48):
            f, s, m = table[(x >> i) & 0xFFFF]
            final |= f << i
            score += s
            moved |= m
        final = transpose(final)
    else:
        for i in (0, 16, 32, 48):
            f, s, m = table[(x >> i) & 0xFFFF]
            final |= f << i
            score += s
            moved |= m

    return final, score, moved


def play_fixed(board=None, press_enter=False):
    """Run 2048 with the fixed move priority L,U,D,R

    As shown by previous networks, L,U,D,R move order
    is superior to L,U,R,D. Using a 1000 sample fixed
    moved test, the p value for the mean is 2x10^-5

    Args (optional):
        board (int64): the starting board state
        press_enter (bool): Whether keyboard press is
            required for each step. Defaults to False.
            Type 'q' to quit when press_enter is True.
    """
    if not board:
        board = generate_init_tiles()
    score = 0
    count = 0
    if press_enter:
        draw(board, score)
    while True:
        if press_enter and input() == 'q':
            break
        for i in (0, 1, 3, 2):
            f, s, m = move(board, i)
            if m:
                board = generate_tile(f)
                score += s
                count += 1
                if press_enter:
                    print(ARROWS[i])
                    draw(board, score)
                break
        else:
            return board, score, count


class BoardArray:
    """Board array object stores 2048 boards and scores

    Numbers are stored as their log-base-2
    Uses the 4-bit nibble tile format (see github/nneonneo/2048-ai)
    Note: int is immutable, so functions do not modify arguments

    Args:
        boards: list of starting boards

    Attributes:
        boards: list of boards
        scores: list of scores
    """
    def __init__(self, boards):
        self.boards = boards
        self.scores = [0] * len(boards)

    def move_batch(self, move_list):
        """Perform moves on a batch of games

        Args:
            move_list: ordered direction indices (0 to 3) for
                each board in self.boards. If a tuple is given,
                it must be length 4 and is used for every board.

        Returns:
            dead boards, dead scores
        """
        if isinstance(move_list, tuple):
            move_list = [move_list] * len(self.boards)
        # elif len(move_list) != len(self.boards):
        #     raise ValueError('move_list not len boards')
        new_b = []
        new_s = []
        dead_s = []
        for board, score, moves in zip(self.boards, self.scores, move_list):
            for i in moves:
                f, s, m = move(board, i)
                if m:
                    new_b.append(generate_tile(f))
                    new_s.append(score + s)
                    break
            else:
                # dead_b.append(board)
                dead_s.append(score)
        self.boards = new_b
        self.scores = new_s
        return dead_s


def play_fixed_batch(number):
    array = BoardArray([generate_init_tiles() for _ in range(number)])
    scores = []
    while array.boards:
        dead_s = array.move_batch((0, 1, 3, 2))
        if dead_s:
            # print('{} died on move {}'.format(len(dead_s), count))
            scores.extend(dead_s)
    return scores




