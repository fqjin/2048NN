import os
import numpy as np
from board import Board, CLEAR, ARROWS


def mcts_fixed_batch(origin, number=10, move_order=(0, 1, 2, 3)):
    """Run batch tree search using a fixed move order

    Input game is copied to a list of games.
    Each line played to end using move_batch
    Code is very similar to `play_fixed_batch`

    Args:
        origin (Board): the starting game state
        number (int): # of lines to simulate for each move.
            Defaults to 10
        move_order: tuple of the 4 move indices in order.
            Defaults to (0, 1, 2, 3)

    Returns:
        list: score increase for each move [Left, Up, Right, Down]

    """
    games = []
    result = [1, 1, 1, 1]
    for i in range(4):
        temp = origin.copy()
        if temp.move(i):
            games.extend([temp.copy() for _ in range(number)])
        else:
            result[i] = 0
    for g in games:
        g.generate_tile()
        # g.moved = 0 not needed because temp.move(i) doesn't change moved

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

    index = 0
    scores = [g.score for g in games]
    for i in range(4):
        if result[i] == 0:
            continue
        else:
            result[i] = sum(scores[index:index+number]) / number - origin.score
            index += number
    return result


def play_mcts_fixed_batch(game, number=5, move_order=(0, 1, 2, 3), verbose=False):
    """Play a game using the default mcts_fixed

    Args:
        game (Board): the starting game state. If `None`
            is passed, a new Board is generated.
        number (int): # of lines to try for each move.
            Defaults to 5
        move_order: tuple of the 4 move indices in order.
            Defaults to (0, 1, 2, 3)
        verbose (bool): whether to print mcts scores
            Defaults to False

    """
    if not game:
        game = Board(device='cpu', gen=True, draw=True)
    while True:
        scores = mcts_fixed_batch(game, number, move_order)
        os.system(CLEAR)
        if verbose:
            print(scores)
        for i in np.flipud(np.argsort(scores)):
            # TODO: Test torch vs numpy speed
            if game.move(i):
                print(ARROWS[i])
                game.generate_tile()
                game.draw()
                break
        else:
            print('Game Over')
            break
