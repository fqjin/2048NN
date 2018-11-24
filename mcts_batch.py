import numpy as np
from board import Board


def mcts_fixed_batch(game, number=10, move_order=(0, 1, 2, 3)):
    """Run tree search using a fixed move order for generating lines
    Lines are kept in a list, and played through as a batch.

    Args:
        game (Board): the starting game state
        number (int): # of lines to try for each move.
            Defaults to 10
        move_order: tuple of the 4 move indices in order.
            Defaults to (0, 1, 2, 3)

    Returns:
        list: score increase for each move [Left, Up, Right, Down]

    """
    original = game.copy()
    scores = [0, 0, 0, 0]
    lines = []
    for i in range(4):
        if game.move(i):
            game.restore(original.board, original.score)
            # Sacrifice one move(i) computation
            for _ in range(number):
                temp = game.copy()
                temp.move(i)
                temp.generate_tile()
                temp.index = i
                lines.append(temp)

    dead = []
    while lines:
        for line in lines:
            for j in move_order:
                if line.move(j):
                    line.generate_tile()
                    break
            else:
                dead.append(line)

        if dead:
            for line in dead:
                scores[line.index] += line.score
                lines.remove(line)
            dead = []

    scores = [score/number - original.score
              if score else 0.0
              for score in scores]
    return scores

