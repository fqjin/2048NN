import numpy as np
from board import Board


def mcts_fixed(game, number=5, move_order=(0, 1, 2, 3)):
    """Run tree search using a fixed move order for generating lines

    Args:
        game (Board): the starting game state
        number (int): # of lines to try for each move.
            Defaults to 5
        move_order: tuple of the 4 move indices in order.
            Defaults to (0, 1, 2, 3)

    Returns:
        list: score increase for each move [Left, Up, Right, Down]

    """
    original = game.copy()
    scores = [0, 0, 0, 0]

    for i in range(4):
        if not game.move(i):
            # scores[i] = 0 or original_score
            # game.restore(original.board, original.score)
            # I think I can pass since moved=False nothing changed
            # TODO: Verify this and simplify if-else
            pass
        else:
            for _ in range(number):
                game.restore(original.board, original.score)
                game.move(i)
                game.generate_tile()
                while True:
                    for j in move_order:
                        if game.move(j):
                            game.generate_tile()
                            break
                    else:
                        # print('Game Over')
                        # game.draw()
                        break
                scores[i] += game.score

            # Calculate average final score
            scores[i] /= number
            scores[i] -= original.score
            game.restore(original.board, original.score)

    return scores


def play_mcts_fixed(game, number=5, move_order=(0, 1, 2, 3), verbose=False):
    """Play a game using the default mcts_fixed

    Args:
        game (Board): the starting game state
        number (int): # of lines to try for each move.
            Defaults to 5
        move_order: tuple of the 4 move indices in order.
            Defaults to (0, 1, 2, 3)
        verbose (bool): whether to print mcts scores
            Defaults to False

    """
    while True:
        scores = mcts_fixed(game, number=number, move_order=move_order)
        if verbose:
            print(scores)
        for i in np.flipud(np.argsort(scores)):
            if game.move(i):
                game.generate_tile()
                game.draw()
                break
        else:
            print('Game Over')
            break

