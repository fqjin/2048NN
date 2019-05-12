import os
import numpy as np
import torch
from board import Board, CLEAR, ARROWS


def mcts_nn(model, origin, number=10):
    """Run tree search with pytorch model making lines.
    Batch implementation for efficiency.

    Args:
        model: pytorch model to predict moves
        origin (Board): the starting game state
        number (int): # of lines to try for each move.
            Defaults to 10

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

    model.eval()
    with torch.no_grad():
        while True:
            for i in range(4):
                subgames = [
                    g for g in games if not g.dead and not g.moved
                ]
                if i == 0:
                    boards = [g.board for g in subgames]
                    preds = model.forward(torch.stack(boards).float())
                    preds = torch.argsort(preds, dim=1, descending=True)
                    for g, p in zip(subgames, preds):
                        g.pred = p
                    moves = preds[:, i]
                else:
                    moves = [g.pred[i] for g in subgames]
                Board.move_batch(subgames, moves)
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
        if result[i]:
            result[i] = sum(scores[index:index+number]) / number - origin.score
            index += number
    return result


def play_nn(model, game, press_enter=False, device='cpu', verbose=False):
    """Play through a game using a pytorch NN.

    Moves are selected by the pytorch model.
    No monte carlo simulations are used.

    Args:
        model: pytorch model to predict moves
        game (Board): the starting game state. If `None`
            is passed, will generate a new Board.
        press_enter (bool): Whether keyboard press is
            required for each step. Defaults to False.
            Type 'q' to quit when press_enter is True.
        device: torch device. Defaults to 'cpu'
        verbose (bool): whether to print mcts scores
            Defaults to False

    """
    if not game:
        game = Board(gen=True, draw=True, device=device)
    model.eval()
    with torch.no_grad():
        while True:
            if press_enter and input() == 'q':
                break
            pred = model.forward(game.board.unsqueeze(0).float())[0]
            os.system(CLEAR)
            if verbose:
                print(pred)
            for i in torch.argsort(pred, descending=True):
                if game.move(i):
                    print(ARROWS[i.item()])
                    game.generate_tile()
                    game.draw()
                    break
            else:
                print('Game Over')
                break


def make_data(game, model, number=10, verbose=False):
    """Plays through one game using mcts_nn. Returns all
    boards and computed scores of the main line for training.

    Args:
        game (Board): the starting game state. If `None`
            is passed, a new Board is generated.
        model: keras model to predict moves
        number (int): # of lines to try for each move.
            Defaults to 10
        verbose (bool): whether to print mcts scores
            Defaults to False

    Returns:
        boards: list of boards
        results: list of mcts_nn scores

    """
    boards = []
    results = []
    if not game:
        game = Board(gen=True)
    while True:
        scores = mcts_nn(game, model, number)
        if verbose:
            print(np.trunc(scores))
        if sum(scores) > 0:
            boards.append(np.copy(game.board))
            results.append(scores)
        for i in np.flipud(np.argsort(scores)):
            if game.move(i):
                game.generate_tile()
                game.draw()
                break
        else:
            print('Game Over')
            break

    return boards, results