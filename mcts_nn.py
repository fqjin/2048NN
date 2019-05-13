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
    result = [0, 0, 0, 0]
    for i in range(4):
        temp = origin.copy()
        if temp.move(i):
            games.extend([temp.copy() for _ in range(number)])
        else:
            result[i] = -1
    if not len(games):
        return result
    for g in games:
        g.generate_tile()
    notdead = games.copy()

    model.eval()
    with torch.no_grad():
        while True:
            for i in range(4):
                subgames = [g for g in notdead if not g.moved]
                if i == 0:
                    boards = [g.board for g in subgames]
                    preds = model.forward(torch.stack(boards).float().unsqueeze(1))
                    preds = torch.argsort(preds, dim=1, descending=True)
                    for g, p in zip(subgames, preds):
                        g.pred = p
                    moves = preds[:, i]
                else:
                    moves = [g.pred[i] for g in subgames]
                Board.move_batch(subgames, moves)
            for g in notdead:
                if g.moved:
                    g.moved = 0
                    g.generate_tile()
                else:
                    g.dead = 1
            notdead = [g for g in notdead if not g.dead]
            if not len(notdead):
                break

    index = 0
    scores = [g.score for g in games]
    for i in range(4):
        if not result[i]:
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
            pred = model.forward(game.board.float()[None, None, ...])[0]
            for i in torch.argsort(pred, descending=True):
                if game.move(i):
                    game.generate_tile()
                    if verbose:
                        os.system(CLEAR)
                        print(pred)
                        print(ARROWS[i.item()])
                        game.draw()
                    break
            else:
                print(game.score)
                print('Game Over')
                break


def selfplay(name, model, game, number=10, device='cpu', verbose=False):
    """Plays through one game using mcts_nn. Returns all
    boards and move choices of the main line for training.

    Args:
        name (int): name for data
        game (Board): the starting game state. If `None`
            is passed, a new Board is generated.
        model: keras model to predict moves
        number (int): # of lines to try for each move.
            Defaults to 10
        device: torch device. Defaults to 'cpu'
        verbose (bool): whether to print anything
            Defaults to False

    Returns:
        boards: list of boards
        moves: list of mcts_nn moves

    """
    if not game:
        game = Board(gen=True, draw=verbose, device=device)
    boards = []
    moves = []
    while True:
        if not len(moves) % 20:
            print('Move {}'.format(len(moves)))
        boards.append(game.board.clone())
        pred = mcts_nn(model, game, number=number)
        # Only need to do argmax. If not possible, game is dead
        i = np.argmax(pred)
        if game.move(i):
            game.generate_tile()
            moves.append(i)
            if verbose:
                os.system(CLEAR)
                print(pred)
                print(ARROWS[i.item()])
                game.draw()
            continue
        else:
            boards.pop()
            break
    print(game.score)
    print('Game Over')
    print('{} moves'.format(len(moves)))
    if isinstance(name, int):
        name = str(name).zfill(5)
    np.savez('selfplay/'+name, boards=torch.stack(boards), moves=moves)
    print('Saved as {}.npz'.format(name))
