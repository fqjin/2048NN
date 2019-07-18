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
    result = np.zeros(4, dtype=np.float32)
    for i in range(4):
        temp = origin.copy()
        if temp.move(i):
            games.extend([temp.copy() for _ in range(number)])
        else:
            result[i] = -1
    if not games:
        return result
    Board.generate_tile_batch(games)
    notdead = games.copy()

    model.eval()
    with torch.no_grad():
        while True:
            for i in range(4):
                if i == 0:
                    subgames = notdead
                    boards = [g.board for g in subgames]
                    preds = model.forward(torch.stack(boards).cuda().float().unsqueeze(1))
                    preds = torch.argsort(preds.cpu(), dim=1, descending=True)
                    # TODO: Convert preds to List of ints
                    for g, p in zip(subgames, preds):
                        g.pred = p
                    moves = preds[:, i]
                else:
                    subgames = [g for g in notdead if not g.moved]
                    moves = [g.pred[i] for g in subgames]
                Board.move_batch(subgames, moves)
            notdead = [g for g in notdead if g.moved]
            if not notdead:
                break
            Board.generate_tile_batch(notdead)

    index = 0
    scores = np.asarray([g.score for g in games])
    scores -= origin.score
    scores = np.log10(scores + 1)  # log conversion shown to help
    for i in range(4):
        if not result[i]:
            result[i] = np.mean(scores[index:index+number])
            index += number
    return result


def mcts_nn_min(model, origin, number=10):
    """Run batch tree search with pytorch model making lines.
    # of moves until the first dead line is used as metric.
    Search is terminated upon first dead line.

    Args:
        model: pytorch model to predict moves
        origin (Board): the starting game state
        number (int): # of lines to try for each move.
            Defaults to 10

    Returns:
        list: score increase for each move [Left, Up, Right, Down]

    """
    games = []
    indices = []
    result = np.zeros(4, dtype=np.float32)
    for i in range(4):
        temp = origin.copy()
        temp.score = 0
        if temp.move(i):
            games.extend([temp.copy() for _ in range(number)])
            indices.append(i)
        else:
            result[i] = -1
    if not games:
        return result
    Board.generate_tile_batch(games)

    counter = 0
    model.eval()
    with torch.no_grad():
        while True:
            for i in range(4):
                if i == 0:
                    subgames = games
                    boards = [g.board for g in subgames]
                    preds = model.forward(torch.stack(boards).cuda().float().unsqueeze(1))
                    preds = torch.argsort(preds.cpu(), dim=1, descending=True)
                    for g, p in zip(subgames, preds):
                        g.pred = p
                    moves = preds[:, i]
                else:
                    subgames = [g for g in games if not g.moved]
                    moves = [g.pred[i] for g in subgames]
                Board.move_batch(subgames, moves)

            moved = [g.moved for g in games]
            for i in range(len(indices)):
                if 0 in moved[number * i:number * (i + 1)]:
                    result[indices[i]] = counter
                    indices[i] = None
            newgames = []
            newindices = []
            for i, idx in enumerate(indices):
                if idx is not None:
                    newgames.extend(games[number * i:number * (i + 1)])
                    newindices.append(idx)
            games, indices = newgames, newindices
            if not games:
                break
            Board.generate_tile_batch(games)
            counter += 1

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
            pred = model.forward(game.board.float().cuda()[None, None, ...])[0]
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
