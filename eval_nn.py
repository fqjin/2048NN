import numpy as np
import torch
from time import time
from board import Board


def eval_nn(name, model, origin=None, number=1000, device='cpu'):
    """Simulate games using a model

    Args:
        name: name to save
        model: pytorch model to predict moves
        origin (Board): the starting game state
            Defaults to None (generate new)
        number (int): # of lines
            Defaults to 1000
        device: torch device if generating new games

    Returns:
        saves

    """
    if origin is None:
        games = [Board(device=device) for _ in range(number)]
    else:
        origin = origin.copy()
        origin.score = 0
        games = [origin.copy() for _ in range(number)]
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

    scores = [g.score for g in games]
    np.savez('models/{}.npz'.format(name), scores=scores)
    print('{} ave score: {} / {}'.format(name,
                                         np.mean(scores),
                                         np.mean(np.log10(scores+1))))


if __name__ == '__main__':
    name = '0_10_epox100_lr0.1_e0'
    device = 'cpu'  # Still faster on cpu using this ConvNet
    from board import play_fixed
    a = Board()
    play_fixed(a)
    a.board -= 1
    a.draw()

    from network import ConvNet
    m = ConvNet()
    m.load_state_dict(torch.load('models/{}.pt'.format(name)))
    m.to(device)
    t = time()
    eval_nn(name, m, origin=a, device=device)
    t = time() - t
    print('{0:.3f} seconds'.format(t))
