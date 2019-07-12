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
                    preds = model.forward(torch.stack(boards).float().unsqueeze(1).cuda())
                    preds = torch.argsort(preds.cpu(), dim=1, descending=True)
                    for g, p in zip(subgames, preds):
                        g.pred = p
                    moves = preds[:, i]
                else:
                    moves = [g.pred[i] for g in subgames]
                Board.move_batch(subgames, moves)
            notdead = [g for g in notdead if g.moved]
            Board.generate_tile_batch(notdead)
            if not notdead:
                break

    scores = np.asarray([g.score for g in games])
    logscores = np.log10(scores+1)
    np.savez('models/{}.npz'.format(name), scores=scores)
    print(name)
    print('Score: {0:.0f} / {1:.0f}'.format(np.mean(scores), np.std(scores)/np.sqrt(number)))
    print('Log Score: {0:.3f} / {1:.3f}'.format(np.mean(logscores), np.std(logscores)/np.sqrt(number)))


if __name__ == '__main__':
    from board import play_fixed
    from network import ConvNet
    a = Board()
    play_fixed(a)
    a.board -= 1
    a.draw()

    for i in [0, 9]:  # 29, 39, 49, 59]:
        name = '20190710/80_100_epox10_lr0.0043pre_e{}'.format(i)

        m = ConvNet(channels=32, num_blocks=4)
        m.load_state_dict(torch.load('models/{}.pt'.format(name)))
        m.to('cuda')
        t = time()
        eval_nn(name, m, origin=a)
        t = time() - t
        print('{0:.3f} seconds'.format(t))
        print('-'*10)
