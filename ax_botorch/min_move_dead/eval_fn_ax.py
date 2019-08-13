import os
import sys
import numpy as np
import torch
from time import time
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('../../')
from board import Board
from game_dataset import GameDataset
from network import ConvNet

start_board = Board()

t_tuple = (int(input('t_start: ')), int(input('t_end: ')))
print('Using games {} to {}'.format(*t_tuple))
# validation not used
batch_size = 1024
mom_tuple = (0.95, 0.85)

os.chdir('../..')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_set = GameDataset('selfplay/min_move_dead/min', t_tuple[0], t_tuple[1], device, augment=False)
train_dat = DataLoader(train_set, batch_size=batch_size, shuffle=True)
os.chdir('ax_botorch/min_move_dead')


def train_loop(model, data, loss_fn, optimizer):
    model.train()
    running_loss = 0
    for x, y in data:
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        running_loss += loss.data.item()
        loss.backward()
        optimizer.step()
    running_loss /= len(data)
    return running_loss


def train(params):
    lr = params['lr']
    decay = params['decay']
    epochs = params['epochs']

    m = ConvNet(channels=32, num_blocks=5)
    # name = '20190710/80_100_epox10_lr0.0043pre_e9'
    # m.load_state_dict(torch.load('../models/{}.pt'.format(name)))
    m.to(device)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(m.parameters(),
                                lr=lr,
                                momentum=mom_tuple[0],
                                nesterov=True,
                                weight_decay=decay)

    lr = np.geomspace(lr, 0.5, epochs//2)
    lr = np.concatenate([lr,
                         np.flip(lr[:-1]),
                         # np.geomspace(lr[0], lr[0]/10, epochs//10+1)[1:],
                         ])
    momentum = np.linspace(mom_tuple[0], mom_tuple[1], epochs//2)
    momentum = np.concatenate([momentum,
                               np.flip(momentum[:-1]),
                               # np.full(epochs//10, momentum[0]),
                               ])

    for epoch in tqdm(range(len(lr))):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr[epoch]
            param_group['momentum'] = momentum[epoch]
        train_loop(m, train_dat, loss_fn, optimizer)
    return m


def eval_nn(model):
    """Eval function for min_move_dead strategy:

    Runs mcts for 1000 games,
    Terminates search when 20 games die (1/50),
    Returns average move count for those 20 games.

    This only approximates the true distribution:
    sample minimum is distributed 1-(1-dist(x))^n,
    so ideally I would need to estimate the entire
    distribution before transforming and finding mean.
    """
    model.eval()

    origin = start_board.copy()
    origin.score = 0
    games = [origin.copy() for _ in range(2000)]
    notdead = games.copy()

    counter = 0
    scores = []
    alive = 2000
    with torch.no_grad():
        while True:
            for i in range(4):
                if i == 0:
                    subgames = notdead
                    boards = [g.board for g in subgames]
                    preds = model.forward(torch.stack(boards).cuda().float().unsqueeze(1))
                    preds = torch.argsort(preds.cpu(), dim=1, descending=True)
                    for g, p in zip(subgames, preds):
                        g.pred = p
                    moves = preds[:, i]
                else:
                    subgames = [g for g in notdead if not g.moved]
                    moves = [g.pred[i] for g in subgames]
                Board.move_batch(subgames, moves)
            notdead = [g for g in notdead if g.moved]
            if len(notdead) < alive:
                scores.extend([counter] * (alive - len(notdead)))
                alive = len(notdead)
            if alive <= 0.98*2000:
                break
            Board.generate_tile_batch(notdead)
            counter += 1
    return scores[:40]


def eval_fn(params):
    model = train(params)
    t = time()
    scores = eval_nn(model)
    t = time() - t
    mu = np.mean(scores)
    # sig = np.std(scores)/np.sqrt(20)
    # sig is not a good estimator of sem
    print('{0:.1f} / {1:.0f} sec'.format(mu, t))
    return mu  # {'min_move': (mu, sig)}


if __name__ == '__main__':
    parameters = {
        'lr': 0.01,
        'decay': 0.0005,
        'epochs': 10,
    }
    print(eval_fn(parameters))
