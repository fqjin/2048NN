import os
import sys
import numpy as np
import torch
from time import time
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('../')
from board import Board
from game_dataset import GameDataset
from network import ConvNet

start_board = Board()

t_tuple = (100, 120)
# validation not used
batch_size = 512
momentum = 0.9

os.chdir('..')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_set = GameDataset('selfplay/', t_tuple[0], t_tuple[1], device, augment=False)
train_dat = DataLoader(train_set, batch_size=batch_size, shuffle=True)
os.chdir('ax_botorch')


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

    name = '20190710/80_100_epox10_lr0.0043pre_e9'
    m = ConvNet(channels=32, num_blocks=4)
    m.load_state_dict(torch.load('../models/{}.pt'.format(name)))
    m.to(device)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(m.parameters(),
                                lr=lr,
                                momentum=momentum,
                                nesterov=True,
                                weight_decay=decay)

    for epoch in tqdm(range(epochs)):
        train_loop(m, train_dat, loss_fn, optimizer)
    return m


def eval_nn(model):
    model.eval()

    origin = start_board.copy()
    origin.score = 0
    games = [origin.copy() for _ in range(1000)]
    notdead = games.copy()

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
    return logscores


def eval_fn(params):
    model = train(params)
    t = time()
    logscores = eval_nn(model)
    t = time() - t
    mu = np.mean(logscores)
    sig = np.std(logscores)/np.sqrt(1000)
    print('{0:.3f} / {1:.3f} / {2:.0f} sec'.format(mu, sig, t))
    return {'log_eval': (mu, sig)}


if __name__ == '__main__':
    parameters = {
        'lr': 0.0043,
        'decay': 0.0012,
        'epochs': 10,
    }
    print(eval_fn(parameters))
