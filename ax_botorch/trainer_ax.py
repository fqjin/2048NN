import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

sys.path.append('../')
from game_dataset import GameDataset
from network import ConvNet

t_tuple = (10, 60)
v_tuple = (0, 10)
epochs = 60
batch_size = 512
momentum = 0.9

os.chdir('..')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_set = GameDataset('selfplay/', t_tuple[0], t_tuple[1], device, augment=False)
valid_set = GameDataset('selfplay/', v_tuple[0], v_tuple[1], device, augment=False)
train_dat = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_dat = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
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


def valid_loop(model, data, loss_fn):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for x, y in data:
            pred = model(x)
            loss = loss_fn(pred, y)
            running_loss += loss.data.item()
    running_loss /= len(data)
    return running_loss


def eval_fn(params):
    channels = 2**params['channels']
    blocks = params['blocks']
    lr = params['lr']
    decay = params['decay']

    m = ConvNet(channels=channels, num_blocks=blocks)
    m.to(device)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(m.parameters(),
                                lr=lr,
                                momentum=momentum,
                                nesterov=True,
                                weight_decay=decay)
    t_loss = []
    v_loss = []
    for epoch in tqdm(range(epochs)):
        t_loss.append(train_loop(m, train_dat, loss_fn, optimizer))
        v_loss.append(valid_loop(m, valid_dat, loss_fn))

    min_val = np.min(v_loss)
    print(min_val)
    return min_val
    # return {'min_val': (min_val, 0.0)}


if __name__ == '__main__':
    parameters = {
        'channels': 1,
        'blocks': 7,
        'lr': 0.1,
        'decay': 1e-3,
    }
    mv = eval_fn(parameters)
