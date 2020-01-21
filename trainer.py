import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from game_dataset import GameDataset
from network import DenseNet
from eval_nn import eval_nn


def train_loop(model, data, loss_fn, optimizer):
    model.train()
    running_loss = 0
    for x, y in tqdm(data):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        running_loss += loss.data.item()
        loss.backward()
        optimizer.step()
    running_loss /= len(data)
    print('Train Loss: {:.3f}'.format(running_loss))
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
    print('Valid Loss: {:.3f}'.format(running_loss))
    return running_loss


def main(t_tuple,
         v_tuple,
         epochs,
         lr,
         batch_size=256,
         momentum=0.9,
         decay=1e-5,
         stopping=None,
         pretrained=None,
         path='selfplay/',
         net_params=None,
         ):
    if stopping is None:
        stopping = epochs
    if net_params is None:
        net_params = dict(channels=32, blocks=5)
    logname = '{}_{}_c{}b{}_lr{}'.format(t_tuple[0], t_tuple[1], net_params['channels'], net_params['blocks'], lr)
    print(logname)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))
    train_set = GameDataset(path, t_tuple[0], t_tuple[1], device)
    valid_set = GameDataset(path, v_tuple[0], v_tuple[1], device)
    train_dat = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_dat = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    m = DenseNet(**net_params)
    if pretrained is not None:
        m.load_state_dict(torch.load('models/{}.pt'.format(pretrained)))
        print('Loaded ' + pretrained)
        logname += 'pre'
    m.to(device)
    torch.save(m.state_dict(), 'models/'+logname+'_e0.pt')
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    # optimizer = torch.optim.SGD(m.parameters(),
    #                             lr=lr,
    #                             momentum=momentum,
    #                             nesterov=True,
    #                             weight_decay=decay)
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    t_loss = []
    v_loss = []
    log_scores = []
    best = 0.0
    timer = stopping
    for epoch in range(epochs):
        print('-' * 10)
        print('Epoch: {}'.format(epoch))
        t_loss.append(train_loop(m, train_dat, loss_fn, optimizer))
        v_loss.append(valid_loop(m, valid_dat, loss_fn))
        mean_ls, max_s, mean_m = eval_nn(m, number=200, device=device)
        log_scores.append(mean_ls)
        print(mean_ls, max_s, mean_m)
        if mean_ls >= best:
            best = mean_ls
            timer = stopping
            torch.save(m.state_dict(), 'models/'+logname+'_e{}.pt'.format(epoch))
        elif timer > 0:
            timer -= 1
        else:
            break

    params = {
        't_tuple': t_tuple,
        'v_tuple': v_tuple,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        # 'decay': decay,
        # 'momentum': momentum,
        'channels': net_params['channels'],
        'blocks': net_params['blocks'],
        'pretrained': pretrained
    }
    np.savez('logs/'+logname, 
             t_loss=t_loss, 
             v_loss=v_loss, 
             log_scores=log_scores, 
             params=params)


if __name__ == '__main__':
    main(t_tuple=(20, 200), v_tuple=(0, 20),
         batch_size=2048,
         epochs=40, stopping=20,
         lr=0.01, decay=1e-5,
         path='selfplay/fixed/fixed',
         net_params=dict(channels=64, blocks=5)
         )
