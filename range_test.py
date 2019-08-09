import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from game_dataset import GameDataset
from network import ConvNet


def train_loop(model, x, y, loss_fn, optimizer):
    model.train()
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    return loss.data.item()


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


def range_test(t_tuple,
               v_tuple,
               steps,
               lr_tuple,
               mom_tuple,
               batch_size=256,
               decay=1e-4,
               pretrained=None,
               path='selfplay/',
               net_params=None,
               ):
    params = locals()
    if net_params is None:
        net_params = dict(channels=32, num_blocks=4)
    start, end = t_tuple
    logname = 'range_test_b{}_d{}'.format(batch_size, decay)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_set = GameDataset(path, start, end, device, augment=False)
    valid_set = GameDataset(path, v_tuple[0], v_tuple[1], device, augment=False)
    train_dat = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_dat = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    lr = np.geomspace(lr_tuple[0], lr_tuple[1], steps)
    momentum = np.linspace(mom_tuple[0], mom_tuple[1], steps)

    m = ConvNet(**net_params)
    if pretrained is not None:
        m.load_state_dict(torch.load('models/{}.pt'.format(pretrained)))
        print('Loaded ' + pretrained)
        logname += 'pre'
    m.to(device)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(m.parameters(),
                                lr=lr[0],
                                momentum=momentum[0],
                                nesterov=True,
                                weight_decay=decay)
    t_loss = []
    v_loss = []
    train_iter = iter(train_dat)
    for step in tqdm(range(steps)):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr[step]
            param_group['momentum'] = momentum[step]
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dat)
            x, y = next(train_iter)
        t_loss.append(train_loop(m, x, y, loss_fn, optimizer))
        v_loss.append(valid_loop(m, valid_dat, loss_fn))
        if step%20 == 0:
            print('LR: {0:.4f}, Train loss: {1:.3f}, Valid loss: {2:.3f}'.format(
                lr[step], t_loss[-1], v_loss[-1]))

    np.savez('logs/'+logname, t_loss=t_loss, v_loss=v_loss, lr=lr, params=params)


if __name__ == '__main__':
    range_test(t_tuple=(20, 100), v_tuple=(0, 20),
               lr_tuple=(0.01, 2.0),
               mom_tuple=(0.95, 0.85),
               batch_size=2048,
               steps=500,
               decay=1e-6,
               path='selfplay/min_move_dead/min',
               net_params=dict(channels=32, num_blocks=5)
               )
