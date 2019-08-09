import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from game_dataset import GameDataset
from network import ConvNet


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
    print('Imp Acc: {:.3f}'.format(np.exp(-1*running_loss)))
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
    print('Imp Acc: {:.3f}'.format(np.exp(-1*running_loss)))
    return running_loss


def main(t_tuple,
         v_tuple,
         epochs,
         lr,
         batch_size=256,
         momentum=0.9,
         decay=1e-4,
         save_period=50,
         pretrained=None,
         path='selfplay/',
         net_params=None,
         ):
    if net_params is None:
        net_params = dict(channels=32, num_blocks=4)
    start, end = t_tuple
    logname = '{}_{}_epox{}_lr{}'.format(start, end, epochs, lr)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_set = GameDataset(path, start, end, device, augment=False)
    valid_set = GameDataset(path, v_tuple[0], v_tuple[1], device, augment=False)
    train_dat = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_dat = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    m = ConvNet(**net_params)
    if pretrained is not None:
        m.load_state_dict(torch.load('models/{}.pt'.format(pretrained)))
        print('Loaded ' + pretrained)
        logname += 'pre'
    m.to(device)
    torch.save(m.state_dict(), 'models/'+logname+'_e0.pt')
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(m.parameters(),
                                lr=lr,
                                momentum=momentum,
                                nesterov=True,
                                weight_decay=decay)
    t_loss = []
    v_loss = []
    for epoch in range(epochs):
        print('-' * 10)
        print('Epoch: {}'.format(epoch))
        t_loss.append(train_loop(m, train_dat, loss_fn, optimizer))
        v_loss.append(valid_loop(m, valid_dat, loss_fn))
        if epoch % save_period == save_period-1:
            torch.save(m.state_dict(), 'models/'+logname+'_e{}.pt'.format(epoch))

    params = {
        't_tuple': t_tuple,
        'v_tuple': v_tuple,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'decay': decay,
        'momentum': momentum,
        'pretrained': pretrained
    }
    np.savez('logs/'+logname, t_loss=t_loss, v_loss=v_loss, params=params)


def cyclic(t_tuple,
           v_tuple,
           epochs,
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
    logname = '{}_{}_epox{}_clr{}'.format(start, end, epochs, lr_tuple[0])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_set = GameDataset(path, start, end, device, augment=False)
    train_dat = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    if v_tuple is not None:
        valid_set = GameDataset(path, v_tuple[0], v_tuple[1], device, augment=False)
        valid_dat = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    lr = np.geomspace(lr_tuple[0], lr_tuple[1], epochs//2)
    lr = np.concatenate([lr,
                         np.flip(lr[:-1]),
                         # np.geomspace(lr[0], lr[0]/10, epochs//10+1)[1:],
                         ])
    momentum = np.linspace(mom_tuple[0], mom_tuple[1], epochs//2)
    momentum = np.concatenate([momentum,
                               np.flip(momentum[:-1]),
                               # np.full(epochs//10, momentum[0]),
                               ])

    m = ConvNet(**net_params)
    if pretrained is not None:
        m.load_state_dict(torch.load('models/{}.pt'.format(pretrained)))
        print('Loaded ' + pretrained)
        logname += 'pre'
    m.to(device)
    torch.save(m.state_dict(), 'models/'+logname+'_e0.pt')
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(m.parameters(),
                                lr=lr[0],
                                momentum=momentum[0],
                                nesterov=True,
                                weight_decay=decay)
    t_loss = []
    v_loss = []
    for epoch in range(len(lr)):
        print('-' * 10)
        print('Epoch: {0}, LR: {1:.3f}'.format(epoch, lr[epoch]))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr[epoch]
            param_group['momentum'] = momentum[epoch]

        t_loss.append(train_loop(m, train_dat, loss_fn, optimizer))
        if v_tuple is not None:
            v_loss.append(valid_loop(m, valid_dat, loss_fn))
    torch.save(m.state_dict(), 'models/'+logname+'_e{}.pt'.format(epoch))

    np.savez('logs/'+logname, t_loss=t_loss, v_loss=v_loss, lr=lr, params=params)


if __name__ == '__main__':
    # main(t_tuple=(20, 100), v_tuple=(0, 20),
    #      batch_size=512,
    #      epochs=50, save_period=50,
    #      lr=0.28, decay=5.5e-4,
    #      path='selfplay/min_move_dead/min',
    #      net_params=dict(channels=32, num_blocks=5)
    #      )
    epox = 69
    cyclic(t_tuple=(20, 100), v_tuple=(0, 20),
           lr_tuple=(0.06, 1.0),
           mom_tuple=(0.95, 0.85),
           batch_size=1024,
           epochs=epox,
           decay=1.7e-3,
           path='selfplay/min_move_dead/min',
           net_params=dict(channels=32, num_blocks=5)
           )
