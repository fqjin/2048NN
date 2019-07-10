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


def main(t_tuple, v_tuple, epochs, lr, batch_size=256, momentum=0.9, decay=1e-4, save_period=50, pretrained=None):
    start, end = t_tuple
    logname = '{}_{}_epox{}_lr{}'.format(start, end, epochs, lr)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_set = GameDataset('selfplay/', start, end, device, augment=False)
    valid_set = GameDataset('selfplay/', v_tuple[0], v_tuple[1], device, augment=False)
    train_dat = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_dat = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    m = ConvNet(channels=32, num_blocks=4)
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


if __name__ == '__main__':
    # main(t_tuple=(10, 60), v_tuple=(0, 10), lr=0.125, epochs=60, save_period=10, batch_size=512, decay=0.0012)
    main(t_tuple=(60, 80), v_tuple=(0, 10), lr=0.0043, epochs=10, save_period=10,
         batch_size=512, decay=0.0012, pretrained='20190625/10_60_epox60_lr0.125_e49')
    pass
