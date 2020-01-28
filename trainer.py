import argparse
import numpy as np
import random
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from game_dataset import OneHotConvGameDataset
from network import ConvNet
from eval_nn import eval_nn_min


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
    print('Train mLoss: {:.3f}'.format(1e3*running_loss))
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
    print('Valid mLoss: {:.3f}'.format(1e3*running_loss))
    return running_loss


def main(args):
    if args.name:
        args.name += '_'
    logname = f'{args.name}{args.t_tuple[0]}_{args.t_tuple[1]}_' \
              f'c{args.channels}b{args.blocks}_p{args.patience}_' \
              f'bs{args.batch_size}lr{args.lr}d{args.decay}_s{args.seed}'
    print(logname)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))
    torch.backends.cudnn.benchmark = True

    train_set = OneHotConvGameDataset(args.path, args.t_tuple[0], args.t_tuple[1], device)
    valid_set = OneHotConvGameDataset(args.path, args.v_tuple[0], args.v_tuple[1], device)
    train_dat = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_dat = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

    m = ConvNet(channels=args.channels, blocks=args.blocks)
    if args.pretrained:
        m.load_state_dict(torch.load('models/{}.pt'.format(args.pretrained), map_location=device))
        print('Loaded ' + args.pretrained)
        logname += 'pre'
    m.to(device)
    # torch.save(m.state_dict(), 'models/'+logname+'_e0.pt')
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    # optimizer = torch.optim.SGD(m.parameters(),
    #                             lr=lr,
    #                             momentum=momentum,
    #                             nesterov=True,
    #                             weight_decay=decay)
    optimizer = torch.optim.Adam(m.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.decay)
    t_loss = []
    v_loss = []
    min_move = []
    best = 0.0
    timer = 0
    if args.patience == 0:
        stop = args.epochs
    else:
        stop = args.patience
    for epoch in range(args.epochs):
        print('-' * 10)
        print('Epoch: {}'.format(epoch))
        t_loss.append(train_loop(m, train_dat, loss_fn, optimizer))
        v_loss.append(valid_loop(m, valid_dat, loss_fn))
        ave_min_move = eval_nn_min(m, number=50, repeats=4, device=device)
        min_move.append(ave_min_move)
        print(ave_min_move)
        timer += 1
        if ave_min_move >= best:
            print('** Best')
            best = ave_min_move
            timer = 0
            torch.save(m.state_dict(), 'models/'+logname+'_best.pt')
        elif timer >= stop:
            print('Ran out of patience')
            print(f'Best score: {best}')
            # torch.save(m.state_dict(), 'models/'+logname+f'_e{epoch}.pt')
            break
        else:
            print(f'{stop - timer} epochs remaining')

    np.savez('logs/'+logname,
             t_loss=t_loss,
             v_loss=v_loss, 
             min_move=min_move,
             params=args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--path', type=str, default='selfplay/',
                   help='path to training data with prefix')
    p.add_argument('--t_tuple', type=int, nargs=2, default=(20, 200),
                   help='tuple for training data range')
    p.add_argument('--v_tuple', type=int, nargs=2, default=(0, 20),
                   help='tuple for validation data range')
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--blocks', type=int, default=5)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=0,
                   help='Early stopping based on log score eval. '
                        'If zero, no early stopping.')
    p.add_argument('--batch_size', type=int, default=2048)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--decay', type=float, default=0.0)
    p.add_argument('--name', type=str, default='',
                   help='Additional prepend output name')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pretrained', type=str, default='',
                   help='Path to network to continue training')

    main(p.parse_args())
