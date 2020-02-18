import argparse
import numpy as np
import random
import torch
from time import time
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from game_dataset import OneHotConvGameDataset
from network import ConvNet
from eval_nn import eval_nn_min


def main(args):
    if args.name:
        args.name += '_'
    logname = f'{args.name}{args.t_tuple[0]}_{args.t_tuple[1]}_soft{args.soft}' \
              f'c{args.channels}b{args.blocks}_p{args.patience}_' \
              f'bs{args.batch_size}lr{args.lr}d{args.decay}_s{args.seed}'
    print(logname)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))
    torch.backends.cudnn.benchmark = True

    train_set = OneHotConvGameDataset(args.path, args.t_tuple[0], args.t_tuple[1], device, soft=args.soft)
    train_dat = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    m = ConvNet(channels=args.channels, blocks=args.blocks)
    if args.pretrained:
        m.load_state_dict(torch.load('models/{}.pt'.format(args.pretrained), map_location=device))
        print('Loaded ' + args.pretrained)
        logname = 'pre_'+logname
    m.to(device)
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(m.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.decay)
    t_loss = []
    min_move = []
    best = 0.0
    timer = 0
    if args.patience == 0:
        stop = args.epochs
    else:
        stop = args.patience

    data_len = len(train_dat)
    for epoch in range(args.epochs):
        print('-' * 10)
        print('Epoch: {}'.format(epoch))
        timer += 1

        m.train()
        running_loss = 0
        for x, y in tqdm(train_dat):
            optimizer.zero_grad()
            pred = m(x)
            loss = loss_fn(pred, y)
            running_loss += loss.data.item()
            loss.backward()
            optimizer.step()
        running_loss /= data_len
        if epoch == 2 and running_loss > 210/1000:
            stop = 0
        print('Train mLoss: {:.3f}'.format(1e3 * running_loss))
        t_loss.append(running_loss)
        
        m.eval()
        time1 = time()
        ave_min_move = eval_nn_min(m, number=10, repeats=40, device=device)
        time_str = ', took {:.0f} seconds'.format(time()-time1)
        min_move.append(ave_min_move)
        if ave_min_move >= best:
            tqdm.write(str(ave_min_move) + ' ** Best' + time_str)
            best = ave_min_move
            timer = 0
            torch.save(m.state_dict(), 'models/' + logname + '_best.pt')
        else:
            tqdm.write(str(ave_min_move) + time_str)

        if timer >= stop:
            print('Ran out of patience')
            print(f'Best score: {best}')
            # torch.save(m.state_dict(), 'models/'+logname+f'_e{epoch}.pt')
            break
        else:
            print(f'{stop - timer} epochs remaining')

    np.savez('logs/'+logname,
             t_loss=t_loss,
             min_move=min_move,
             params=args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--path', type=str, default='selfplay/',
                   help='path to training data with prefix')
    p.add_argument('--t_tuple', type=int, nargs=2, default=(20, 200),
                   help='tuple for training data range')
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--blocks', type=int, default=3)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--patience', type=int, default=10,
                   help='Early stopping based on log score eval. '
                        'If zero, no early stopping.')
    p.add_argument('--batch_size', type=int, default=2048)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--decay', type=float, default=0.0)
    p.add_argument('--soft', type=float, default=3.0)
    p.add_argument('--name', type=str, default='',
                   help='Additional prepend output name')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pretrained', type=str, default='',
                   help='Path to network to continue training')

    main(p.parse_args())
