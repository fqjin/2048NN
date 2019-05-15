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
    print('Loss: {:.3f}'.format(running_loss))
    return running_loss


def main(start, end, epochs, lr, batch_size=256, momentum=0.9, decay=1e-4):
    logname = '{}_{}_epox{}_lr{}'.format(start, end, epochs, lr)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = GameDataset('selfplay/', start, end, device, augment=False)
    data = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    m = ConvNet()
    m.to(device)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(m.parameters(),
                                lr=lr,
                                momentum=momentum,
                                nesterov=True,
                                weight_decay=decay)
    loss = []
    for epoch in range(epochs):
        if epoch % 99 == 0:
            torch.save(m.state_dict(), 'models/'+logname+'.pt')
        print('-' * 10)
        print('Epoch: {}'.format(epoch))
        loss.append(train_loop(m, data, loss_fn, optimizer))

    params = {
        'start': start,
        'end': end,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'decay': decay,
        'momentum': momentum,
    }
    np.savez('logs/'+logname, loss=loss, params=params)


if __name__ == '__main__':
    main(0, 10, epochs=100, lr=1.0)
