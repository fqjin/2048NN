import numpy as np
import torch
from torch.utils.data import Dataset


class GameDataset(Dataset):
    def __init__(self, gamepath, start, end, device='cpu'):
        self.boards = []
        self.results = []
        for i in range(start, end):
            game = np.load(gamepath+str(i).zfill(5)+'.npz')
            self.boards.append(game['boards'])  # uint64
            self.results.extend(game['results'])  # float32
        self.boards = np.concatenate(self.boards)
        data = []
        for _ in range(16):
            data.append(self.boards & 0xF)
            self.boards >>= 4
        self.boards = torch.tensor(data,
                                   dtype=torch.float32,
                                   device=device).transpose(0, 1)
        self.results = torch.tensor(self.results,
                                    dtype=torch.float32,
                                    device=device)
        self.results = torch.softmax(self.results, dim=1)

    def __len__(self):
        return self.results.size(0)

    def __getitem__(self, index):
        return self.boards[index], self.results[index]


class ConvGameDataset(GameDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.boards = self.boards.view(-1, 1, 4, 4)


class OneHotConvGameDataset(GameDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        onehot = [self.boards == i for i in range(16)]
        onehot = torch.stack(onehot, dim=1)
        self.boards = onehot.float().view(-1, 16, 4, 4)
