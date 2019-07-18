import numpy as np
import torch
from torch.utils.data import Dataset


class GameDataset(Dataset):
    def __init__(self, gamepath, start, end, device='cpu', augment=False):
        # self.gameptah = gamepath
        # self.start = start
        # self.end = end
        # self.device = device
        # self.augment = augment
        self.boards = []
        self.moves = []
        for i in range(start, end):
            game = np.load(gamepath+str(i).zfill(5)+'.npz')
            self.boards.append(game['boards'])
            self.moves.extend(game['moves'])
        self.boards = np.concatenate(self.boards)
        self.boards = torch.from_numpy(self.boards)
        self.boards = self.boards.to(device).float()
        self.boards = self.boards.unsqueeze(1)
        self.moves = torch.tensor(self.moves,
                                  dtype=torch.long,
                                  device=device)

    def __len__(self):
        return self.moves.size(0)

    def __getitem__(self, index):
        return self.boards[index], self.moves[index]
