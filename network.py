import torch
import torch.nn as nn
from board import SIZE, SIZE_SQRD


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.layer = nn.Linear(SIZE_SQRD, SIZE)
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, SIZE_SQRD)
        x = self.layer(x)
        x = self.soft(x)
        return x


if __name__ == '__main__':
    m = TestNet()
    params = sum(p.numel() for p in m.parameters())
    print(params)
