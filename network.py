import torch
import torch.nn as nn
from board import SIZE, SIZE_SQRD


class Fixed:
    def __init__(self):
        self.out = torch.tensor([3.0, 2.0, 1.0, 0.0])

    def eval(self):
        pass

    def forward(self, x):
        b = x.size(0)
        return self.out.repeat(b, 1)


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.layer = nn.Linear(SIZE_SQRD, 4)
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, SIZE_SQRD)
        x = self.layer(x)
        x = self.soft(x)
        return x


class ConvNet(nn.Module):
    """Architecture based on AlphaZero Paper

    Args:
        channels: Defaults to 16 (256 in paper)
        num_blocks: Defaults to 6 (19 in paper)
        out_c: Defaults to 4 (2 in paper)

    """
    def __init__(self, channels=16, num_blocks=6, out_c=4):
        super(ConvNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.in_block = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels)
            ) for _ in range(num_blocks)
        ])
        self.out_block = nn.Sequential(
            # I use 4 output channels (2 in paper)
            nn.Conv2d(channels, out_c, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.out_size = out_c * SIZE_SQRD
        self.policy = nn.Linear(self.out_size, 4)
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.in_block(x)
        for block in self.blocks:
            x = x + block(x)
            x = self.relu(x)
        x = self.out_block(x)
        x = x.view(-1, self.out_size)
        x = self.policy(x)
        x = self.soft(x)
        return x


if __name__ == '__main__':
    m = ConvNet()
    params = sum(p.numel() for p in m.parameters())
    print(params)
    # 28540
