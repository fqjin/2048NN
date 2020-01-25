import torch
import torch.nn as nn


class FixedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = torch.tensor([3.0, 2.0, 0.0, 1.0], dtype=torch.float32)
        self.output = nn.LogSoftmax(dim=0)(self.output)

    def forward(self, x):
        b = x.size(0)
        return self.output.repeat(b, 1)


class DenseNet(nn.Module):
    """Feedforward fully connected network"""
    def __init__(self, channels=16, blocks=0):
        super().__init__()
        self.in_block = nn.Sequential(
            nn.Linear(16, channels, bias=True),
            nn.ReLU(inplace=True)
        )
        mid = []
        for _ in range(blocks):
            mid.append(nn.Linear(channels, channels, bias=True))
            mid.append(nn.ReLU(inplace=True))
        self.mid_block = nn.Sequential(
            *mid
        )
        self.out_block = nn.Sequential(
            nn.Linear(channels, 4, bias=True),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.in_block(x)
        x = self.mid_block(x)
        x = self.out_block(x)
        return x


class ConvNet(nn.Module):
    """Residual convolutional network

    Args:
        channels: Defaults to 16 (256 in paper)
        blocks: Defaults to 5 (19 in paper)
        out_c: Defaults to 4 (2 in paper)
    """
    def __init__(self, channels=16, blocks=5, out_c=4):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.in_block = nn.Sequential(
            nn.Conv2d(16, channels, 3, padding=1, bias=False),
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
            ) for _ in range(blocks)
        ])
        self.out_block = nn.Sequential(
            # I use 4 output channels (2 in paper)
            nn.Conv2d(channels, out_c, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.out_size = out_c * 16
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


class SepConvNet(nn.Module):
    """Lighter weight convnet

    Args:
        channels: Defaults to 128
        blocks: Defaults to 5
        out_c: Defaults to 4
    """
    def __init__(self, channels=128, blocks=5, out_c=4):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.in_block = nn.Sequential(
            nn.Conv2d(16, channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, groups=channels, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels//4, 1, padding=0, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels//4, channels, 1, padding=0, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, groups=channels, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels//4, 1, padding=0, bias=False),
                nn.BatchNorm2d(channels//4),
            ) for _ in range(blocks)
        ])
        self.out_block = nn.Sequential(
            nn.Conv2d(channels//4, out_c, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.out_size = out_c * 16
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
    for m in [FixedNet(),  # 0
              DenseNet(channels=64, blocks=5),  # 22148
              ConvNet(channels=128, blocks=5),  # 1496588
              SepConvNet(channels=128, blocks=5),  # 59404
              ]:
        params = sum(p.numel() for p in m.parameters())
        print(params)
