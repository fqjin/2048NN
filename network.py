import torch
import torch.nn as nn


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(16, 4)
        self.log_soft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 16)
        x = self.layer(x)
        x = self.log_soft(x)
        return x


class DenseNet(nn.Module):
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
        x = x.view(-1, 16)
        x = self.in_block(x)
        x = self.mid_block(x)
        x = self.out_block(x)
        return x


# class ConvNet(nn.Module):
#     """Architecture based on AlphaZero Paper
#
#     Args:
#         channels: Defaults to 16 (256 in paper)
#         num_blocks: Defaults to 6 (19 in paper)
#         out_c: Defaults to 4 (2 in paper)
#
#     """
#     def __init__(self, channels=16, num_blocks=6, out_c=4):
#         super().__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.in_block = nn.Sequential(
#             nn.Conv2d(1, channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True)
#         )
#         self.blocks = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(channels, channels, 3, padding=1, bias=False),
#                 nn.BatchNorm2d(channels),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(channels, channels, 3, padding=1, bias=False),
#                 nn.BatchNorm2d(channels)
#             ) for _ in range(num_blocks)
#         ])
#         self.out_block = nn.Sequential(
#             # I use 4 output channels (2 in paper)
#             nn.Conv2d(channels, out_c, 1, padding=0, bias=False),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True)
#         )
#         self.out_size = out_c * SIZE_SQRD
#         self.policy = nn.Linear(self.out_size, 4)
#         self.soft = nn.LogSoftmax(dim=1)
#
#     def forward(self, x):
#         x = self.in_block(x)
#         for block in self.blocks:
#             x = x + block(x)
#             x = self.relu(x)
#         x = self.out_block(x)
#         x = x.view(-1, self.out_size)
#         x = self.policy(x)
#         x = self.soft(x)
#         return x


if __name__ == '__main__':
    m = DenseNet(channels=32, blocks=5)
    params = sum(p.numel() for p in m.parameters())
    print(params)
    # 5956
