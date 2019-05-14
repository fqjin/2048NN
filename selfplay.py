from argparse import ArgumentParser
from board import *
from mcts_nn import *
from network import *

parser = ArgumentParser()
parser.add_argument('--num', type=int, default=12345,
                    help='Number to use as seed and save name')
parser.add_argument('--verbose', type=int, default=0,
                    help='Verbose output. Default 0.')
args = parser.parse_args()

s = args.num
random.seed(s)
np.random.seed(s)
torch.manual_seed(s)

a = Board(device='cpu', draw=True)

m = Fixed()
# m = ConvNet()

selfplay(s, m, a, number=50, verbose=args.verbose)
