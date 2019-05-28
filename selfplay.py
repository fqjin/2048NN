from argparse import ArgumentParser
from board import *
from mcts_nn import *
from network import *

parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=12345,
                    help='Number to use as seed and save name')
parser.add_argument('--verbose', type=int, default=0,
                    help='Verbose output. Default 0.')
args = parser.parse_args()

s = args.seed
random.seed(s)
np.random.seed(s)
torch.manual_seed(s)

a = Board(device='cpu', draw=True)

# m = Fixed()
name = '0_10_epox100_lr0.1_e99'
print('Using model: {}'.format(name))
m = ConvNet()
m.load_state_dict(torch.load('models/{}.pt'.format(name)))
m.to('cuda')

selfplay(s, m, a, number=50, verbose=args.verbose)
