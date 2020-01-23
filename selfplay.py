from argparse import ArgumentParser
from time import time
from board import *
from mcts import mcts_fixed
from mcts_nn import mcts_nn
from network import ConvNet


def selfplay_fixed(name, board=None, number=10, verbose=False):
    """Plays through one game using fixed. Returns all
    boards and move choices of the main line for training.

    Args:
        name (str): name for data
        board: the starting game state. Defaults to new board
        number (int): # of lines to try for each move.
            Defaults to 10
        verbose (bool): whether to print anything
            Defaults to False

    Returns:
        boards: list of boards
        scores: list of mcts scores
    """
    if not board:
        board = generate_init_tiles()
    score = 0
    if verbose:
        draw(board, score)

    boards = []
    results = []
    while True:
        result = mcts_fixed(board, number)
        i = np.argmax(result)
        f, s, m = move(board, i)
        if not m:
            break
        boards.append(board)
        results.append(result)

        board = generate_tile(f)
        score += s
        if verbose:
            os.system(CLEAR)
            print(ARROWS[i])
            print('[{:.2f} {:.2f} {:.2f} {:.2f}]'.format(*result))
            draw(board, score)
    print('Score {}'.format(score))
    print('{} moves'.format(len(boards)))
    np.savez('selfplay/fixed' + name,
             boards=np.array(boards, dtype=np.uint64),
             results=np.array(results, dtype=np.float32),
             score=score)
    print('Saved as fixed{}.npz'.format(name))


def selfplay(name, model, board=None, number=10, device='cuda', verbose=False):
    """Plays through one game using mcts_nn. Returns all
    boards and move choices of the main line for training.

    Args:
        name (str): name for data
        model: torch model to predict moves
        board (int64): the starting game state.
             Default a new Board is generated.
        number (int): # of lines to try for each move.
            Defaults to 10
        device: torch device. Defaults to 'cuda'
        verbose (bool): whether to print anything
            Defaults to False

    Returns:
        boards: list of boards
        moves: list of mcts_nn moves

    """
    start_time = time()
    print('Seed {}'.format(name))
    if not board:
        board = generate_init_tiles()
    score = 0
    if verbose:
        draw(board, score)

    boards = []
    results = []
    while True:
        result = mcts_nn(model, board, number, device)
        i = np.argmax(result)
        f, s, m = move(board, i)
        if not m:
            break
        boards.append(board)
        results.append(result)

        board = generate_tile(f)
        score += s
        if verbose:
            os.system(CLEAR)
            print(ARROWS[i])
            print('[{:.2f} {:.2f} {:.2f} {:.2f}]'.format(*result))
            draw(board, score)
    print('Score {}'.format(score))
    print('{} moves'.format(len(boards)))
    np.savez('selfplay/' + name,
             boards=np.array(boards, dtype=np.uint64),
             results=np.array(results, dtype=np.float32),
             score=score)
    print('Saved {}.npz'.format(name))
    total_time = time() - start_time
    print(f'Took {total_time//60} minutes')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=12345,
                        help='Number to use as seed and save name')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbose output. Default 0')
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    name = str(seed).zfill(5)

    m_name = '20200123/onehot_20_200_c128b5_p20_bs2048lr0.01d0.0_s2_best'
    print('Using model: {}'.format(m_name))
    model = ConvNet(channels=128, blocks=5)
    model.load_state_dict(torch.load('models/{}.pt'.format(m_name)))
    model.to('cuda')

    # selfplay_fixed(name, number=50, verbose=args.verbose)
    selfplay(name, model, number=50, verbose=args.verbose)
