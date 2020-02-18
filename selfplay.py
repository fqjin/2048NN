from argparse import ArgumentParser
from time import time
from board import *
from mcts import mcts_fixed_min
from mcts_nn import mcts_nn
from network import ConvNet


def selfplay_fixed(name, board=None, number=10, times=1, verbose=False):
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
    print(f'fixed {number} lines, {times} times')
    if not board:
        board = generate_init_tiles()
    score = 0
    if verbose:
        draw(board, score)

    boards = []
    results = []
    while True:
        result = np.zeros(4)
        for _ in range(times):
            result += mcts_fixed_min(board, number)
        result /= times
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
            print('[{:.1f} {:.1f} {:.1f} {:.1f}]'.format(*result))
            draw(board, score)
    print('Score {}'.format(score))
    print('{} moves'.format(len(boards)))
    np.savez('selfplay/fixed10/' + name,
             boards=np.array(boards, dtype=np.uint64),
             results=np.array(results, dtype=np.float32),
             score=score)
    print('Saved as {}.npz'.format(name))


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
    print('Seed {}'.format(name))
    if not board:
        board = generate_init_tiles()
    score = 0
    if verbose:
        draw(board, score)

    boards = []
    results = []
    model.eval()
    with torch.no_grad():
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
                print('[{:.1f} {:.1f} {:.1f} {:.1f}]'.format(*result))
                draw(board, score)
    print('Score {}'.format(score))
    print('{} moves'.format(len(boards)))
    np.savez('selfplay/' + name,
             boards=np.array(boards, dtype=np.uint64),
             results=np.array(results, dtype=np.float32),
             score=score)
    print('Saved {}.npz'.format(name))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=12345,
                        help='Number to use as seed and save name')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()

    seed = args.seed
    if seed < 0:
        raise ValueError('Seed is {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    name = str(seed).zfill(5)

    start_time = time()
    use_network = False
    if not use_network:
        selfplay_fixed(name, number=10, times=4, verbose=args.verbose)
    else:
        torch.backends.cudnn.benchmark = True
        m_name = 'models/20200207/best_s7_jit.pth'
        if not os.path.exists(m_name):
            full_name = '20200207/0_1000_soft3.0c64b3_p10_bs2048lr0.1d0.0_s7_best'
            print('Tracing: {}'.format(full_name))
            model = ConvNet(channels=64, blocks=3)
            model.load_state_dict(torch.load('models/{}.pt'.format(full_name)))
            model.to('cuda')
            model.eval()
            model = torch.jit.trace(model, torch.randn(10, 16, 4, 4, dtype=torch.float32, device='cuda'))
            torch.jit.save(model, m_name)
            print('Jit model saved')
        else:
            print(m_name)
            model = torch.jit.load(m_name)
            selfplay(name, model, number=10, times=4, verbose=args.verbose)

    total_time = time() - start_time
    print(f'Took {total_time/60:.1f} minutes')
