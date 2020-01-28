from argparse import ArgumentParser
from time import time
from board import *
from mcts import mcts_fixed_min
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
        result = mcts_fixed_min(board, number)
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
    if seed < 400:
        raise ValueError('Seed is {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    name = str(seed).zfill(5)

    torch.backends.cudnn.benchmark = True
    # m_name = '20200128/20_400_soft3.5c64b3_p10_bs2048lr0.08d0.0_s2pre_best'
    # print('Using model: {}'.format(m_name))
    # model = ConvNet(channels=64, blocks=3)
    # model.load_state_dict(torch.load('models/{}.pt'.format(m_name)))
    # model.to('cuda') 
    # model.eval()
    # model = torch.jit.trace(model, torch.randn(50, 16, 4, 4, dtype=torch.float32, device='cuda'))
    # torch.jit.save(model, 'models/20200128/best_s2_jit.pth')
    # print('Jit traced model saved')
    # raise SystemExit
    m_name = '20200128/best_s2_jit.pth'
    print(m_name)
    model = torch.jit.load('models/' + m_name)

    start_time = time()
    # selfplay_fixed(name, number=50, verbose=args.verbose)
    selfplay(name, model, number=50, verbose=args.verbose)
    total_time = time() - start_time
    print(f'Took {total_time/60:.1f} minutes')
