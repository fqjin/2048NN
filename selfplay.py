from argparse import ArgumentParser
from board import *  # os, np, torch, board, CONSTANTS
from mcts_batch import mcts_fixed_batch
from mcts_nn import mcts_nn, mcts_nn_min
from network import *


def selfplay_fixed(name, game, number=50, verbose=False):
    """Plays through one game using fixed. Returns all
    boards and move choices of the main line for training.

    Args:
        name (str): name for data
        game (Board): the starting game state. If `None`
            is passed, a new Board is generated.
        number (int): # of lines to try for each move.
            Defaults to 10
        verbose (bool): whether to print anything
            Defaults to False

    Returns:
        boards: list of boards
        moves: list of mcts moves

    """
    if not game:
        game = Board(gen=True, draw=verbose)
    boards = []
    moves = []
    while True:
        if not len(moves) % 20:
            print('Move {}'.format(len(moves)))
        boards.append(game.board.clone())
        pred = mcts_fixed_batch(game, number=number)
        # Only need to do argmax. If not possible, game is dead
        i = np.argmax(pred)
        if game.move(i):
            game.generate_tile()
            moves.append(i)
            if verbose:
                os.system(CLEAR)
                print('Seed {}'.format(name))
                print(pred)
                print(ARROWS[i.item()])
                game.draw()
            continue
        else:
            boards.pop()
            break
    print(game.score)
    print('Game Over')
    print('{} moves'.format(len(moves)))
    np.savez('selfplay/'+name,
             boards=torch.stack(boards),
             moves=moves,
             score=game.score,
             method=0)  # method 0 is fixed
    print('Saved as {}.npz'.format(name))


def selfplay(name, model, game, number=10, device='cpu',
             verbose=False, mcts_min=False):
    """Plays through one game using mcts_nn. Returns all
    boards and move choices of the main line for training.

    Args:
        name (str): name for data
        game (Board): the starting game state. If `None`
            is passed, a new Board is generated.
        model: model to predict moves
        number (int): # of lines to try for each move.
            Defaults to 10
        device: torch device. Defaults to 'cpu'
        verbose (bool): whether to print anything
            Defaults to False
        mcts_min (bool): whether to use conservative
            strategy, max move number before single death.

    Returns:
        boards: list of boards
        moves: list of mcts_nn moves

    """
    print('Seed {}'.format(name))
    if not game:
        game = Board(gen=True, draw=verbose, device=device)
    boards = []
    moves = []
    while True:
        if verbose and not len(moves) % 20:
            print('Move {}'.format(len(moves)))
        boards.append(game.board.clone())
        if mcts_min:
            pred = mcts_nn_min(model, game, number=number)
        else:
            pred = mcts_nn(model, game, number=number)
        # Only need to do argmax. If not possible, game is dead
        i = np.argmax(pred)
        if game.move(i):
            game.generate_tile()
            moves.append(i)
            if verbose:
                os.system(CLEAR)
                print('Seed {}'.format(name))
                print(pred)
                print(ARROWS[i.item()])
                game.draw()
            continue
        else:
            boards.pop()
            break
    print('Game Over')
    game.draw()
    print('{} moves'.format(len(moves)))
    if mcts_min:
        name = 'min_move_dead/min' + name
    np.savez('selfplay/'+name,
             boards=torch.stack(boards),
             moves=moves,
             score=game.score)
    print('Saved as {}.npz'.format(name))


if __name__ == '__main__':
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
    name = str(s).zfill(5)

    m_name = '20190815/0_600_epox30_clr0.01_ex'
    print('Using model: {}'.format(m_name))
    m = ConvNet(channels=32, num_blocks=5)
    m.load_state_dict(torch.load('models/{}.pt'.format(m_name)))
    m.to('cuda')
    # m = Fixed()

    a = Board(device='cpu', draw=True)

    selfplay(name, m, a, number=50, verbose=args.verbose, mcts_min=True)
