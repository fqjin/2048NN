from argparse import ArgumentParser
from board import *
from mcts_batch import *
# from mcts_nn import *
# from network import *


def selfplay_fixed(name, game, number=50, verbose=False):
    """Plays through one game using fixed. Returns all
    boards and move choices of the main line for training.

    Args:
        name (int): name for data
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
    if isinstance(name, int):
        name = str(name).zfill(5)
    np.savez('selfplay/'+name, boards=torch.stack(boards), moves=moves, score=game.score,
             method=0)  # method 0 is fixed
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

    a = Board(device='cpu', draw=True)

    # name = '0_10_epox100_lr0.1_e99'
    name = 'fixed'
    print('Using model: {}'.format(name))
    # m = ConvNet()
    # m.load_state_dict(torch.load('models/{}.pt'.format(name)))
    # m.to('cuda')

    # selfplay(s, m, a, number=50, verbose=args.verbose)
    selfplay_fixed(s, a, number=50, verbose=args.verbose)