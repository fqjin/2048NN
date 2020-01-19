from argparse import ArgumentParser
from board import *  # os, np, torch, board, CONSTANTS
from mcts import mcts_fixed
# from mcts_nn import mcts_nn, mcts_nn_min
# from network import *


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


# def selfplay(name, model, game, number=10, device='cpu',
#              verbose=False, mcts_min=False):
#     """Plays through one game using mcts_nn. Returns all
#     boards and move choices of the main line for training.
#
#     Args:
#         name (str): name for data
#         game (Board): the starting game state. If `None`
#             is passed, a new Board is generated.
#         model: model to predict moves
#         number (int): # of lines to try for each move.
#             Defaults to 10
#         device: torch device. Defaults to 'cpu'
#         verbose (bool): whether to print anything
#             Defaults to False
#         mcts_min (bool): whether to use conservative
#             strategy, max move number before single death.
#
#     Returns:
#         boards: list of boards
#         moves: list of mcts_nn moves
#
#     """
#     print('Seed {}'.format(name))
#     if not game:
#         game = Board(gen=True, draw=verbose, device=device)
#     boards = []
#     moves = []
#     while True:
#         if verbose and not len(moves) % 20:
#             print('Move {}'.format(len(moves)))
#         boards.append(game.board.clone())
#         if mcts_min:
#             pred = mcts_nn_min(model, game, number=number)
#         else:
#             pred = mcts_nn(model, game, number=number)
#         # Only need to do argmax. If not possible, game is dead
#         i = np.argmax(pred)
#         if game.move(i):
#             game.generate_tile()
#             moves.append(i)
#             if verbose:
#                 os.system(CLEAR)
#                 print('Seed {}'.format(name))
#                 print(pred)
#                 print(ARROWS[i.item()])
#                 game.draw()
#             continue
#         else:
#             boards.pop()
#             break
#     print('Game Over')
#     game.draw()
#     print('{} moves'.format(len(moves)))
#     if mcts_min:
#         name = 'min_move_dead/min' + name
#     np.savez('selfplay/'+name,
#              boards=torch.stack(boards),
#              moves=moves,
#              score=game.score)
#     print('Saved as {}.npz'.format(name))


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

    # m_name = '20190812/0_400_epox22_clr0.038_e20'
    # print('Using model: {}'.format(m_name))
    # m = ConvNet(channels=32, num_blocks=5)
    # m.load_state_dict(torch.load('models/{}.pt'.format(m_name)))
    # m.to('cuda')

    selfplay_fixed(name, number=50, verbose=args.verbose)
