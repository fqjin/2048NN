from board import *


def mcts_nn(model, origin, number=10, device='cpu'):
    """Run tree search with pytorch model making lines.
    Batch implementation for efficiency.

    Args:
        model: pytorch model to predict moves
        origin (int64): the starting game state
        number (int): # of lines to try for each move.
            Defaults to 10
        device: default 'cpu'

    Returns:
        list: log mean score for each move [Left, Up, Right, Down]
    """
    result = []
    for i in range(4):
        b, _, m = move(origin, i)
        if m:
            array = BoardArray([generate_tile(b) for _ in range(number)])
            count = 1
            while array.boards:
                b = np.array(array.boards, dtype=np.uint64)
                data = []
                for _ in range(16):
                    data.append(b & 0xF)
                    b >>= 4
                b = torch.tensor(data,
                                 dtype=torch.float32,
                                 device=device).transpose(0, 1)
                b = [b == i for i in range(16)]
                b = torch.stack(b, dim=1).float()
                preds = model(b.view(-1, 16, 4, 4))
                preds = torch.argsort(preds.cpu(), dim=1, descending=True)
                if array.fast_move_batch(preds):
                    result.append(count)
                    break
                count += 1
        else:
            result.append(0)
    return result


def play_nn(model, game=None, press_enter=False, device='cpu', verbose=False):
    """Play through a game using a pytorch NN.

    Moves are selected by the pytorch model.
    No monte carlo simulations are used.

    Args:
        model: pytorch model to predict moves
        game (int64): the starting game state.
            Default will generate a new Board.
        press_enter (bool): Whether keyboard press is
            required for each step. Defaults to False.
            Type 'q' to quit when press_enter is True.
        device: torch device. Defaults to 'cpu'
        verbose (bool): whether to print mcts scores
            Defaults to False

    """
    if game is None:
        game = generate_init_tiles()
    score = 0
    count = 0
    if press_enter:
        draw(game, score)
    model.eval()
    with torch.no_grad():
        while True:
            if press_enter and input() == 'q':
                break
            b = torch.tensor([get_tiles(game)],
                             dtype=torch.float32,
                             device=device)
            b = [b == i for i in range(16)]
            b = torch.stack(b, dim=1).float()
            output = model(b.view(-1, 16, 4, 4))
            preds = torch.argsort(output, dim=1, descending=True)
            for i in preds[0]:
                f, s, m = move(game, i.item())
                if m:
                    game = generate_tile(f)
                    score += s
                    count += 1
                    if press_enter:
                        print(output)
                        print(ARROWS[i.item()])
                        draw(game, score)
                    break
            else:
                return game, score, count


if __name__ == '__main__':
    from time import time
    from network import *
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}')
    torch.backends.cudnn.benchmark = True

    t = time()
    # m = ConvNet(channels=64, blocks=3)
    # name = '20200126/soft3.5_20_200_c64b3_p10_bs2048lr0.08d0.0_s4_best'
    # name = '20200128/20_400_soft3.5c64b3_p10_bs2048lr0.08d0.0_s2pre_best'
    # name = '20200126/soft3.5_s4_jit.pth'
    name = '20200128/best_s2_jit.pth'

    print(name)
    # m.load_state_dict(torch.load('models/{}.pt'.format(name), map_location=device))
    # m.to(device)
    # m.eval()
    # m = torch.jit.trace(m, torch.randn(50, 16, 4, 4, dtype=torch.float32, device=device))
    m = torch.jit.load('models/' + name)
    print(time()-t)
    # Using jit: 10.3  vs 9.1 sec, about 10% speed up
    # Using jit: 10.4 vs 11.8 sec, about 13% speed up   
    # Loading from jit saves 0.4 seconds compared to tracing each time

    x = [generate_init_tiles() for _ in range(4)]
    with torch.no_grad():
        print(mcts_nn(m, x[0], number=50, device=device))
        t = time()
        print(mcts_nn(m, x[1], number=50, device=device))
        print(mcts_nn(m, x[2], number=50, device=device))
        print(mcts_nn(m, x[3], number=50, device=device))
        t = time() - t
    print('{0:.3f} seconds'.format(t))
    print('-' * 10)
