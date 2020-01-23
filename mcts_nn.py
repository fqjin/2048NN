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
        b, s, m = move(origin, i)
        if m:
            array = BoardArray([generate_tile(b) for _ in range(number)])
            scores = []
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
                dead_s = array.move_batch(preds)
                if dead_s:
                    scores.extend(dead_s)
            scores = np.array(scores)
            result.append(np.mean(np.log10(scores+s+1)))
        else:
            result.append(-1)
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
