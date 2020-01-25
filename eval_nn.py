from board import *


def eval_nn(model, name=None, origin=None, number=100, device='cpu', verbose=False):
    """Simulate games using a model

    Args:
        model: pytorch model to predict moves
        name: name to save. If no name given, status are returned
        origin (int64): the starting game state
            Defaults to None (generate new)
        number (int): # of lines
            Defaults to 100
        device: torch device
        verbose:
    """
    if origin is None:
        games = BoardArray([generate_init_tiles() for _ in range(number)])
    else:
        games = BoardArray([origin] * number)

    count = 0
    scores = []
    dead_boards = []
    moves = []
    model.eval()
    with torch.no_grad():
        while games.boards:
            b = np.array(games.boards, dtype=np.uint64)
            data = []
            for _ in range(16):
                data.append(b & 0xF)
                b >>= 4
            b = torch.tensor(data,
                             dtype=torch.float32,
                             device=device).transpose(0, 1)
            # TODO: more efficient one-hot conversion?
            b = [b == i for i in range(16)]
            b = torch.stack(b, dim=1).float()
            preds = model(b.view(-1, 16, 4, 4))
            preds = torch.argsort(preds.cpu(), dim=1, descending=True)
            dead_s, dead_b = games.move_batch(preds, get_boards=True)
            if dead_s:
                scores.extend(dead_s)
                dead_boards.extend(dead_b)
                moves.extend([count]*len(dead_s))
            count += 1
            if verbose and count % 100 == 0:
                print(count)
    scores = np.array(scores)
    dead_boards = np.asarray(dead_boards, dtype=np.uint64)

    if name is None:
        return np.mean(np.log10(scores+1)), np.amax(scores), np.mean(moves)
    else:
        np.savez('models/{}.npz'.format(name), scores=scores, moves=moves, boards=dead_boards)
        print(name)
        print(np.mean(np.log10(scores+1)), np.amax(scores), np.mean(moves))


if __name__ == '__main__':
    from time import time
    from network import ConvNet
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}')

    names = [f'20200123/onehot_20_200_c128b5_p20_bs2048lr0.01d0.0_s{s}_best' for s in range(5)]
    for name in names:
        m = ConvNet(**{'channels': 128, 'blocks': 5})
        m.load_state_dict(torch.load('models/{}.pt'.format(name), map_location=device))
        m.to(device)
        t = time()
        print(eval_nn(m, name, number=500, device=device, verbose=True))
        t = time() - t
        print('{0:.3f} seconds'.format(t))
        print('-'*10)
