if __name__ == '__main__':
    from board import *
    from mcts_nn import *
    from network import *
    from time import time

    device = 'cpu'
    # initializer = torch.ones((10, 10), dtype=torch.int, device=device)
    a = Board(draw=False, device=device)
    # play_fixed(a)
    # a.board -= 1
    a.draw()
    name = '0_10_epox100_lr0.1_e0'
    # m = Fixed()
    m = ConvNet()
    m.load_state_dict(torch.load('models/{}.pt'.format(name)))
    m.to('cuda')

    if True:
        t = time()
        x = mcts_nn(m, a, number=50)
        print(time() - t)
    print(x)

    # selfplay(0, m, a, number=10, verbose=True)