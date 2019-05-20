if __name__ == '__main__':
    from mcts_nn import *
    from network import *
    from time import time

    device = 'cuda'
    a = Board(draw=True, device=device)
    name = '0_10_epox100_lr0.1_e0'
    m = ConvNet()
    m.load_state_dict(torch.load('models/{}.pt'.format(name)))
    m.to(device)

    if True:
        t = time()
        x = mcts_nn(m, a, number=50)
        print(time() - t)
    print(x)

    # selfplay(0, m, a, number=10, verbose=True)