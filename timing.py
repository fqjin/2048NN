import numpy as np
from time import time
from board import *


def nobatch(n, device):
    """Timing for play_fixed

    Args:
        n: number of games
        device: torch device

    """
    t = time()
    for _ in range(n):
        play_fixed(device=device)
    t = time() - t
    return t


def batch(n, device):
    """Timing for play_fixed_batch

    Args:
        n: number of games
        device: torch device

    """
    t = time()
    play_fixed_batch(number=n, device=device)
    t = time() - t
    return t


def multi(m, use_batch, device, n=1):
    """Run n games m times and calculate stats

    Args:
        m: number of times
        use_batch: boolean to use batch or not
        device: torch device
        n: number of games per run. Defaults to 1

    """
    times = []
    for _ in range(m):
        if use_batch:
            times.append(batch(n, device))
        else:
            times.append(nobatch(n, device))
    print('Mean: {}'.format(np.mean(times)))
    print('Std: {}'.format(np.std(times)))
