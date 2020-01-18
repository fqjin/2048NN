from mcts_batch import *

# Notes
# 5 -> 50 takes 9x longer (but 2.5 times less trials)
# 50->500 takes 6x longer (but 2.5 times less trials)
# Median takes about 1/2 as long
# Halving moves in mean is approximately the same time (slightly less)
# Min takes 4x less than median for 50 and 8x less for 500
# At fixed time control (~25 min for 20 games), mean still performs best

# Fixed move order
x = [play_fixed() for _ in range(5000)]
x = [[max(get_tiles(y[0])), y[1], y[2]] for y in x]
x = np.array(x)
np.save('distribution/fixed.npy', x)

# Fixed MCTS mean score
for num, times in zip((5, 50, 500), (50, 20, 8)):
    x = [play_mcts_fixed(number=num, mode=0) for _ in range(times)]
    x = [[max(get_tiles(y[0])), y[1], y[2]] for y in x]
    x = np.array(x)
    np.save('distribution/mcts{}.npy'.format(num), x)

# Fixed MCTS median moves
for num, times in zip((5, 50, 500), (50, 20, 8)):
    x = [play_mcts_fixed(number=num, mode=1) for _ in range(times)]
    x = [[max(get_tiles(y[0])), y[1], y[2]] for y in x]
    x = np.array(x)
    np.save('distribution/mcts_median{}.npy'.format(num), x)

# Fixed MCTS min moves
for num, times in zip((5, 50, 500), (50, 20, 8)):
    x = [play_mcts_fixed(number=num, mode=2) for _ in range(times)]
    x = [[max(get_tiles(y[0])), y[1], y[2]] for y in x]
    x = np.array(x)
    np.save('distribution/mcts_min{}.npy'.format(num), x)
