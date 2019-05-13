Speed comparisons (in seconds) for various number of games. Result for 1 game is the average of 10 games. Standard deviation in parens. 
* Batching is slower for a single game, equal at about 10, and significantly faster at 100 games in parallel.
* No batch GPU is much slower than on CPU for all cases. In other testing, I found that tensors need to have on the order of 10^6 elements for GPU acceleration to start beating CPU. May be worth examining contribution of `play_fixed_batch` vs `move_batch` vs `merge_row_batch`.
* CPU: Intel Xeon E5-1620 @ 3.60 GHz
* GPU: GeForce RTX 2070 @ 645 MHz, CUDA 10.1

**CPU**

| n | nobatch | batch |
| --- | --- | --- |
| 1 | 0.189 (0.073) | 0.468 (0.139) |
| 10 | 1.70 | 1.25 |
| 100 | 16.5 | 3.27 |
| 1000 | x | 17.6 |

**CUDA**

| n | nobatch | batch |
| --- | --- | --- |
| 1 | 0.626 (0.244) | 1.14 (0.308) |
| 10 | 6.19 | 2.95 |
| 100 | 66.2 | 9.47 |
| 1000 | x | 63.9 |


Timings for `mcts_nn` with `number=100`. Running mcts is about 3x slower using cuda tensors. GPU gives about 3x faster evaluation with the CNN. However, given the overhead of running the mcts, running on GPU is still slower. ConvNet game is faster than TestNet because the mcts lines die earlier.

| Network | CPU | CUDA |
| --- | --- | --- |
| TestNet | 11.7| 47.3 |
| ConvNet | 9.77 | 26.4 |

Timings for 'play_nn' which does not do mcts (it only plays 1 game). Even this is slower due to slower `move` and `generate_tiles` functions on GPU. I should plan to optimize these functions in the future.

| Network | CPU | CUDA |
| --- | --- | --- |
| ConvNet | 1.13 | 1.27 |
