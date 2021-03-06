# 2048NN
### Train a neural network to play 2048
This project uses a policy network and tree search to find the optimal moves.
The neural network is trained through self-play reinforcement learning.


### The nibble update
* Changed board engine to use nibbles and bitwise operators, as proposed in github/nneonneo/2048-ai.
* `play_fixed` is 80 times faster
* `play_fixed_batch` is 11 times faster (previous batch methods were 7.5x faster, new method has no batch acceleration)
* `mcts_fixed_batch` (mean log score) is 4.7 times faster
* Unfortunately, network forward is still the bottleneck.

### Other changes
* Soft classification targets during training.
* min_move_dead playouts
* Fixed10: On 20200213, remade all training data using fixed LUDR with 10 playouts, min_move_dead averaged over 4 runs.

### Milestones:
Network name: % policy games that acheive 2048

| Network Name | % Policy games acheiving 2048 |
|---|---|
| `20200126/soft3.5_20_200_c64b3_p10_bs2048lr0.08d0.0_s4_best` | 0.4 |
| `20200128/20_400_soft3.5c64b3_p10_bs2048lr0.08d0.0_s2pre_best` | 1.88 |
| `20200130/0_600_soft3.5c64b3_p10_bs2048lr0.08d0.0_s7pre_best` | 4.2 |
| `20200205/0_800_soft3.0c64b3_p10_bs2048lr0.1d0.0_s0_best` | 6.16 |
| `20200207/0_1000_soft3.0c64b3_p10_bs2048lr0.1d0.0_s7_best` | 7.32 |
| `20200213/0_800_soft3.0c64b3_p10_bs2048lr0.1d0.0_s2_best` | 9.0 |
| `20200213/0_3400_soft3.0c64b3_p10_bs2048lr0.1d0.0_s0_best` | 24.0 |
| x | x |


### Findings:
* Using fixed move order (Left, Up, Right, Down) can reach 2048 occasionally.
* Hyperparameter optimization is necessary for training strong models.
* Models tend to play better during the 'late game' (higher score boards).
Possibly due to training data distribution.
* Strong trained models prefer the move order (Left, Up, Down, Right).
This fixed order is indeed slightly stronger than the initially proposed order.
It makes sense in retrospect.


### Monte Carlo playout process:
Given the current board, for each legal move, a number (*e.g.* 50) of games starting from that move are played to the end.
Subsequent moves in each playout game are made according to either a fixed move order or the output of a neural network model (*i.e.* a policy network).
The log of the scores of each playout are averaged to produce a final log score for each initial legal move.
The chosen move for the initial board state is the one with the highest log score.
No bias is used for the initial move.

This Monte Carlo playout process results in much stronger moves than the policy generating the moves during playouts. 
These stronger moves allow the main game line to reach much higher scores and tile values.
The stronger moves from the playout process are then used for training the neural network to increase the strength of its policy, which feeds back into stronger playout results.

TODO: describe min_move_dead


#### Dependencies:
- numpy
- torch [pytorch]
- ax [ax-platform] (for optimization)
- curses [ncurses / windows-curses] (for manual play)
