# 2048NN
### Train a neural network to play 2048
This project uses a policy network and tree search to find the optimal moves.
The neural network is trained through self-play reinforcement learning.


### New changes (nibble):
* Changed board processing code to use nibbles and bitwise operators, as proposed in github/nneonneo/2048-ai. This provides a large speedup in board operations.
* Neural network changed from convolutional network to full connected network operating on the flattened 16 tiles.
* Median of rollout scores is used as the final score. The score distributions are extremely right skew, so median is a better summary statistic that helps favor 'safety' rather than 'expected value'. Median is also faster because only 50% of rollouts need to terminate.


### Milestones:
* First 2048 tile achieved ...
* First 4096 tile achieved ...
* First 8192 tile achieved ...


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


#### Dependencies:
- numpy
- torch [pytorch]
- ax [ax-platform] (for optimization)
- curses [ncurses / windows-curses] (for manual play)
