# 2048NN
### Train a neural network to play 2048
There are many approaches to an AI algorithm that plays the game of "2048".
This project uses a policy neural network and Monte Carlo search to find the optimal moves.
The neural network is trained through self-play reinforcement learning.


### Milestones:
* First 2048 tile achieved (game 0) using fixed move order 
* First 4096 tile achieved (game 79) using model `20190625/10_60_epox60_lr0.125_e49`
* First 8192 tile achieved (game 100) using model `20190710/80_100_epox10_lr0.0043pre_e9`


* No games below 512 tile: fixed move order
* No games below 1024 tile: model `20190701/60_80_epox10_lr0.0043pre_e9`
* No games below 2048 tile (goal): model `to be determined`


### Findings:
* Using fixed move order (Left, Up, Right, Down) can reach 2048 occasionally.
* Hyperparameter optimization is necessary for training strong models.
* Models tend to play better during the 'late game' (higher score boards).
Possibly due to training data distribution.
* Strong trained models prefer the move order (Left, Up, Down, Right).
This makes sense in retrospect.


### Monte Carlo playout process:
Given the current board, for each legal move, a number (*e.g.* 50) of games starting from that move are played to the end.
Subsequent moves in each playout game are made according to either a fixed move order or the output of a neural network model.
The log of the scores of each playout are averaged to produce a final log score for each initial legal move.
The chosen move for the initial board state is the one with the highest log score.
No bias is used for the initial move.

This Monte Carlo playout process results in much stronger moves than the policy generating the moves during playouts. 
These stronger moves allow the main game line to reach much higher scores and tile values.
The stronger moves from playout process are then used to train the neural network to increase the strength of its policy, which feeds back into stronger playout results.


#### Dependencies:
- numpy
- torch [pytorch]
- ax [ax-platform] (for optimization)
- curses [ncurses / windows-curses] (for manual play)


#### Module documentation (needs updating):
board.py:
- Board (class): tile values and positions in an array, game score, logic for generating new tiles, logic for executing moves in all 4 directions.
- play_fixed: autoplay with a fixed move order (L,U,R,D)

manual.py:
- play_manual: play manually using the arrow keys

mcts.py:
- mcts_fixed: performs the monte-carlo tree search documented below, using a fixed move order
- play_mcts_fixed: autoplay using first-order MCTS results with a fixed move order method for generating lines

mcts_batch.py
- mcts_fixed_batch: performs tree search using a batch process

mcts_nn.py:
- simple_model: creates a new keras neural network, simple & fully-connected
- get_model: loads a previously-saved neural network
- play_nn: autoplay using a neural network to select moves (without using monte-carlo search)
- mcts_nn: performs monte-carlo search using the neural network to generate lines. Uses batch process for efficiency.
- make_data: play a game using mcts_nn, returns every board in main line and the final mcts move scores
- train: train a neural network using a set of boards and MCTS search results
