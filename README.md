# 2048NN
### Train a neural network to play 2048

### Stable Version 1: 2048 game and MCTS, single script, without neural network

Numerous AI algorithms have been delevoped to play the game of "2048".

This project seeks to implement an algorithm (inspired by AlphaZero) which uses:
- monte carlo tree search
- no or minimal playing heuristics
- reinforcement learning through self-play
- (dual policy & value networks were not required)

Language: python3

Dependencies:
- random
- numpy
- ~~tensorflow~~
- ~~keras~~
- curses (for manual play)

Module documentation:
- Board (class): tile values and positions in an array, game score, logic for generating new tiles, logic for executing moves in all 4 directions.
- play_manual: play manually using the arrow keys
- play_fixed: autoplay with a fixed move order (L,U,R,D)
- MCTS: performs the monte-carlo tree search documented below
- play_MCTS: autoplay using first-order MCTS results with a fixed move order method for generating lines

MCTS process:

For each possible initial move, generate multiple possible lines to the end-of-game to estimate the expected value of each move.
Current version has no initial bias for starting moves and will search each move equally. 
Starting after the initial move, a playing method is used to generate further moves until the game is over. 
The result is a 'line' in the tree. 
This process is repeated a number of times and the final scores are combined to estimate the true score of each move.

