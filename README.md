# 2048NN
### Train a neural network to play 2048

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
- tensorflow
- keras
- curses (for manual play)

Module documentation:
- Board (class): stores tile values and positions in an array, stores game score, logic for generating new tiles, logic for executing moves in all 4 directions.
- play_manual: manual play using the arrow keys
- play_fixed: autoplay with a fixed move order (L,U,R,D)
- MCTS: performs the monte-carlo tree search documented below
- play_MCTS: autoplay using first-order MCTS results with a fixed move order method for generating lines

MCTS process:

For each possible initial move, generates multiple possible lines to end-of-game to estimate the expected value of each move.
Current version has no initial bias for starting moves and will search each move equally. 
Starting after the initial move, a playing method is used to generate moves until the game is over. The result is a 'line' in the tree. 
This process is repeated a number of times and the final scores combined to estimate the true score of each move.

Note 1: This method is not truely a 'tree search' because it does not need to remember intermediate states between starting and final state. It does not use information about previous searches for generating new lines. Each line is able to be unique because of the probabilistic nature of generating new tiles.

Note 2: A more intelligent generating method (rather than fixed move order), such as with a neural network, will significantly improve the relevance of each line to the true value of each starting move. This will increase the accuracy of of the search as well as decrease the number of lines needed.





