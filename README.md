# 2048NN
### Train a neural network to play 2048

Numerous AI algorithms have been developed to play the game of "2048".

This project seeks to implement an algorithm which uses:
- monte carlo tree search
- no playing heuristics
- reinforcement learning through self-play
- (dual policy & value networks were not required)

#### Language: python3

#### Dependencies:
- numpy
- tensorflow (as backend)
- keras
- curses (for manual play)

#### Module documentation:
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

#### Tree search process:

For each possible initial move, generate multiple possible lines to the end-of-game to estimate the expected value of each move.
Current version has no initial bias for starting moves and will search each move equally. 
Starting after the initial move, a playing method is used to generate further moves until the game is over. 
The result is a 'line' in the tree. 
This process is repeated a number of times and the final scores are combined to estimate the true score of each move.
