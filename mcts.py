from board import *
# includes SIZE, DIMENSIONS, SIZE_SQRD
# includes Board class
# includes randrange, randint, numpy as np


def method_fixed(board):
    """Returns L,U,R,D move priority as a tuple"""
    return (0,1,2,3)
    

def mcts(game, method = method_fixed, number = 5):
    """
    Run Monte Carlo Tree Search
    
    Args:
        game (Board): the starting game state
        method: method for selecting moves. Default is method_fixed
        number (int): # of lines to try for each move. Default is 5
    Returns:
        scores for each move as a list [Left, Up, Right, Down]
        
    """
    # With a neural network, it may be more efficient to pass mulitple boards simultaneously
    original_board = np.copy(game.board)
    original_score = game.score
    scores_list = [0,0,0,0]
    
    for i in range(4):
        if not game.moves[i]():
            scores_list[i] = original_score
            game.restore(original_board, original_score)
        else:
            for _ in range(number):
                game.restore(original_board, original_score)
                game.moves[i]()
                game.generate_tile()
                while True:
                    move_order = method(game.board)
                    for j in range(4):
                        if game.moves[move_order[j]]():
                            game.generate_tile()
                            break
                    else:        
                        # print('Game Over')
                        # game.draw()
                        break
                scores_list[i] += game.score       
            scores_list[i] /= number  # Calculate average final score
            game.restore(original_board, original_score)
            
    return scores_list
    
    
def play_mcts(game, number = 5):
    """ 
    Play a game using the default mcts
    Args:
        game (Board): the starting game state
        number (int): Default is 5
    """
    while True:
        scores_list = mcts(game, number = number)
        # print(scores_list)
        for i in np.flipud(np.argsort(scores_list)):
            if game.moves[i]():
                game.generate_tile()
                game.draw()
                break
        else:        
            print('Game Over')
            break


# FOR TESTING
print('a = Board()')
a = Board()
a.generate_tile()
a.generate_tile()
a.draw()

