from board import *
# includes SIZE, DIMENSIONS, SIZE_SQRD
# includes Board class
# includes randrange, randint, numpy as np

# import tensorflow as tf
import keras
from keras.layers import Activation, Dense, Flatten, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.regularizers import l2
# Suppress Tensorflow build from source messages
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL']='2'


def generate_model(name):
    # Architecture
    # First try fully connected without convolution. CNN would be more better theoretically given spatial element.
    # First try without L2-regularizer
    """Creates and returns a new NN, and saves as 'nn2048_name_.h5'"""
    model = Sequential()
    model.add(Dense(16, input_dim = SIZE_SQRD))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(16, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(8, activation = 'relu', kernel_initializer='uniform'))
    model.add(Dense(4, activation = 'softmax', kernel_initializer='uniform'))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    
    model.save('nn2048'+name+'.h5')
    print('Saved as: nn2048'+name+'.h5')
    return model

    
def get_model(name):
    """Return the NN 'nn2048_name_.h5'"""
    return load_model('nn2048'+name+'.h5')
    
    
def play_nn(game, model, press_enter = False):
    while True:
        if press_enter and input() == 'q':
            break
        scores_list = model.predict(game.board.reshape((1,SIZE_SQRD)))[0]
        print(scores_list)
        for i in np.flipud(np.argsort(scores_list)):
            if game.moves[i]():
                game.generate_tile()
                game.draw()
                break
        else:        
            print('Game Over')
            break
    
    
def method_model(board, model):
    scores_list = model.predict(board.reshape((1,SIZE_SQRD)))[0]
    return np.flipud(np.argsort(scores_list))

    
def mcts_nn(game, model, number = 5):
    """
    Run Monte Carlo Tree Search using NN for generating lines
    
    Args:
        game (Board): the starting game state
        model (Sequential): keras NN for selecting moves
        number (int): # of lines to try for each move. Default is 5
    Returns:
        scores for each move as a list [Left, Up, Right, Down]
        
    """
    # With a neural network, it may be more efficient to pass mulitple boards simultaneously
    original_board = np.copy(game.board)
    original_score = game.score
    scores_list = np.zeros(4)
    
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
                    predictions = model.predict(game.board.reshape((1,SIZE_SQRD)))[0]
                    for j in np.flipud(np.argsort(predictions)):
                        if game.moves[j]():
                            game.generate_tile()
                            break
                    else:
                        break
                scores_list[i] += game.score       
            scores_list[i] /= number  # Calculate average final score
            game.restore(original_board, original_score)
            
    return scores_list
    
    
def make_data(game, model, number = 5):
    # First try without batching mcts, but it is slow. It takes 5 minutes to complete a game.
    # First try pure self-learning without relying on starter fixed_order training data
    boards = []
    results = []
    while True:
        scores_list = mcts_nn(game, model, number)
        print(scores_list)
        boards.append(game.board.flatten())
        change = scores_list - game.score
        if sum(change) == 0:
            results.append(np.full(4,0.25))
        else: 
            results.append(change / sum(change))
        
        for i in np.flipud(np.argsort(scores_list)):
            if game.moves[i]():
                game.generate_tile()
                game.draw()
                break
        else:        
            print('Game Over')
            boards.pop()
            results.pop()            
            break
            
    return boards, results

            
def training(boards, results, model, epochs = 10):
    model.fit(np.vstack(boards), np.vstack(results), epochs)


# FOR TESTING
print('a = Board()')
a = Board()
a.generate_tile()
a.generate_tile()
a.draw()
