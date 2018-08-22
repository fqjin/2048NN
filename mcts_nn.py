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
from keras.utils import to_categorical
# Suppress Tensorflow build from source messages
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL']='2'


def generate_model(number):
    # Architecture
    """
    Creates and returns a new NN, and saves as 'nn2048v_number_.h5'
    Args: number should be a str of format '#-#' but can be an int
    """
    model = Sequential()
    model.add(Dense(16, input_dim = SIZE_SQRD))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(16, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(8, activation = 'relu', kernel_initializer='uniform'))
    model.add(Dense(4, activation = 'softmax', kernel_initializer='uniform'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    
    model.save('nn2048v'+str(number)+'.h5')
    print('Saved as: nn2048v'+str(number)+'.h5')
    return model

    
def get_model(number):
    """Return the NN 'nn2048v_number_.h5'"""
    return load_model('nn2048v'+str(number)+'.h5')

    
def save_model(model, number):
    """Save the NN as 'nn2048v_number_.h5'"""
    model.save('nn2048v'+str(number)+'.h5')   
    
    
def play_nn(game, model, press_enter = False):
    """Automatically play through game using neural network choices (w/o monte carlo)"""
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
    # Unused
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
        score difference from current state for each move as a list [L, U, R, D]  
    """
    # With a neural network, it may be more efficient to pass mulitple boards simultaneously
    original_board = np.copy(game.board)
    original_score = game.score
    scores_list = np.zeros(4)
    
    for i in range(4):
        if not game.moves[i]():
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
                scores_list[i] += game.score  # Add to score for each run
            scores_list[i] /= number  # Calculate average final score
            scores_list[i] -= original_score  # Subtract off original score
            game.restore(original_board, original_score)
            
    return scores_list
    
    
def make_data(game, model, number = 5):
    """
    Plays through one game using monte-carlo search.
    Returns all boards and computed scores for the main line for training use.
    """
    boards = []
    results = []
    while True:
        scores_list = mcts_nn(game, model, number)
        print(scores_list)
        order = np.flipud(np.argsort(scores_list))
        if sum(scores_list) > 0:
            boards.append(game.board.flatten())
            results.append(order[0])
        else:
            print('Null scores list')
            
        for i in order:
            if game.moves[i]():
                game.generate_tile()
                game.draw()
                break
        else:       
            print('Game Over')
            break
            
    return boards, results

            
def training(boards, results, model, epochs = 5):
    """Trains a model using model.fit on a set of boards and associated results"""
    model.fit(np.vstack(boards), to_categorical(results, num_classes = 4), epochs = epochs)


# FOR TESTING
print('a = Board(gen = True)')
a = Board(gen = True)

