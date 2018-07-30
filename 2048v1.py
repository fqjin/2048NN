#Suppress Tensorflow build from source messages
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
#import tensorflow as tf
from random import randrange, randint
#import keras
#from keras.layers import Activation, Dense, Flatten, Conv2D
#from keras.layers.normalization import BatchNormalization
#from keras.models import Sequential, load_model
#from keras.regularizers import l2

#Board dimensions
dim = (4,4)
elements = 16


class Board:
    """Board object stores 2048 board state
    Numbers are stored as log_2 format
    """
    
    def __init__(self):
        self.board = np.zeros(dim)
        self.score = 0
    
    def draw(self):
        """Prints board state"""
        print(str(2**self.board).replace('1',' ',16))
    
    def checkFull(self):
        """Checks if board is full and has no empty tiles"""
        return np.count_nonzero(self.board) == elements
    
    def generate(self):
        """Places a 2 or 4 in a random empty tiles
        Unhandled error if board is full
        Chance of 2 is 90%
        """
        open = np.transpose(np.where(self.board == 0))
        position = open[randrange(len(open))]
        if randint(0,9):
            self.board[position[0],position[1]] = 1
        else:
            self.board[position[0],position[1]] = 2

    def moveUp(self):
        """Execute up move or returns False if unable"""
        
        
        return False
        
        

raise SystemExit(0)
