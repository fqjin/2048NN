# Suppress Tensorflow build from source messages
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
from random import randrange, randint
#import tensorflow as tf
#import keras
#from keras.layers import Activation, Dense, Flatten, Conv2D
#from keras.layers.normalization import BatchNormalization
#from keras.models import Sequential, load_model
#from keras.regularizers import l2

# Board dimensions
SIZE = 4
DIMENSIONS = (SIZE,SIZE)  # (4,4)
SIZE_SQRD = SIZE*SIZE  # 16


class Board:
    """ Board object stores 2048 board state
        Numbers are stored as log_2 format
    """
    
    def __init__(self):
        self.board = np.zeros(DIMENSIONS)
        self.score = 0
    
    def draw(self):
        """Prints board state"""
        print(str(2**self.board).replace('1',' ',SIZE_SQRD))
    
    def check_full(self):
        """Checks if board is full and has no empty tiles"""
        # Do I actually need this function?
        return np.count_nonzero(self.board) == SIZE_SQRD
    
    def generate_tile(self):
        """ Places a 2 or 4 in a random empty tile
            Unhandled error if board is full
            Chance of 2 is 90%
        """
        open = np.transpose(np.where(self.board == 0))
        position = open[randrange(len(open))]
        if randint(0,9):
            self.board[position[0],position[1]] = 1
        else:
            self.board[position[0],position[1]] = 2
        #   self.board[tuple(position)] is 3 times slower

    def merge_row(self, row):
        """Merges input row and shifts tiles to the left side"""
        final = []
        base = 0
        for tile in row:
            if tile == 0: continue  # Skips zeros
            if base == tile:
                final.append(tile+1)
                self.score += 2**(tile+1)
                base = 0
            else:
                if base: final.append(base)  # Don't append zeros
                base = tile
        if base: final.append(base)
        final += [0]*(SIZE-len(final))  # Pad with zeros
        return np.array(final)
            
    def move_left(self):
        """Execute left move or returns False if unable"""
        # Row by Row
        moved = False
        for i in range(SIZE):
            row = self.board[i]
            new_row = self.merge_row(row)
            if any(new_row != row):
                moved = True
                self.board[i] = new_row
        return moved         
            
    def move_up(self):
        """Execute up move or returns False if unable"""
        # Column by Column
        moved = False
        for i in range(SIZE):
            row = self.board[:,i]
            new_row = self.merge_row(row)
            if any(new_row != row):
                moved = True
                self.board[:,i] = new_row
        return moved  
        
    def move_right(self):
        """Execute right move or returns False if unable"""
        moved = False
        for i in range(SIZE):
            row = self.board[i,::-1]
            new_row = self.merge_row(row)
            if any(new_row != row):
                moved = True
                self.board[i,::-1] = new_row
        return moved
            
    def move_down(self):
        """Execute down move or returns False if unable"""
        moved = False
        for i in range(SIZE):
            row = self.board[::-1,i]
            new_row = self.merge_row(row)
            if any(new_row != row):
                moved = True
                self.board[::-1,i] = new_row
        return moved
    
    
raise SystemExit(0)
