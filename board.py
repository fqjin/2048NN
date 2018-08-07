# Suppress Tensorflow build from source messages
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
# Do not use scientific notation:
np.set_printoptions(suppress = True)
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
        self.moves = [self.move_left, 
                      self.move_up, 
                      self.move_right, 
                      self.move_down]
    
    def restore(self, board, score):
        """Sets board and score to input values"""
        self.board = np.copy(board)
        self.score = score
    
    def draw(self):
        """Prints board state"""
        print(str(2**self.board).replace('1.',' .',SIZE_SQRD))
        print(' Score : {}'.format(self.score))
    
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
                self.score += 2**(int(tile)+1)
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


# Can split into separate modules
def play_manual():
    """Play 2048 manually with arrow keys"""
    import curses
    screen = curses.initscr()
    curses.noecho()
    curses.cbreak()
    screen.keypad(True)
    try:
        
        def draw_curses(self,screen):
            screen.erase()
            screen.addstr(str(2**self.board).replace('1.',' .',SIZE_SQRD))
            screen.addstr('\n Score : {}'.format(self.score))
            
        Board.draw_curses = draw_curses
        game = Board()
        game.generate_tile()
        game.generate_tile()
        game.draw_curses(screen)

        while True:
            char = screen.getch()
            if char == ord('q'):
                break     
            elif char == curses.KEY_LEFT:
                if game.move_left():
                    game.generate_tile()
                    game.draw_curses(screen)
            elif char == curses.KEY_UP:
                if game.move_up():
                    game.generate_tile()
                    game.draw_curses(screen)
            elif char == curses.KEY_RIGHT:
                if game.move_right():
                    game.generate_tile()
                    game.draw_curses(screen)
            elif char == curses.KEY_DOWN:
                if game.move_down():
                    game.generate_tile()
                    game.draw_curses(screen)   
                    
    finally:
        curses.nocbreak()
        screen.keypad(False)
        curses.echo()
        curses.endwin()
        print('Game Over')
        game.draw()


def play_fixed(press_enter = False):
    """ Run 2048 with the fixed move priority L,U,R,D.
        press_enter (bool) : Defaults to False
    """
    # Score: 3380, 2636, 2628, 1480, 1152. Game over with 128 or 256 tile.
    game = Board()
    game.generate_tile()
    game.generate_tile()
    game.draw()
    while True:
        if press_enter and input() == 'q':
            break
        if game.move_left():
            game.generate_tile()
            game.draw()
            continue
        elif game.move_up():
            game.generate_tile()
            game.draw()
            continue
        elif game.move_right():
            game.generate_tile()
            game.draw()
            continue    
        elif game.move_down():
            game.generate_tile()
            game.draw()
            continue
        else:
            print('Game Over')
            break


def method_fixed(board):
    """Returns L,U,R,D move priority as a tuple"""
    return (0,1,2,3)
    

def MCTS(game, method = method_fixed, number = 5):
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
    
    
def play_MCTS(game, number = 5):
    """ 
    Play a game using the default MCTS
    Args:
        game (Board): the starting game state
        number (int): Default is 5
    """
    # Score: 6300, 11536, 10520, with 1024 tile
    # With number = 10, score is 15520 with a 1024, 512, 256, and two 64 tiles
    while True:
        scores_list = MCTS(game, number = number)
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
a = Board()
a.generate_tile()
a.generate_tile()
a.draw()

