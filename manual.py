from board import *
# includes SIZE, DIMENSIONS, SIZE_SQRD
# includes Board class
# includes randrange, randint, numpy as np

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
            i = 0
            if char == ord('q'): break
            elif char == curses.KEY_LEFT: i = 1
            elif char == curses.KEY_UP:   i = 2
            elif char == curses.KEY_RIGHT:i = 3
            elif char == curses.KEY_DOWN: i = 4
            if i:
                if game.moves[i-1]():
                    game.generate_tile()
                    game.draw_curses(screen)
                    
    finally:  # cleanup curses
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
    game = Board()
    game.generate_tile()
    game.generate_tile()
    game.draw()
    while True:
        if press_enter and input() == 'q':
            break
        for i in range(4):
            if game.moves[i]():
                game.generate_tile()
                game.draw()
                break
        else:
            print('Game Over')
            break
