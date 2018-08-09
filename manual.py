from board import *
# includes SIZE, DIMENSIONS, SIZE_SQRD
# includes Board class
# includes randrange, randint, numpy as np

# Can compact code using Board.moves[i] to call different moves
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
