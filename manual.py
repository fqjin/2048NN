import curses  # pip install windows-curses
from board import Board


def play_fixed(press_enter=False):
    """Run 2048 with the fixed move priority L,U,R,D.

    Args:
        press_enter (bool): Whether keyboard press is
            required for each step. Defaults to False.
            Type 'q' to quit when press_enter is True.

    """
    game = Board(gen=True)
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


def play_manual():
    """Play 2048 manually with the arrow keys"""
    screen = curses.initscr()
    curses.noecho()
    curses.cbreak()
    screen.keypad(True)

    try:
        # draw_curses function with screen.addstr not needed
        game = Board(gen=True)
        while True:
            char = screen.getch()
            i = -1
            if char == ord('q'): break
            elif char == curses.KEY_LEFT: i = 0
            elif char == curses.KEY_UP:   i = 1
            elif char == curses.KEY_RIGHT:i = 2
            elif char == curses.KEY_DOWN: i = 3
            if i != -1:
                if game.moves[i]():
                    game.generate_tile()
                    game.draw()

    finally:  # cleanup curses
        curses.nocbreak()
        screen.keypad(False)
        curses.echo()
        curses.endwin()
        print('Game Over')


if __name__ == '__main__':
    play_manual()
