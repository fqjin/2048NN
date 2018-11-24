import curses  # pip install windows-curses
from board import Board


def play_manual(game=None):
    """Play 2048 manually with the arrow keys

    Args (optional):
        game (Board): the starting game state.
            Default will generate a new Board.
    """
    screen = curses.initscr()
    curses.noecho()
    curses.cbreak()
    screen.keypad(True)

    try:
        # draw_curses function with screen.addstr not needed
        if not game:
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
                if game.move(i):
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
