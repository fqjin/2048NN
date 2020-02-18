import curses  # pip install windows-curses
from board import *


def play_manual(board=None):
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
        if not board:
            board = generate_init_tiles()
        score = 0
        curses_draw(screen, board, score)
        while True:
            char = screen.getch()
            i = -1
            if char == ord('q'): break
            elif char == curses.KEY_LEFT: i = 0
            elif char == curses.KEY_UP:   i = 1
            elif char == curses.KEY_RIGHT:i = 2
            elif char == curses.KEY_DOWN: i = 3
            if i != -1:
                f, s, m = move(board, i)
                if m:
                    board = generate_tile(f)
                    score += s
                    curses_draw(screen, board, score)

    finally:  # cleanup curses
        curses.nocbreak()
        screen.keypad(False)
        curses.echo()
        curses.endwin()
        print('Game Over')
        print('Score: {}'.format(score))
        return board


def curses_draw(screen, x, score):
    """Draw function for curses"""
    screen.clear()
    tiles = get_tiles(x)
    expo = np.power(2, tiles)
    expo = expo.reshape(4, 4)
    expo = str(expo)
    screen.addstr(expo.replace('1 ', '. ', 12).replace('1]', '.]', 4))
    screen.addstr('\n Score : {}'.format(score))
    screen.refresh()


if __name__ == '__main__':
    play_manual()
