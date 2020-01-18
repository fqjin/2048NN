from board import *


def mcts_fixed_batch(origin, number=10, move_order=(0, 1, 3, 2)):
    """Run batch tree search using a fixed move order

    A BoardArray is made for each possible move
    Lines in each array are played to end
    Returns mean score for each initial move

    Args:
        origin: the starting game state
        number (int): # of lines to simulate for each move.
            Defaults to 10
        move_order: tuple of the 4 move indices in order.
            Defaults to (0, 1, 3, 2)

    Returns:
        list: mean score increase for each move [Left, Up, Right, Down]
    """
    result = []
    for i in range(4):
        board, s, moved = move(origin, i)
        if moved:
            array = BoardArray(number, board)
            array.boards = [generate_tile(b) for b in array.boards]
            scores = []
            while array.boards:
                dead_s = array.move_batch(move_order)
                if dead_s:
                    scores.extend(dead_s)
            scores = np.array(scores)
            result.append(np.mean(np.log10(scores+s+1)))
        else:
            result.append(-1)
    return result


def mcts_fixed_batch_moves(origin, number=10, move_order=(0, 1, 3, 2)):
    """Run batch tree search using a fixed move order

    A BoardArray is made for each possible move
    Lines in each array are played to end
    Returns median move count for each initial move (xx times faster)

    Args:
        origin: the starting game state
        number (int): # of lines to simulate for each move.
            Defaults to 10
        move_order: tuple of the 4 move indices in order.
            Defaults to (0, 1, 3, 2)

    Returns:
        list: median moves for each move [Left, Up, Right, Down]
    """
    result = []
    for i in range(4):
        board, score, moved = move(origin, i)
        if moved:
            array = BoardArray(number, board)
            array.boards = [generate_tile(b) for b in array.boards]
            count = 1
            dead = 0
            while array.boards:
                dead_s = array.move_batch(move_order)
                if dead_s:
                    dead += len(dead_s)
                    if dead >= number // 2:
                        result.append(count)
                        break
                count += 1
        else:
            result.append(0)
    return result


def play_mcts_fixed_batch(board=None, number=10, move_order=(0, 1, 3, 2), press_enter=False, median=False):
    """Play a game using the default mcts_fixed

    Args:
        board: the starting game state. If `None`
            is passed, a new Board is generated.
        number (int): # of lines to try for each move.
            Defaults to 10
        move_order: tuple of the 4 move indices in order.
            Defaults to (0, 1, 3, 2)
        press_enter (bool)
        median: use mcts_fixed_batch_moves instead of mcts_fixed_batch
    """
    if not board:
        board = generate_init_tiles()
    score = 0
    count = 0
    draw(board, score)
    while True:
        if press_enter and input() == 'q':
            break
        if median:
            result = mcts_fixed_batch_moves(board, number, move_order)
        else:
            result = mcts_fixed_batch(board, number, move_order)
        os.system(CLEAR)
        for i in np.argsort(result)[::-1]:
            f, s, m = move(board, i)
            if m:
                board = generate_tile(f)
                score += s
                print(ARROWS[i])
                print(result)
                draw(board, score)
                break
        else:
            # if not verbose:
            #     draw(board, score)
            # print('Game Over')
            return board, score, count
