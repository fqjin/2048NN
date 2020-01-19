from board import *


def mcts_fixed(origin, number=10):
    """Run batch tree search using a fixed move order

    A BoardArray is made for each possible move
    Lines in each array are played to end
    Returns mean score for each initial move

    Args:
        origin: the starting game state
        number (int): # of lines to simulate for each move.
            Defaults to 10

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
                dead_s = array.move_batch((0, 1, 3, 2))
                if dead_s:
                    scores.extend(dead_s)
            scores = np.array(scores)
            result.append(np.mean(np.log10(scores+s+1)))
        else:
            result.append(-1)
    return result


def mcts_fixed_moves(origin, number=10):
    """Run batch tree search using a fixed move order

    A BoardArray is made for each possible move
    Lines in each array are played to end
    Returns median move count for each initial move (xx times faster)

    Args:
        origin: the starting game state
        number (int): # of lines to simulate for each move.
            Defaults to 10

    Returns:
        list: median moves for each move [Left, Up, Right, Down]
    """
    result = []
    for i in range(4):
        board, _, moved = move(origin, i)
        if moved:
            array = BoardArray(number, board)
            array.boards = [generate_tile(b) for b in array.boards]
            count = 1
            dead = 0
            while array.boards:
                dead_s = array.move_batch((0, 1, 3, 2))
                if dead_s:
                    dead += len(dead_s)
                    if dead >= number // 2:
                        result.append(count)
                        break
                count += 1
        else:
            result.append(-1)
    return result


def mcts_fixed_min(origin, number=10):
    """Run batch tree search using a fixed move order

    A BoardArray is made for each possible move
    Lines in each array are played until the first death

    Args:
        origin: the starting game state
        number (int): # of lines to simulate for each move.
            Defaults to 10

    Returns:
        list: min moves to dead for each move
    """
    result = []
    for i in range(4):
        board, _, moved = move(origin, i)
        if moved:
            array = BoardArray(number, board)
            array.boards = [generate_tile(b) for b in array.boards]
            count = 1
            while array.boards:
                dead_s = array.move_batch((0, 1, 3, 2))
                if dead_s:
                    result.append(count)
                    break
                count += 1
        else:
            result.append(-1)
    return result


def play_mcts_fixed(board=None, number=10, press_enter=False, mode=0):
    """Play a game using the default mcts_fixed

    Args:
        board: the starting game state. If `None`
            is passed, a new Board is generated.
        number (int): # of lines to try for each move.
            Defaults to 10
        press_enter (bool)
        mode:
            0: mcts_fixed
            1: mcts_fixed_moves
            2: mcts_fixed_min
    """
    if not board:
        board = generate_init_tiles()
    score = 0
    count = 0
    if press_enter:
        draw(board, score)
    if mode == 0:
        mcts_fn = mcts_fixed
    elif mode == 1:
        mcts_fn = mcts_fixed_moves
    elif mode == 2:
        mcts_fn = mcts_fixed_min
    while True:
        if press_enter and input() == 'q':
            break
        result = mcts_fn(board, number)
        i = np.argmax(result)
        f, s, m = move(board, i)
        if not m:  # first argsort move is legal if any move is legal
            break
        board = generate_tile(f)
        score += s
        count += 1
        if press_enter:
            os.system(CLEAR)
            print(ARROWS[i])
            print(result)
            draw(board, score)
    return board, score, count
