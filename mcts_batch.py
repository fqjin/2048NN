from board import Board


def mcts_fixed_batch(origin, number=10, move_order=(0, 1, 2, 3)):
    """Run batch tree search using a fixed move order

    Input game is copied to a list of games.
    Each line played to end using move_batch
    Code is very similar to `play_fixed_batch`

    Args:
        origin (Board): the starting game state
        number (int): # of lines to simulate for each move.
            Defaults to 10
        move_order: tuple of the 4 move indices in order.
            Defaults to (0, 1, 2, 3)

    Returns:
        list: score increase for each move [Left, Up, Right, Down]

    """
    games = []
    result = [1, 1, 1, 1]
    for i in range(4):
        if origin.copy().move(i):
            lines = [origin.copy() for _ in range(number)]
            Board.move_batch(lines, i)
            games.extend(lines)
        else:
            result[i] = 0
    for g in games:
        g.generate_tile()

    while True:
        for i in range(4):
            subgames = [
                g for g in games if not g.dead and not g.moved
            ]
            Board.move_batch(subgames, i)
        for g in games:
            if g.moved:
                g.moved = 0
                g.generate_tile()
            else:
                g.dead = 1
        if 0 not in [g.dead for g in games]:
            break

    index = 0
    scores = [g.score for g in games]
    for i in range(4):
        if result[i] == 0:
            continue
        else:
            result[i] = sum(scores[index:index+number]) / number - origin.score
            index += number
    return result
