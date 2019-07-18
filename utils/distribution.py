"""Calculate the statistical distribuiton of tree search"""
if __name__ == '__main__':
    import numpy as np
    from board import *
    number = 1000

    origin = Board()
    play_fixed(origin)
    origin.board -= 1
    origin.score = 0
    origin.draw()

    games = []
    result = [1, 1, 1, 1]
    for i in range(4):
        temp = origin.copy()
        if temp.move(i):
            games.extend([temp.copy() for _ in range(number)])
        else:
            result[i] = 0
    for g in games:
        g.generate_tile()

    movenum = 0
    while True:
        movenum += 1
        if not movenum % 50:
            print(movenum)
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
        if result[i]:
            result[i] = np.asarray(scores[index:index+number])
            print(np.mean(result[i]))
            index += number
    np.savez('distribution/dist',
             left=result[0],
             up=result[1],
             right=result[2],
             down=result[3])
