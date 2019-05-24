"""Utility function to calculate score from
a series of boards and moves"""
import numpy as np
from board import Board
from game_dataset import GameDataset


def calc_score(boards, moves):
    games = [Board(gen=False, board=board) for board in boards]
    # Note: device attribute of games is not set
    Board.move_batch(games, moves)
    assert 0 not in [g.moved for g in games]
    return sum([g.score for g in games])


if __name__ == '__main__':
    gamepath = 'selfplay/'
    for i in range(14):
        print('---- {} ----'.format(i))
        dataset = GameDataset(gamepath, i, i+1, 'cpu', augment=False)
        b = dataset.boards.squeeze().byte()
        s = calc_score(b, dataset.moves)
        print(s)
        x = np.load(gamepath+str(i).zfill(5)+'.npz')
        try:
            print(x['score'])
        except KeyError:
            print('Adding score to data')
            np.savez(gamepath+str(i).zfill(5), boards=x['boards'], moves=x['moves'], score=s)
