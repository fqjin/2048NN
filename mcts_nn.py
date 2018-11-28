import numpy as np
from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Flatten, Dense
from keras.utils import to_categorical

from board import Board
from board import SIZE, SIZE_SQRD, DIMENSIONS

if True:
    from numpy.random import seed
    seed(5678)
    from tensorflow import set_random_seed
    set_random_seed(5678)

# Suppress Tensorflow build from source messages
# from os import environ
# environ['TF_CPP_MIN_LOG_LEVEL']='2'


def simple_model(features=16, layers=1, name=None):
    """Creates and returns a simple keras model

    Fully connected model:
    -Flatten board grid into vector
    -`layers` # of Dense layers with `features` # of nodes
    -Dense layer with 8 nodes
    -Dense layer of size 4 using softmax for output

    Args:
        features: number of nodes for middle layers
            Defaults to 16
        layers: number of middle layers
            Defaults to 1
        name: optional name for model

    """
    model = Sequential()
    if name:
        model.name = name
    model.add(Flatten(input_shape=(SIZE, SIZE)))
    # TODO: Test effect of batch norm
    for _ in range(layers):
        model.add(Dense(features, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam')
    return model


def save_model(model, name):
    """Save the model as 'nn2048v_name_.h5'

    name should be a string of the form '3.1'
    Method sets model.name to the given name

    """
    # TODO: Just save the weights, not the whole model
    model.name = name
    model.save('models/nn2048v{}.h5'.format(name))


def get_model(name):
    """Return the model named 'nn2048v_name_.h5'"""
    return load_model('models/nn2048v{}.h5'.format(name))


def play_nn(game, model, press_enter=False):
    """Play through a game using a keras NN.

    Moves are selected purely by keras model.
    No monte carlo simulations are used.

    Args:
        game (Board): the starting game state. If `None`
            is passed, will generate a new Board.
        model: keras model to predict moves
        press_enter (bool): Whether keyboard press is
            required for each step. Defaults to False.
            Type 'q' to quit when press_enter is True.

    """
    if not game:
        game = Board(gen=True)
    while True:
        if press_enter and input() == 'q':
            break
        pred = model.predict(np.expand_dims(game.board, 0))[0]
        print(pred)
        for i in np.flipud(np.argsort(pred)):
            if game.move(i):
                game.generate_tile()
                game.draw()
                break
        else:
            print('Game Over')
            break


def mcts_nn(game, model, number=10):
    """Run tree search with keras model making lines.
    Batch implementation for efficiency.

    Args:
        game (Board): the starting game state
        model: keras model to predict moves
        number (int): # of lines to try for each move.
            Defaults to 10

    Returns:
        list: score increase for each move [Left, Up, Right, Down]

    """
    original = game.copy()
    scores = [0, 0, 0, 0]
    lines = []
    for i in range(4):
        if game.move(i):
            game.restore(original.board, original.score)
            # Sacrifice one move(i) computation
            for _ in range(number):
                temp = game.copy()
                temp.move(i)
                temp.generate_tile()
                temp.index = i
                lines.append(temp)

    dead = []
    while lines:
        preds = model.predict(np.asarray([line.board for line in lines]))
        preds = np.fliplr(np.argsort(preds))
        for line, order in zip(lines, preds):
            for j in order:
                if line.move(j):
                    line.generate_tile()
                    break
            else:
                dead.append(line)

        if dead:
            for line in dead:
                scores[line.index] += line.score
                lines.remove(line)
            dead = []

    scores = [score / number - original.score
              if score else 0.0
              for score in scores]
    return scores


def make_data(game, model, number=10, verbose=False):
    """Plays through one game using mcts_nn. Returns all
    boards and computed scores of the main line for training.

    Args:
        game (Board): the starting game state. If `None`
            is passed, a new Board is generated.
        model: keras model to predict moves
        number (int): # of lines to try for each move.
            Defaults to 10
        verbose (bool): whether to print mcts scores
            Defaults to False

    Returns:
        boards: list of boards
        results: list of mcts_nn scores

    """
    boards = []
    results = []
    if not game:
        game = Board(gen=True)
    while True:
        scores = mcts_nn(game, model, number)
        if verbose:
            print(np.trunc(scores))
        if sum(scores) > 0:
            boards.append(np.copy(game.board))
            results.append(scores)
            # TODO: only store argmax in results
        for i in np.flipud(np.argsort(scores)):
            if game.move(i):
                game.generate_tile()
                game.draw()
                break
        else:
            print('Game Over')
            break

    return boards, results


def augment(boards, results):
    """Returns augmented dataset for training.

    Dataset is augmented by a factor of x8 from the
    transformations of symmetry group D4. For boards,
    this is done by starting with 1/8 corner and:
        - getting the neighboring corner
        - flipping the quadrant along one axis
        - flipping the half-square along the other axis

    Args:
        boards: input list of boards
        results: input results. Can be either a list of
            move indices or a list of score vectors

    Returns:
        boards: list of boards
        results: list of mcts_nn scores

    Raises:
        TypeError: if results contains incorrect type

    """
    if type(results[0]) == list:
        result_type = 'vector'
    elif type(results[0]) == int:
        result_type = 'index'
    elif np.issubdtype(results[0], np.integer):
        result_type = 'index'
    else:
        raise TypeError('results contains incorrect type')

    boards = np.concatenate((boards, np.swapaxes(boards, 1, 2)))
    boards = np.concatenate((boards, np.flip(boards, axis=1)))
    boards = np.concatenate((boards, np.flip(boards, axis=2)))
    if result_type == 'index':
        # The mapping of indices preserves cycle/anticycle, so could
        # do with modular arithmetic, but a lookup table is easier
        INDEX_DICT = {
            0: [0, 1, 0, 3, 2, 1, 2, 3],
            1: [1, 0, 3, 0, 1, 2, 3, 2],
            2: [2, 3, 2, 1, 0, 3, 0, 1],
            3: [3, 2, 1, 2, 3, 0, 1, 0]
        }
        results = [INDEX_DICT[index] for index in results]
        results = np.transpose(results).flatten()

    else:
        # TODO: Implement score vector augmentation
        raise NotImplementedError

    return boards, results


def train(model, boards, results, epochs=10, do_augment=True):
    """Trains a model using model.fit on a set of boards and results

    mcts_nn scores are converted to one hot before training

    Args:
        model: keras model to train
        boards: list of boards
        results: list of mcts_nn scores
        epochs (int): number of epochs
            Defaults to 10
        do_augment (bool): whether to do data augmentation
            Defaults to True

    """
    boards = np.asarray(boards)
    results = np.argmax(results, axis=-1)
    if do_augment:
        boards, results = augment(boards, results)
    # TODO: Update model name
    return model.fit(boards,
                     to_categorical(results, num_classes=4),
                     epochs=epochs,
                     verbose=2)


def benchmark(model, number=1000, save=False):
    """Perform a benchmark for a given model.

    Args:
        model: keras model to benchmark. Can be a model
            or the name of a model to load.
        number: number of games to run.
            Defaults to 1000
        save (bool): Whether to save benchmark results to
            a file of the model's name. Defaults to False

    Returns: average and standard deviation of scores

    """
    if type(model) is str:
        model = get_model(model)
    lines = [Board(gen=True, draw=False) for _ in range(number)]
    dead = []
    scores = []
    counter = number // 200
    while lines:
        if len(lines) // 200 < counter:
            counter -= 1
            print('>{} remaining'.format(counter * 200))
        preds = model.predict(np.asarray([line.board for line in lines]))
        preds = np.fliplr(np.argsort(preds))
        for line, order in zip(lines, preds):
            for j in order:
                if line.move(j):
                    line.generate_tile()
                    break
            else:
                dead.append(line)
        if dead:
            for line in dead:
                scores.append(line.score)
                lines.remove(line)
            dead = []
    scores = np.asarray(scores)

    if save:
        file_name = 'benchmarks/nn2048v{}.csv'.format(model.name)
        with open(file_name, 'w+') as file:
            for score in scores:
                file.write(str(score) + '\n')
        print('saved to {}'.format(file_name))

    ave = scores.mean()
    std = scores.std()
    print('Ave {}'.format(ave))
    print('Std {}'.format(std))
    return ave, std


def play_mcts_nn(game, model, number=10, verbose=False):
    """Plays through one game using mcts_nn. Does not return
    board/score history, only final result.

    Args:
        game (Board): the starting game state. If `None`
            is passed, a new Board is generated.
        model: keras model to predict moves
        number (int): # of lines to try for each move.
            Defaults to 10
        verbose (bool): whether to print mcts scores
            Defaults to False

    Returns:
        final_board, final_score

    """
    if not game:
        game = Board(gen=True)
    while True:
        scores = mcts_nn(game, model, number)
        if verbose:
            print(np.trunc(scores))
        for i in np.flipud(np.argsort(scores)):
            if game.move(i):
                game.generate_tile()
                game.draw()
                break
        else:
            print('Game Over')
            break
    return game.board, game.score


def main_train():
    m = simple_model()
    m.summary()
    high_score = []
    max_tile = []

    for i in range(5):
        a = Board()
        b, r = make_data(a, m)
        high_score.append(a.score)
        max_tile.append(a.board.max())
        train(m, b, r, epochs=50, do_augment=True)
        del b, r
        save_model(m, '2.{}'.format(i + 1))

    for i, (s, t) in enumerate(zip(high_score, max_tile)):
        print('Game {}, Score {}, Max Tile {}'.
              format(i, s, int(2 ** t)))


def main_test():
    for name in ['2.1', '2.2', '2.3']:
        m = get_model(name)
        benchmark(m, save=True)


def main_mcts_benchmark():
    m = get_model('2.2')
    final_boards = []
    final_scores = []
    for _ in range(5):
        b, s = play_mcts_nn(None, m)
        final_boards.append(int(2 ** b.max()))
        final_scores.append(s)
    print('\n------\n')
    print(final_boards)
    print(final_scores)


if __name__ == '__main__':
    main_mcts_benchmark()
