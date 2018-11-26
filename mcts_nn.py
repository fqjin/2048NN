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

    scores = [score/number - original.score
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
            print(scores)
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

# TODO: data augmentation


def train(model, boards, results, epochs=10):
    """Trains a model using model.fit on a set of boards and results

    mcts_nn scores are converted to one hot before training

    Args:
        model: keras model to train
        boards: list of boards
        results: list of mcts_nn scores
        epochs (int): number of epochs

    """
    model.fit(np.asarray(boards),
              to_categorical(np.argmax(results, axis=-1), num_classes=4),
              epochs=epochs,
              verbose=2)


def benchmark(model, number=1000, save=False):
    """Perform a benchmark for a given model.

    Args:
        model: keras model to benchmark. Can be a model
            or the name of a model to load.
        number: number of games to run.
            Defaults to 1000.
        save (bool): Whether to save benchmark results to
            a file of the model's name. Defaults to False

    Returns: average and standard deviation of scores

    """
    if type(model) is str:
        model = get_model(model)
    lines = [Board(gen=True, draw=False) for _ in range(number)]
    dead = []
    scores = []
    counter = number//200
    while lines:
        if len(lines)//200 < counter:
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
                file.write(str(score)+'\n')
        print('saved to {}'.format(file_name))
    return scores.mean(), scores.std()


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
        train(m, b, r, epochs=50)
        del b, r
        save_model(m, '1.{}'.format(i+1))

    for i, (s, t) in enumerate(zip(high_score, max_tile)):
        print('Game {}, Score {}, Max Tile {}'.
              format(i, s, int(2**t)))


def main_test():
    for name in ['1.1', '1.3', '1.5']:
        m = get_model(name)
        ave, std = benchmark(m, save=False)
        print('Ave {}'.format(ave))
        print('Std {}'.format(std))


if __name__ == '__main__':
    main_test()
