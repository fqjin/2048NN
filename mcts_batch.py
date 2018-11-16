from board import *
# includes SIZE, DIMENSIONS, SIZE_SQRD
# includes Board class
# includes randrange, randint, numpy as np
# import keras


def mcts_batch(game, model, number = 5):
    """
    Run Monte Carlo Tree Search using NN for generating lines
    All lines combined in batch before NN prediction
    
    Args:
        game (Board): the starting game state
        model (Sequential): keras NN for selecting moves
        number (int): # of lines to try for each move. Default is 5
    Returns:
        score difference from current state for each move as a list [L, U, R, D]  
    """
    
    original_board = np.copy(game.board)
    original_score = game.score
    scores_list = np.zeros(4)
    indices  = []
    lines = []

    for i in range(4):
        if not game.moves[i]():
            game.restore(original_board, original_score)
        else:
            # This code saves one game.moves[i]() but is not pretty
            indices.extend([i]*number)
            game.generate_tile()
            lines.append(game.copy())
            game.restore(original_board, original_score)
            for _ in range(number - 1):
                game.moves[i]()
                game.generate_tile()
                lines.append(game.copy())
                game.restore(original_board, original_score)
    
    while indices:
        predictions = model.predict(np.asarray(
                        [line.board.flatten() for line in lines]))
        new_indices = []
        new_lines = []
        for index, line, pred in zip(indices, lines, predictions):
            for j in np.flipud(np.argsort(pred)):
                if line.moves[j]():
                    line.generate_tile()
                    new_indices.append(index)
                    new_lines.append(line)
                    break
            else:
                scores_list[index] += line.score
        indices = new_indices
        lines = new_lines
    
    for i in range(4):   
        if scores_list[i]:
            scores_list[i] /= number  # Calculate average final score
            scores_list[i] -= original_score  # Subtract off original score
            
    return scores_list
 
