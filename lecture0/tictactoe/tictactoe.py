"""
Tic Tac Toe Player
"""

from copy import deepcopy

X     = "X"
O     = "O"
EMPTY = None

# A few game contsants
BOARD_SIZE  = 3
X_WIN_VALUE = 1
O_WIN_VALUE = -1
DRAW_VALUE  = 0
INF = X_WIN_VALUE + 1


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_num = 0
    o_num = 0
    
    for line in board:
        x_num += line.count(X)
        o_num += line.count(O)
        
    return X if x_num == o_num else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    
    for y, line in enumerate(board):
        for x, cell in enumerate(line):
            if cell == EMPTY:
                actions.add((y, x))
    
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    board_copy = deepcopy(board)
    if action in actions(board_copy):
        y, x             = action
        board_copy[y][x] = player(board_copy)
    else:
        raise ValueError('The action you tried to take is invalid! Try another one, please.')
    
    return board_copy


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Yeah, there are a lot of simpler ways to implement this, but I hate 'hardcoding' and 'magic numbers'
    # Check horizontal lines
    for line in board:
        if line.count(X) == BOARD_SIZE:
            return X
        if line.count(O) == BOARD_SIZE:
            return O
    
    # Check vertical lines
    for x in range(BOARD_SIZE):
        x_num = 0
        o_num = 0
        for y in range(BOARD_SIZE):
            if board[y][x] == X:
                x_num += 1
            elif board[y][x] == O:
                o_num += 1
                
        if x_num == BOARD_SIZE:
            return X
        if o_num == BOARD_SIZE:
            return O
        
    # Check left-top -> right-bottom diagonal
    x_num = 0
    o_num = 0
    for xy in range(BOARD_SIZE):
        if board[xy][xy] == X:
            x_num += 1
        elif board[xy][xy] == O:
            o_num += 1
            
    if x_num == BOARD_SIZE:
        return X
    if o_num == BOARD_SIZE:
        return O
    
    # Check left-bottom -> right-top diagonal
    x_num = 0
    o_num = 0
    for xy in range(BOARD_SIZE):
        if board[BOARD_SIZE - xy - 1][xy] == X:
            x_num += 1
        elif board[BOARD_SIZE - xy - 1][xy] == O:
            o_num += 1
            
    if x_num == BOARD_SIZE:
        return X
    if o_num == BOARD_SIZE:
        return O
    
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board):
        return True
    
    for line in board:
        if line.count(EMPTY):
            return False
        
    return True

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winner_sign = winner(board)
    return X_WIN_VALUE if winner_sign == X else O_WIN_VALUE if winner_sign == O else DRAW_VALUE


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    
    possible_actions = actions(board)
    optimal_action = None
    
    if player(board) == X:
        optimal_action_value = -INF
        for action in possible_actions:
            action_value = min_value(result(board, action), -INF, INF)
            
            if action_value > optimal_action_value:
                if action_value == X_WIN_VALUE:
                    return action
                
                optimal_action = action
                optimal_action_value = action_value
    else:
        optimal_action_value = INF
        for action in possible_actions:
            action_value = max_value(result(board, action), -INF, INF)
            
            if action_value < optimal_action_value:
                if action_value == O_WIN_VALUE:
                    return action
                
                optimal_action = action
                optimal_action_value = action_value
    
    return optimal_action
    
def max_value(board, alpha, beta):
    value = -INF
    
    if terminal(board):
        return utility(board)
    
    for action in actions(board):
        action_value = min_value(result(board, action), alpha, beta)

        if action_value >= beta:
            return action_value

        if action_value > alpha:
            alpha = action_value

        value = max(value, action_value)

    return value

def min_value(board, alpha, beta):
    value = INF
    
    if terminal(board):
        return utility(board)
    
    for action in actions(board):
        action_value = max_value(result(board, action), alpha, beta)

        if action_value <= alpha:
            return action_value

        if action_value < beta:
            beta = action_value
            
        value = min(value, action_value)

    return value