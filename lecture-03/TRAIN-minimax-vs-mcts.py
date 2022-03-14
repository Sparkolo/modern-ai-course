from math import inf
import random
from copy import deepcopy
from typing import Sequence
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import time
import numpy as np

NONE = '.'
MAX = 'X'
MIN = 'O'
COLS = 7
ROWS = 6
N_WIN = 4
USE_TABLE = True

class ArrayState:
    def __init__(self, board, heights, n_moves):
        self.board = board
        self.heights = heights
        self.n_moves = n_moves

    @staticmethod
    def init():
        board = [[NONE] * ROWS for _ in range(COLS)]
        return ArrayState(board, [0] * COLS, 0)


class TranspositionTable:
    def __init__(self, size=1_000_000):
        self.size = size
        self.vals = [None] * size

    def board_str(self, state: ArrayState):
        return ''.join([''.join(c) for c in state.board])

    def put(self, state: ArrayState, utility: float):
        bstr = self.board_str(state)
        idx = hash(bstr) % self.size
        self.vals[idx] = (bstr, utility)

    def get(self, state: ArrayState):
        bstr = self.board_str(state)
        idx = hash(bstr) % self.size
        stored = self.vals[idx]
        if stored is None:
            return None
        if stored[0] == bstr:
            return stored[1]
        else:
            return None

class MCTS:
    "Monte Carlo tree searcher."
    def __init__(self, exploration_weight=1):
        self.rewards = defaultdict(int)
        self.visitCounts = defaultdict(int)
        self.children = dict()
        self.explorationWeight = exploration_weight

    def choose(self, state : ArrayState) -> ArrayState:
        if terminal_test(state):
            raise RuntimeError("Choose called on terminal state")

        if state not in self.children:
            return random_branch_state(state)

        def score(s):
            if self.visitCounts[s] == 0:
                return float('-inf') # avoid choosing unseed moves
            return self.rewards[s] / self.visitCounts[s]

        return max(self.children[state], key=score)
        

    def do_rollout(self, state : ArrayState):
        "Train for one iteration."
        path = self._select(state)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)
        

    def _select(self, state : ArrayState):
        "Find an unexplored descendent of the `state`"
        path = []
        while True:
            path.append(state)
            if state not in self.children or not self.children[state]: # unexplored or terminal path
                return path
            unexplored = self.children[state] - self.children.keys()
            if unexplored:
                nextState = unexplored.pop()
                path.append(nextState)
                return path
            state = self._uct_select(state) # go one layer deeper if no unexplored or terminal path is found
            

    def _expand(self, state : ArrayState):
        "Expand the `state` with all states reachable from it"
        if state in self.children:
            return # This board state has already been expanded
        self.children[state] = branch_states(state)


    def _simulate(self, state : ArrayState):
        "Returns the reward for a random simulation (to completion) of the `state`"
        invert_reward = state.n_moves % 2 == 0
        while True:
            if terminal_test(state):
                stateScore = utility(state)
                reward = 1.0 if stateScore == float('inf') else 0.0 if stateScore == float('-inf') else 0.5
                return 1 - reward if invert_reward else reward
            state = random_branch_state(state)
            invert_reward = not invert_reward # Switch to the other player
            

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for state in reversed(path):
            self.visitCounts[state] += 1
            self.rewards[state] += reward
            reward = 1 - reward # A reward for player MAX is the inverse for player MIN
        

    def _uct_select(self, state : ArrayState):
        "Select a child of state, balancing exploration & exploitation"
        assert all(s in self.children for s in self.children[state]) # all children of this state have been expanded already

        def uct(s):
            "Upper confidence bound for trees"
            return self.rewards[s] / self.visitCounts[s] + self.explorationWeight * math.sqrt(math.log(self.visitCounts[s]) / self.visitCounts[s])
        
        return max(self.children[state], key=uct)

    def train_model(self, state : ArrayState, num_rollouts : int):
        for _ in range(num_rollouts):
            self.do_rollout(state)

TABLE = TranspositionTable()

def result(state: ArrayState, action: int) -> ArrayState:
    """Insert in the given column."""
    assert 0 <= action < COLS, "action must be a column number"

    if state.heights[action] >= ROWS:
        raise Exception('Column is full')

    player = MAX if state.n_moves % 2 == 0 else MIN

    board = deepcopy(state.board)
    board[action][ROWS - state.heights[action] - 1] = player

    heights = deepcopy(state.heights)
    heights[action] += 1

    return ArrayState(board, heights, state.n_moves + 1)


def actions(state: ArrayState) -> Sequence[int]:
    bestActions = [3,2,4,1,5,0,6]
    return [a for a in bestActions if state.heights[a] < ROWS]

def random_branch_state(state: ArrayState) -> ArrayState:
    """ get a random state reachable from the current state """
    return result(state, random.choice(actions(state)))

def branch_states(state: ArrayState) -> Sequence[ArrayState]:
    """get all reachable states from the current state: useful for MCTS+"""
    return [result(state, a) for a in actions(state)]

def utility(state: ArrayState) -> float:
    """Get the winner on the current board."""
    global TABLE
    if(USE_TABLE):
        hashedScore = TABLE.get(state)
        if(hashedScore != None):
            return hashedScore

    board = state.board

    def diagonalsPos():
        """Get positive diagonals, going from bottom-left to top-right."""
        for di in ([(j, i - j) for j in range(COLS)] for i in range(COLS + ROWS - 1)):
            yield [board[i][j] for i, j in di if i >= 0 and j >= 0 and i < COLS and j < ROWS]

    def diagonalsNeg():
        """Get negative diagonals, going from top-left to bottom-right."""
        for di in ([(j, i - COLS + j + 1) for j in range(COLS)] for i in range(COLS + ROWS - 1)):
            yield [board[i][j] for i, j in di if i >= 0 and j >= 0 and i < COLS and j < ROWS]

    lines = board + \
            list(zip(*board)) + \
            list(diagonalsNeg()) + \
            list(diagonalsPos())

    max_win = MAX * N_WIN
    min_win = MIN * N_WIN

    curScore = 0.0

    for line in lines:
        str_line = "".join(line)
        if max_win in str_line:
            if USE_TABLE: 
                TABLE.put(state, float('inf'))
            return float('inf')
        elif min_win in str_line:
            if USE_TABLE: 
                TABLE.put(state, float('-inf'))
            return float('-inf')
        elif('.' + MAX * 3 + '.') in str_line:
            curScore += 0.7
        elif('.' + MIN * 3 + '.') in str_line:
            curScore -= 0.7
        elif ('.' + MAX * 3) in str_line or (MAX * 3 + '.') in str_line or (MAX * 2 + '.' + MAX) in str_line or (MAX + '.' + MAX * 2) in str_line:
            curScore += 0.5
        elif ('.' + MIN * 3) in str_line or (MIN * 3 + '.') in str_line or (MIN * 2 + '.' + MAX) in str_line or (MIN + '.' + MIN * 2) in str_line:
            curScore -= 0.5
        if ('..' + MAX * 2) in str_line or (MAX * 2 + '..') in str_line or ('.' + MAX * 2 + '.') in str_line or (MAX + '..' + MAX) in str_line or (MAX + '.' + MAX + '.') in str_line or ('.' + MAX + '.' + MAX) in str_line:
            curScore += 0.1
        elif ('..' + MIN * 2) in str_line or (MIN * 2 + '..') in str_line or ('.' + MIN * 2 + '.') in str_line or (MIN + '..' + MIN) in str_line or (MIN + '.' + MIN + '.') in str_line or ('.' + MIN + '.' + MIN) in str_line:
            curScore -= 0.1

    if USE_TABLE: 
        TABLE.put(state, curScore)
    return curScore


def terminal_test(state: ArrayState) -> bool:
    return state.n_moves >= COLS * ROWS or abs(utility(state)) == float('inf')


def printBoard(state: ArrayState):
    board = state.board
    """Print the board."""
    print('  '.join(map(str, range(COLS))))
    for y in range(ROWS):
        print('  '.join(str(board[x][y]) for x in range(COLS)))
    print()


def minimax(state: ArrayState, depth: int, alpha: float, beta: float, isMaxPlayer: bool):
    if depth == 0 or terminal_test(state):
        return (None, utility(state))
    
    if(isMaxPlayer):
        maxValue = float('-inf')
        bestMaxAction = random.choice(actions(state))
        for a in actions(state):
            eval = minimax(result(state, a), depth - 1, alpha, beta, False)[1]
            if eval > maxValue:
                maxValue = eval
                bestMaxAction = a
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return (bestMaxAction, maxValue)

    else:
        minValue = float('inf')
        bestMinAction = random.choice(actions(state))
        for a in actions(state):
            eval = minimax(result(state, a), depth - 1, alpha, beta, True)[1]
            if eval < minValue:
                minValue = eval
                bestMinAction = a
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return (bestMinAction, minValue)


if __name__ == '__main__':
    start_time = time.time()
    num_of_wins = {}
    for w in np.arange(1.245, 1.249, 0.005):
        win_count = 0
        for _ in range(50):
            s = ArrayState.init()
            while not terminal_test(s):
                if(s.n_moves % 2 == 0):
                    a = minimax(s, 1, float('-inf'), float('inf'), True)[0]
                    s = result(s, a)
                else:
                    tree = None
                    tree = MCTS(w)
                    tree.train_model(s, 500)
                    s = tree.choose(s)
                #printBoard(s)
            if(utility(s) == float('-inf')): 
                win_count += 1
        num_of_wins[w] = win_count
        print("{}: {}/50".format(w,win_count))
    #endResult = utility(s)
    #if(endResult == float('inf')):
    #    print("MAX Player wins")
    #elif(endResult == float('-inf')):
    #    print("MIN Player wins")
    #else:
    #    print("Tie")
    #print("--- %s seconds ---" % (time.time() - start_time))
    print(max(num_of_wins, key=num_of_wins.get))