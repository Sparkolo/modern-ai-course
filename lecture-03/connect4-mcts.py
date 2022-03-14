import random
from copy import deepcopy
from typing import Sequence
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import time

NONE = '.'
MAX = 'X'
MIN = 'O'
COLS = 7
ROWS = 6
N_WIN = 4
USER_CONTROLLED = True
PRE_TRAINED = False


class ArrayState:
    def __init__(self, board, heights, n_moves):
        self.board = board
        self.heights = heights
        self.n_moves = n_moves

    @staticmethod
    def init():
        board = [[NONE] * ROWS for _ in range(COLS)]
        return ArrayState(board, [0] * COLS, 0)


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
                reward = utility(state)
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
    return [i for i in range(COLS) if state.heights[i] < ROWS]

def random_branch_state(state: ArrayState) -> ArrayState:
    """ get a random state reachable from the current state """
    return result(state, random.choice(actions(state)))

def branch_states(state: ArrayState) -> Sequence[ArrayState]:
    """get all reachable states from the current state: useful for MCTS+"""
    return [result(state, a) for a in actions(state)]
    

def utility(state: ArrayState) -> float:
    """Get the winner on the current board."""
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

    for line in lines:
        str_line = "".join(line)
        if max_win in str_line:
            return 1.0
        elif min_win in str_line:
            return 0.0
    return 0.5


def terminal_test(state: ArrayState) -> bool:
    utilityTest = utility(state)
    return state.n_moves >= COLS * ROWS or utilityTest != 0.5


def printBoard(state: ArrayState):
    board = state.board
    """Print the board."""
    print('  '.join(map(str, range(COLS))))
    for y in range(ROWS):
        print('  '.join(str(board[x][y]) for x in range(COLS)))
    print()



if __name__ == '__main__':
    start_time = time.time()
    tree = MCTS()
    s = ArrayState.init()
    if PRE_TRAINED:
        tree.train_model(s, 10000)
    while not terminal_test(s):
        if(s.n_moves % 2 == 0):
            if not PRE_TRAINED:
                tree = None
                tree = MCTS(1.245)
                tree.train_model(s, 500)
            s = tree.choose(s)
        else:
            if USER_CONTROLLED:
                print("Player turn. Select a column (0-6):")
                a = int(input())
                s = result(s, a)
            else:
                if not PRE_TRAINED:
                    tree = None
                    tree = MCTS(1.245)
                    tree.train_model(s, 2000)
                s = tree.choose(s)
        printBoard(s)
    endResult = utility(s)
    if(endResult == 1.0):
        print("MAX Player wins")
    elif(endResult == 0.0):
        print("MIN Player wins")
    else:
        print("Tie")
    print("--- %s seconds ---" % (time.time() - start_time))