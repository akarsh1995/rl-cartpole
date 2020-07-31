from random import choice

import numpy as np


class Agent:

    def __init__(self, i=0, j=0):
        self.i = i
        self.j = j

    @property
    def loc(self):
        return self.i, self.j

    def move_up(self):
        self.i += -1

    def move_down(self):
        self.i += 1

    def move_right(self):
        self.j += 1

    def move_left(self):
        self.j += -1


class QMatrix:
    loc_map = {
        'up': 0,
        'down': 1,
        'left': 2,
        'right': 3
    }

    def __init__(self, possible_states, possible_actions, learning_rate=0.1,
                 discount_factor=.99):
        self.q = np.zeros((possible_states, possible_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.action_index_map = {v: k for k, v in self.loc_map.items()}

    def update_matrix(self, state, action, reward, next_state):
        q = self.q
        col = self._action_col(action)
        q[state, col] = (
                (1 - self.learning_rate)
                * q[state, col]
                + self.learning_rate
                * (reward + self.discount_factor * np.max(q[next_state]))
        )

    def get_best_action(self, state):
        col = np.argmax(self.q[state])
        return self.action_index_map[col]

    def _action_col(self, action):
        return self.loc_map[action]

    def __repr__(self):
        return self.q.__repr__()


class Maze:

    def __init__(self, n_rows, n_cols):
        self.maze = np.zeros((n_rows, n_cols))
        self.mousey = Agent(0, 0)
        self.maze[-1, -1] = 10
        self.n_rows, self.n_cols = self.maze.shape
        self.set_mousey_loc_on_maze()

    def reset(self):
        self.mousey = Agent(0, 0)
        self.maze[-1, -1] = 10
        self.set_mousey_loc_on_maze()

    @property
    def state(self):
        return self.mousey.i * self.n_rows + self.mousey.j

    def set_mousey_loc_on_maze(self):
        self.maze[self.mousey.i, self.mousey.j] = 8

    def in_bounds(self, i, j):
        if (0 <= i < self.n_rows) and (0 <= j < self.n_cols):
            return True
        return False

    @property
    def start_loc(self):
        return 0, 0

    @property
    def goal_loc(self):
        return self.n_rows - 1, self.n_cols - 1

    def make_random_move(self):
        next_location = choice(self.valid_locations())
        return self.make_move(next_location)

    def make_move(self, next_location):
        current_location = self.mousey.loc
        self.maze[current_location] = 0
        getattr(self.mousey, f'move_{next_location}')()
        self.set_mousey_loc_on_maze()
        return 10 if self.goal_is_achieved() else -0.1

    def valid_locations(self):
        # conditions to be met
        # agent should be in bounds and should not move to -1 location
        valid_moves = []
        i, j = self.mousey.loc
        down = i + 1, j
        up = i - 1, j
        left = i, j - 1
        right = i, j + 1
        possibilities = {
            down: 'down',
            up: 'up',
            left: 'left',
            right: 'right',
        }
        for loc, name in possibilities.items():
            if self.valid_movement(loc):
                valid_moves.append(name)
        return valid_moves

    def valid_movement(self, loc):
        if self.in_bounds(*loc):
            return self.maze[loc] != -1
        return False

    def populate_obstacles(self, obstacles_probability=.3):
        for i_row in range(self.n_rows):
            for j_col in range(self.n_cols):
                if (((i_row, j_col) == self.start_loc)
                        or ((i_row, j_col) == self.goal_loc)):
                    continue
                if np.random.random() < obstacles_probability:
                    self.maze[i_row, j_col] = -1

    def take_user_choice(self):
        satisfied = False
        self.populate_obstacles()
        print(self)
        while not satisfied:
            keep_maze = input('Satisfied with the maze ?')
            if keep_maze.lower() == 'n':
                self.__init__(self.n_rows, self.n_cols)
                self.populate_obstacles()
                print(self)
            else:
                satisfied = True

    def goal_is_achieved(self):
        return self.goal_loc == self.mousey.loc

    def __repr__(self):
        return str(self.maze)


def main():
    n_rows = n_cols = 10
    m = Maze(n_rows, n_cols)
    q = QMatrix(n_rows ** 2, 4)
    m.take_user_choice()
    for nth_episode in range(200):
        m.reset()
        n_moves = 0
        while not m.goal_is_achieved():

            if np.random.random() > .5 or nth_episode < 100:
                actions = m.valid_locations()
                action = choice(actions)
            else:
                action = q.get_best_action(m.state)

            current_state = m.state
            reward = m.make_move(action)
            next_state = m.state
            q.update_matrix(current_state, action, reward, next_state)
            n_moves += 1
        print(f'Episode {nth_episode} finished in {n_moves}')

    n_moves = 0
    m.reset()
    while not m.goal_is_achieved():
        action = q.get_best_action(m.state)
        m.make_move(action)
        n_moves += 1
    print(f'Took {n_moves} moves after learning the q values for each state.')


if __name__ == '__main__':
    main()
