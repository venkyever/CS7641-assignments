import numpy as np
import math


class BlackjackAgent(object):
    # steps = 0
    epsilon_min = 0.01

    def __init__(self, action_space, q_table, alpha, gamma, epsilon, eps_dr):
        self.action_space = action_space
        self.q_table = q_table
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = eps_dr

    def action_selection(self, state):
        # self.steps += 1
        # epsilon = initial * math.exp(-decay * episode)
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            state = self._santize_states(state)
            action = np.argmax(self.q_table[state[0], state[1], state[2], :])
        return action

    def q_learn_step(self, state, action, reward, next_state, done):
        '''
         Q(s,a) --> r + gamma*max_act(Q(s',a))
        '''

        state = self._santize_states(state)
        next_state = self._santize_states(next_state)
        self.q_table[state[0], state[1], state[2], action] += self.alpha * (
                reward + self.gamma * (
                np.max(self.q_table[next_state[0], next_state[1], next_state[2]]) * (not done)) - self.q_table[
                    state[0], state[1], state[2], action])

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _santize_states(self, state):
        return (state[0], state[1], int(state[2]))

    def get_q_table(self):
        return self.q_table

    def get_epsilon(self):
        return self.epsilon
