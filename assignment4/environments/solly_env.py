import numpy as np
from gym import spaces


class SollyEnv:
    """Simple blackjack environment
    """

    def __init__(self):
        self.env_name = 'solly'
        self.action_space = spaces.Discrete(4)
        self.observation_space = np.arange(7)
        self.transition_mtx = self.set_transition_mtx()
        self.reward_mtx = self.set_reward_mtx()
        self.current_state = self.reset()

        # Restart the env

    def step(self, action):
        assert self.action_space.contains(action)

        transition_probs = self.transition_mtx[action, self.current_state, :]
        next_state = np.random.choice(self.observation_space,1, False, p=transition_probs)[0]
        done = 1 if next_state == 6 else 0

        reward = self.reward_mtx[action, self.current_state, next_state]
        self.current_state = next_state

        return self._get_obs(), reward, done, {}



    def get_transition_mtx(self):
        return self.transition_mtx

    def get_reward_mtx(self):
        return self.reward_mtx

    def set_transition_mtx(self):
        # MDP is of size a, s, s, where state and action spaces.
        transition_mtx_sleep = np.zeros(shape=(7, 7))
        transition_mtx_eat = np.zeros(shape=(7, 7))
        transition_mtx_play = np.zeros(shape=(7, 7))
        transition_mtx_walk = np.zeros(shape=(7, 7))

        # sleep transitions
        transition_mtx_sleep[0, 0] = 0.7
        transition_mtx_sleep[0, 1] = 0.3
        transition_mtx_sleep[1, 1] = 1
        transition_mtx_sleep[2, 2] = 0.8
        transition_mtx_sleep[2, 4] = 0.2
        transition_mtx_sleep[3, 3] = 0.8
        transition_mtx_sleep[3, 4] = 0.2
        transition_mtx_sleep[4, 4] = 1
        transition_mtx_sleep[5, 5] = 1
        # transition_mtx_sleep[6, 0] = 1
        transition_mtx_sleep[6, 6] = 1

        # eat transitions
        transition_mtx_eat[0, 0] = 1
        transition_mtx_eat[1, 2] = 0.71428
        transition_mtx_eat[1, 3] = 0.28572
        transition_mtx_eat[2, 2] = 1
        transition_mtx_eat[3, 3] = 1
        transition_mtx_eat[4, 4] = 1
        transition_mtx_eat[5, 6] = 1
        transition_mtx_eat[6, 6] = 1

        # play transitions
        transition_mtx_play[0, 0] = 0.9
        transition_mtx_play[0, 1] = 0.1
        transition_mtx_play[1, 1] = 1
        transition_mtx_play[2, 2] = 0.95
        transition_mtx_play[2, 4] = 0.05
        transition_mtx_play[3, 3] = 0.9
        transition_mtx_play[3, 4] = 0.1
        transition_mtx_play[4, 4] = 1
        transition_mtx_play[5, 5] = 1
        transition_mtx_play[6, 6] = 1

        # walk transitions
        transition_mtx_walk[0, 1] = 1
        transition_mtx_walk[1, 1] = 1
        transition_mtx_walk[2, 2] = 1
        transition_mtx_walk[3, 3] = 1
        transition_mtx_walk[4, 5] = 1
        transition_mtx_walk[5, 6] = 1
        transition_mtx_walk[6, 6] = 1

        transition_mtx = np.zeros((4, 7, 7))
        transition_mtx[0] = transition_mtx_sleep
        transition_mtx[1] = transition_mtx_eat
        transition_mtx[2] = transition_mtx_play
        transition_mtx[3] = transition_mtx_walk

        return transition_mtx

    def set_reward_mtx(self):
        # MDP is of size s, s, a where state and action spaces.
        reward_mtx = np.zeros(shape=(4, 7, 7))

        # rewards from morning pre-walk
        reward_mtx[0, 0, 0] = 1
        reward_mtx[0, 0, 1] = -1
        reward_mtx[1, 0, 0] = -1
        reward_mtx[2, 0, 0] = 2
        reward_mtx[2, 0, 1] = -3
        reward_mtx[3, 0, 1] = 3

        # rewards from after morning walk
        reward_mtx[1, 1, 2] = 5
        reward_mtx[1, 1, 3] = 5

        # rewards for owners at work
        reward_mtx[0, 2, 2] = 1
        reward_mtx[0, 2, 4] = 3
        reward_mtx[2, 2, 2] = 1
        reward_mtx[2, 2, 4] = 5

        # rewards for owners at home
        reward_mtx[0, 3, 3] = 1
        reward_mtx[0, 3, 4] = 3
        reward_mtx[2, 3, 3] = 3
        reward_mtx[2, 3, 4] = 3

        # rewards for evening
        reward_mtx[2, 4, 4] = 1
        reward_mtx[3, 4, 5] = 3

        # rewards for post evening walk
        reward_mtx[1, 5, 6] = 5
        reward_mtx[2, 5, 5] = 1
        reward_mtx[3, 5, 6] = 3

        # rewards evening
        # abosrbing state so nothing

        return reward_mtx

    def reset(self):
        self.current_state = 0

        return self._get_obs()

    def _get_obs(self):
        return self.current_state

    def get_env_name(self):
        return self.env_name
