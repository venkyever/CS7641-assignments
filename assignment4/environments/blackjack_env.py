import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


# stolen and extended to include MDP retreival from:
#  https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py

def cmp(a, b):
    return float(a > b) - float(a < b)


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    return int(np_random.choice(deck))


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv(gym.Env):
    """Simple blackjack environment
    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with each (player and dealer) having one face up and one
    face down card.
    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.
    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.
    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html
    """

    def __init__(self, natural=False, seed=42):
        self.env_name = 'blackjack'
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self.seed(seed)
        self.state_dict, self.to_state_dict = self.make_state_dict(32, 11, 2)
        self.transition_mtx = self.set_transition_mtx()
        self.reward_mtx = self.set_reward_mtx()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural

        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1:
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return sum_hand(self.player), self.dealer[0], usable_ace(self.player)

    def reset(self):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        return self._get_obs()

    def set_transition_mtx(self):
        # action space = 2, state space = 32, 11, 2
        num_states = len(self.state_dict)

        transition_mtx = np.zeros((2, num_states, num_states))

        # we know dealers dont transition

        # set terminal states
        for i in range(11):
            for j in range(2):
                terminal_state = self.state_dict.get((0, i, j))
                transition_mtx[0, terminal_state, terminal_state] = 1

        for i in range(32):
            for j in range(11):
                for k in range(2):
                    if i > 21:
                        state = self.state_dict.get((i, j, k))
                        terminal_state = self.state_dict.get((0, j, k))
                        transition_mtx[0, state, terminal_state] = 1
                        transition_mtx[1, state, terminal_state] = 1

        # for action = 0 (dont take more cards)
        for i in range(32):  # sum of cards
            for j in range(11):  # faceup dealer
                for k in range(2):  # usable ace
                    state = self.state_dict.get((i, j, k))
                    terminal_state = self.state_dict.get((0, j, k))
                    transition_mtx[0, state, terminal_state] = 1

        # for action = 1 (take another card from [1,2,3,4,5,6,7,8,9,10,10,10,10]
        # dealers card doesnt transition
        for i in range(32):  # sum of cards current
            for j in range(32):  # sum of cards next
                for k in range(11):  # faceup dealer
                    for l in range(2):  # usable ace start
                        state = self.state_dict.get((i, k, l))
                        if i > 21:  # terminal state
                            terminal_state = self.state_dict.get((0, k, l))
                            transition_mtx[1, state, terminal_state] = 1
                        elif j <= i:
                            next_state = self.state_dict.get((j, k, l))
                            transition_mtx[1, state, next_state] = 0
                        elif (j - i) == 1 and l == 0:
                            next_state = self.state_dict.get((j, k, 1))
                            transition_mtx[1, state, next_state] = 1 / 13
                        else:
                            next_state = self.state_dict.get((j, k, l))
                            if (j - i) == 10:
                                transition_mtx[1, state, next_state] = 4 / 13
                            elif (j - i) < 10:
                                next_state = self.state_dict.get((j, k, l))
                                transition_mtx[1, state, next_state] = 1 / 13
                            else:
                                transition_mtx[1, state, next_state] = 0

        print(num_states)
        for i in range(num_states):
            if np.sum(transition_mtx[1, i, :]) != 1:
                print(self.to_state_dict.get(i), np.sum(transition_mtx[1, i, :]))

        return transition_mtx

        # TODO check that sums to 1

    def set_reward_mtx(self):
        num_states = len(self.state_dict)
        # R(A,S,S')
        reward_mtx = np.zeros((2, num_states, num_states))
        # https://wizardofodds.com/games/blackjack/appendix/1/

        # for all states where sum >21 and action = -1, for >21 and no action = 0
        for i in range(32):
            for j in range(32):
                for k in range(11):
                    for l in range(2):
                        state = self.state_dict.get((i, k, l))
                        next_state = self.state_dict.get((j, k, l)) if j - i != 1 else self.state_dict.get((j, k, 1))
                        if j > 21 > i:
                            reward_mtx[1, state, next_state] = -1

        # set expected rewards for sum 0-16
        expected_returns_a = [-0.666951, -0.292784, -0.252250, -0.211063, -0.167193, -0.153699, -0.475375, -0.510518,
                              -0.543150, -0.540430]
        for i in range(17):
            for j, k in zip(range(11), expected_returns_a):
                state = self.state_dict.get((i, j, 0))
                # set terminal states
                terminal_state = self.state_dict.get((0, j, 0))

                reward_mtx[0, state, terminal_state] = k
                # same for usable ace for all less than 7
                if i < 7:
                    # set terminal states
                    terminal_state = self.state_dict.get((0, j, 1))
                    state = self.state_dict.get((i, j, 1))
                    reward_mtx[0, state, terminal_state] = k

        # 17 sum
        expected_returns_b = [-0.478033, -0.152975, -0.117216, -0.080573, -0.044941, 0.011739, -0.106809, -0.381951,
                              -0.423154, -0.419721]
        for j, k in zip(range(11), expected_returns_b):
            state = self.state_dict.get((17, j, 0))
            # set terminal states
            terminal_state = self.state_dict.get((0, j, 0))
            reward_mtx[0, state, terminal_state] = k

            with_ace = self.state_dict.get((7, j, 1))
            # set terminal states
            terminal_state = self.state_dict.get((0, j, 1))
            reward_mtx[0, with_ace, terminal_state] = k

        # 18 sum
        expected_returns_c = [-0.100199, 0.121742, 0.148300, 0.175854, 0.199561, 0.283444, 0.399554, 0.105951,
                              -0.183163, -0.178301]
        for j, k in zip(range(11), expected_returns_c):
            state = self.state_dict.get((18, j, 0))
            # set terminal states
            terminal_state = self.state_dict.get((0, j, 0))
            reward_mtx[0, state, terminal_state] = k

            with_ace = self.state_dict.get((8, j, 1))
            # set terminal states
            terminal_state = self.state_dict.get((0, j, 1))
            reward_mtx[0, with_ace, terminal_state] = k

        # 19 sum
        expected_returns_d = [0.277636, 0.386305, 0.404363, 0.423179, 0.439512, 0.495977, 0.615976, 0.593854, 0.287597,
                              0.063118]
        for j, k in zip(range(11), expected_returns_d):
            state = self.state_dict.get((19, j, 0))
            # set terminal states
            terminal_state = self.state_dict.get((0, j, 0))
            reward_mtx[0, state, terminal_state] = k

            with_ace = self.state_dict.get((9, j, 1))
            # set terminal states
            terminal_state = self.state_dict.get((0, j, 1))
            reward_mtx[0, with_ace, terminal_state] = k

        # 20 sum
        expected_returns_e = [0.655470, 0.639987, 0.650272, 0.661050, 0.670360, 0.703959, 0.773227, 0.791815, 0.758357,
                              0.554538]
        for j, k in zip(range(11), expected_returns_e):

            state = self.state_dict.get((20, j, 0))
            reward_mtx[0, state, terminal_state] = k
            # set terminal states
            terminal_state = self.state_dict.get((0, j, 0))
            with_ace = self.state_dict.get((10, j, 1))
            # set terminal states
            terminal_state = self.state_dict.get((0, j, 1))
            reward_mtx[0, with_ace, terminal_state] = k

        # 21 sum
        expected_returns_f = [0.922194, 0.882007, 0.885300, 0.888767, 0.891754, 0.902837, 0.925926, 0.930605, 0.939176,
                              0.962624]
        for j, k in zip(range(11), expected_returns_f):
            state = self.state_dict.get((21, j, 0))
            # set terminal states
            terminal_state = self.state_dict.get((0, j, 0))
            reward_mtx[0, state, terminal_state] = k

            with_ace = self.state_dict.get((11, j, 1))
            # set terminal states
            terminal_state = self.state_dict.get((0, j, 1))
            reward_mtx[0, with_ace, terminal_state] = k

        return reward_mtx

        # for all states >21 payoff is -1

    def get_reward_mtx(self):
        return self.reward_mtx

    def get_transition_mtx(self):
        return self.transition_mtx

    @staticmethod
    def make_state_dict(sum_of_cards, dealer_face_up, usable_ace):
        state_dict = {}
        to_state_dict = {}
        key = 0
        for i in range(sum_of_cards):
            for j in range(dealer_face_up):
                for k in range(usable_ace):
                    state_dict[(i, j, k)] = key
                    to_state_dict[key] = (i, j, k)
                    key += 1

        return state_dict, to_state_dict

    def get_env_name(self):
        return self.env_name

    def get_to_state_dict(self):
        return self.to_state_dict
