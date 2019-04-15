import argparse
import random

import gym

from gym import logger
import numpy as np

from assignment4.environments.blackjack_env import BlackjackEnv
from assignment4.environments.solly_env import SollyEnv
from assignment4.experiments import ValueIteration, PolicyIteration, QLearning
from assignment4.experiments.plotting import Plot
from assignment4.pymdptoolbox_cs7641.src import mdptoolbox

# todo include in readme how to get this with pip -e, i had to import


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    #parser.add_argument('env_id', nargs='?', default='Blackjack-v0', help='Select the environment to run')
    parser.add_argument('-l', '--learner', choices=['q_learning', 'value_iter', 'policy_iter', 'plot'])
    #parser.add_argument('-a', '--agent', required=True, choices=['solly', 'blackjack'])
    parser.add_argument('-a', '--all', choices=['True', 'False'])
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    np.random.seed(42)
    random.seed(42)

    if args.learner == 'value_iter' or args.all:
        solly_env = SollyEnv()
        blackjack_env = BlackjackEnv()

        ValueIteration(solly_env, "solly_VI", logger).run()
        ValueIteration(blackjack_env, "blackjack_VI", logger).run()

    if args.learner == 'policy_iter' or args.all:
        solly_env = SollyEnv()
        blackjack_env = BlackjackEnv()

        #TODO map states for PI and VI in csv
        PolicyIteration(solly_env, "solly_PI", logger).run()
        PolicyIteration(blackjack_env, "blackjack_PI", logger).run()

    if args.learner == 'q_learning' or args.all:
        solly_env = SollyEnv()
        blackjack_env = BlackjackEnv()

        # state = sum_hand(self.player), self.dealer[0], usable_ace(self.player)
        Q_table_blackjack = np.zeros(
            [blackjack_env.observation_space.spaces[0].n,
             blackjack_env.observation_space.spaces[1].n,
             blackjack_env.observation_space.spaces[2].n,
             blackjack_env.action_space.n])

        Q_table_solly = np.zeros(
            [len(solly_env.observation_space),
             solly_env.action_space.n]
        )

        QLearning(solly_env, 'solly_QL', logger, Q_table_solly).perform()
        QLearning(blackjack_env, 'blackjack_QL', logger, Q_table_blackjack).perform()

    if args.learner == 'plot' or args.all:
        Plot().run()
