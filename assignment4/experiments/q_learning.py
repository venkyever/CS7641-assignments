import json
import os
import time
import numpy as np
import pandas as pd

from assignment4.blackjack_agent import BlackjackAgent
from assignment4.solly_agent import SollyAgent

OUTPUT_DIRECTORY = './output'


if not os.path.exists(OUTPUT_DIRECTORY + '/Q'):
    os.makedirs(OUTPUT_DIRECTORY + '/Q')
if not os.path.exists(OUTPUT_DIRECTORY + '/images/Q'):
    os.makedirs(OUTPUT_DIRECTORY + '/images/Q')


class QLearning:
    def __init__(self, env, env_file_name, logger, init_q_table, max_episodes = 2000, gammas=np.arange(0.1, 1.0, 0.1)):
        self.env = env
        self.env_file_name = env_file_name
        self.gammas = gammas
        self.logger = logger
        self.max_episodes = 10001 #100001 for blackjack?
        self.init_q_table = init_q_table
        self.q_table = init_q_table

    def run(self, alpha, gamma, eps_dr):

        G = 0
        ALPHA = alpha
        GAMMA = gamma
        EPS_DR = eps_dr

        ql_results = {
            "alpha": list(),
            "gamma": list(),
            "epsilon": list(),
            "eps_decay": list(),
            "episode": list(),
            "steps": list(),
            "episode_time": list(),
            "time": list(),
            "reward": list()
        }

        ql_final_results_blackjack = {
            "sum_hand":  list(),
            "dealer_hand": list(),
            "usable_ace": list(),
            "action": list(),
            "q_value": list()
        }

        ql_final_results = dict() if self.env.get_env_name() == 'solly' else ql_final_results_blackjack

        if self.env.get_env_name() == 'solly':
            agent = SollyAgent(action_space=self.env.action_space, q_table=self.q_table, alpha=ALPHA, gamma=GAMMA,
                               epsilon=1,
                               eps_dr=EPS_DR)
        elif self.env.get_env_name() == 'blackjack':
            self.max_episodes = 500001
            agent = BlackjackAgent(action_space=self.env.action_space, q_table=self.q_table, alpha=ALPHA, gamma=GAMMA,
                               epsilon=1,
                               eps_dr=EPS_DR)

        timer, last_time = time.clock(), time.clock()
        for episode in range(1, self.max_episodes):
            done = False
            G, reward = 0, 0
            state = self.env.reset()
            steps = 0
            while not done:
                action = agent.action_selection(state=state)  # 1
                next_state, reward, done, info = self.env.step(action)  # step returns self._get_obs(), reward, done, {}


                agent.q_learn_step(state, action, reward, next_state, done)

                # Q[state, action] += alpha * (reward + gamma * (np.max(Q[state2]) * (not done)) - Q[state, action])
                G += reward
                state = next_state
                steps += 1

            # Store episode data
            ql_results['alpha'].append(alpha)
            ql_results['gamma'].append(alpha)
            ql_results['epsilon'].append(agent.get_epsilon())
            ql_results['eps_decay'].append(EPS_DR)
            ql_results['episode'].append(episode)
            ql_results['steps'].append(steps)
            ql_results['episode_time'].append(time.clock()-last_time)
            ql_results['time'].append(time.clock())
            ql_results['reward'].append(G)

            last_time = time.clock()
            agent.update_epsilon()

        self.q_table = agent.get_q_table()

        if self.env.get_env_name() == 'solly':
            ql_final_results["state"] = np.repeat(np.arange(7), 4).tolist()
            ql_final_results["actions"] = np.tile(np.arange(4), 7).tolist()
            ql_final_results["q_value"] = self.q_table.flatten().tolist()

        elif self.env.get_env_name() == 'blackjack':
            for i in range(32):
                for j in range(11):
                    for k in range(2):
                        for l in range(2):
                            ql_final_results["sum_hand"].append(i)
                            ql_final_results["dealer_hand"].append(j)
                            ql_final_results["usable_ace"].append(k)
                            ql_final_results["action"].append(l)
                            ql_final_results["q_value"].append(self.q_table[i, j, k, l])


        return ql_results, ql_final_results




    def perform(self):
        alphas = [0.1, 0.5, 0.9]
        # TODO: q_inits = ['random', 0]
        epsilon_decays = [0.995]

        for alpha in alphas:
            #for q_init in q_inits:
            for eps_dr in epsilon_decays:
                for gamma in self.gammas:
                    ql_results, ql_final_results = self.run(alpha, gamma, eps_dr)

                    pd.DataFrame.from_dict(ql_results).to_csv(f'{OUTPUT_DIRECTORY}/Q/{self.env_file_name}_a_{alpha}_g_{gamma}_edr_{eps_dr}.csv')
                    pd.DataFrame.from_dict(ql_final_results).to_csv(
                        f'{OUTPUT_DIRECTORY}/Q/FINAL_{self.env_file_name}_a_{alpha}_g_{gamma}_edr_{eps_dr}.csv')

        # TODO FIND OPTIMAL PARAMS
        #                     self.log("{}/{} Processing Q with alpha {}, q_init {}, epsilon {}, epsilon_decay {},"
        #                              " discount_factor {}".format(
        #                         runs, dims, alpha, q_init, epsilon, epsilon_decay, discount_factor
        #                     ))
        #
        #                     qs = solvers.QLearningSolver(self._details.env, self.max_episodes,
        #                                                  discount_factor=discount_factor,
        #                                                  alpha=alpha,
        #                                                  epsilon=epsilon, epsilon_decay=epsilon_decay,
        #                                                  q_init=q_init, verbose=self._verbose)
        #
        #                     stats = self.run_solver_and_collect(qs, self.convergence_check_fn)
        #
        #                     self.log("Took {} episodes".format(len(stats.steps)))
        #                     stats.to_csv('{}/Q/{}_{}_{}_{}_{}_{}.csv'.format(OUTPUT_DIRECTORY, self._details.env_name,
        #                                                                   alpha, q_init, epsilon, epsilon_decay,
        #                                                                   discount_factor))
        #                     stats.pickle_results('{}/Q/pkl/{}_{}_{}_{}_{}_{}_{}.pkl'.format(OUTPUT_DIRECTORY,
        #                                                                                     self._details.env_name,
        #                                                                                     alpha, q_init, epsilon,
        #                                                                                     epsilon_decay,
        #                                                                                     discount_factor,
        #                                                                                     '{}'), map_desc.shape,
        #                                           step_size=self.max_episodes/20.0)
        #                     stats.plot_policies_on_map('{}/images/Q/{}_{}_{}_{}_{}_{}_{}.png'.format(OUTPUT_DIRECTORY,
        #                                                                                           self._details.env_name,
        #                                                                                           alpha, q_init, epsilon,
        #                                                                                           epsilon_decay,
        #                                                                                           discount_factor,
        #                                                                                           '{}_{}'),
        #                                                map_desc, self._details.env.colors(),
        #                                                self._details.env.directions(),
        #                                                'Q-Learner', 'Episode', self._details,
        #                                                step_size=self.max_episodes / 20.0,
        #                                                only_last=True)
        #
        #                     # We have extra stats about the episode we might want to look at later
        #                     episode_stats = qs.get_stats()
        #                     episode_stats.to_csv('{}/Q/{}_{}_{}_{}_{}_{}_episode.csv'.format(OUTPUT_DIRECTORY,
        #                                                                                      self._details.env_name,
        #                                                                                      alpha, q_init, epsilon,
        #                                                                                      epsilon_decay,
        #                                                                                      discount_factor))
        #
        #                     optimal_policy_stats = self.run_policy_and_collect(qs, stats.optimal_policy)
        #                     self.log('{}'.format(optimal_policy_stats))
        #                     optimal_policy_stats.to_csv('{}/Q/{}_{}_{}_{}_{}_{}_optimal.csv'.format(OUTPUT_DIRECTORY,
        #                                                                                          self._details.env_name,
        #                                                                                          alpha, q_init, epsilon,
        #                                                                                          epsilon_decay,
        #                                                                                          discount_factor))
        #
        #                     with open(grid_file_name, 'a') as f:
        #                         f.write('"{}",{},{},{},{},{},{},{}\n'.format(
        #                             json.dumps({
        #                                 'alpha': alpha,
        #                                 'q_init': q_init,
        #                                 'epsilon': epsilon,
        #                                 'epsilon_decay': epsilon_decay,
        #                                 'discount_factor': discount_factor,
        #                             }).replace('"', '""'),
        #                             time.clock() - t,
        #                             len(optimal_policy_stats.rewards),
        #                             optimal_policy_stats.reward_mean,
        #                             optimal_policy_stats.reward_median,
        #                             optimal_policy_stats.reward_min,
        #                             optimal_policy_stats.reward_max,
        #                             optimal_policy_stats.reward_std,
        #                         ))
        #                     runs += 1
