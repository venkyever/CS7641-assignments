import json
import os
import time
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from assignment4.pymdptoolbox_cs7641.src import mdptoolbox


OUTPUT_DIRECTORY = './output'


if not os.path.exists(OUTPUT_DIRECTORY + '/PI'):
    os.makedirs(OUTPUT_DIRECTORY + '/PI')
if not os.path.exists(OUTPUT_DIRECTORY + '/images/PI'):
    os.makedirs(OUTPUT_DIRECTORY + '/images/PI')


class PolicyIteration:
    def __init__(self, env, env_file_name, logger, gammas=np.arange(0.1, 1.0, 0.1)):
        self.env = env
        self.env_file_name = env_file_name
        self.gammas = gammas
        self.logger = logger

    def run(self):
        reward_mtx = self.env.get_reward_mtx()
        transition_mtx = self.env.get_transition_mtx()

        pi_results = {
            "gamma": list(),
            "step": list(),
            "step_time": list(),
            "time": list(),
            "values": list(),
            "policies": list() #,
            #"policy_changes": list()
        }

        pi_optimal_results = {
            "gamma": list(),
            "values": list(),
            "policy": list(),
            "time": list(),
            "steps": list()
        }

        for gamma in self.gammas:  # Gamma
            print(gamma)
            # need to reinstantiate every change in gamma
            pi = mdptoolbox.mdp.PolicyIteration(transition_mtx, reward_mtx, gamma)
            pi.setVerbose()

            step = 0
            start_time = time.clock()
            total_time = 0
            while not pi.step():
                step_time = time.clock() - start_time
                total_time += step_time

                pi_results['gamma'].append(gamma)
                pi_results['step'].append(step)
                pi_results['step_time'].append(step_time)
                pi_results['time'].append(total_time)
                pi_results['values'].append(pi.V)
                pi_results['policies'].append(pi.policy)
                #pi_results['policy_changes'].append(pi.n_policy_changes)

                step += 1
                start_time = time.clock()

            # save optimal of gamma
            pi_optimal_results['gamma'].append(gamma)
            pi_optimal_results['values'].append(pi.V)
            pi_optimal_results['policy'].append(pi.policy)
            pi_optimal_results['time'].append(pi.time)
            pi_optimal_results['steps'].append(step)

            if self.env.env_name == 'blackjack':
                pi_blackjack_states = {
                    'sum_hand': list(),
                    'dealer_hand': list(),
                    'usable_ace': list(),
                    'values': list(),
                    'policy': list()
                }

                to_state_dict = self.env.get_to_state_dict()
                for i, j, k in zip(range(len(pi.V)), pi.V, pi.policy):
                    pi_blackjack_states['sum_hand'].append(to_state_dict.get(i)[0])
                    pi_blackjack_states['dealer_hand'].append(to_state_dict.get(i)[1])
                    pi_blackjack_states['usable_ace'].append(to_state_dict.get(i)[2])
                    pi_blackjack_states['values'].append(j)
                    pi_blackjack_states['policy'].append(k)

                pd.DataFrame.from_dict(pi_blackjack_states).to_csv(f'{OUTPUT_DIRECTORY}/PI/states_values_{self.env_file_name}_g_{gamma}.csv')

            self.logger.info(f'____Policy Iteration Results for Gamma: {gamma}____')
            self.logger.info(f'policy: {pi.policy}')
            self.logger.info(f'expected values: {pi.V}')

        if self.env.env_name == 'solly':
            generate_solly_value_function_plots(pd.DataFrame.from_dict(pi_optimal_results))
        pd.DataFrame.from_dict(pi_results).to_csv(f'{OUTPUT_DIRECTORY}/PI/{self.env_file_name}.csv')
        pd.DataFrame.from_dict(pi_optimal_results).to_csv(f'{OUTPUT_DIRECTORY}/PI/OPT_{self.env_file_name}.csv')


def plot_solly_value_function(df, title="Value Function", file_name="value_fnc.png"):
    # take action value function df
        with plt.style.context('seaborn'):
            plt.bar(np.arange(7), df['values'].values[0])
            plt.title(title)
            plt.xlabel('state')
            plt.ylabel('Value')
            plt.savefig(file_name + '.png')


def generate_solly_value_function_plots(df):
    alphas = [0.1, 0.5, 0.9]
    gammas = np.arange(0.1, 1.0, 0.1)
    eps_drs = [0.995]

    #solly_vi_vf_pd = df #pd.DataFrame.from_csv(f'{OUTPUT_DIRECTORY}/VI/OPT_solly_VI.csv')
    solly_pi_vf_pd = df #pd.DataFrame.from_csv(f'{OUTPUT_DIRECTORY}/PI/OPT_solly_PI.csv')
    for gamma in gammas:
        #print(gamma)
        #file_name_vi = f"{OUTPUT_DIRECTORY}/images/VI/solly_VI_g_{gamma}_value_func"
        file_name_pi = f"{OUTPUT_DIRECTORY}/images/PI/solly_PI_g_{gamma}_value_func"

        # plot_solly_value_function(solly_vi_vf_pd.loc[solly_vi_vf_pd['gamma'] == gamma], "Solly VI Value Function",
        #                           file_name_vi, False)
        plot_solly_value_function(solly_pi_vf_pd.loc[solly_pi_vf_pd['gamma'] == gamma], "Solly PI Value Function",
                                  file_name_pi)