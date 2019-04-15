import json
import os
import time
import numpy as np
import pandas as pd
from assignment4.pymdptoolbox_cs7641.src import mdptoolbox
import matplotlib.pyplot as plt

OUTPUT_DIRECTORY = './output'

if not os.path.exists(OUTPUT_DIRECTORY + '/VI'):
    os.makedirs(OUTPUT_DIRECTORY + '/VI')
if not os.path.exists(OUTPUT_DIRECTORY + '/images/VI'):
    os.makedirs(OUTPUT_DIRECTORY + '/images/VI')


class ValueIteration:
    def __init__(self, env, env_file_name, logger, gammas=np.arange(0.1, 1.0, 0.1)):
        self.env = env
        self.env_file_name = env_file_name
        self.gammas = gammas
        self.logger = logger

    def run(self):
        reward_mtx = self.env.get_reward_mtx()
        transition_mtx = self.env.get_transition_mtx()

        vi_results = {
            "gamma": list(),
            "step": list(),

            "step_time": list(),
            "time": list(),
            "values": list(),
            "variation": list(),
            "policies": list(),
            "policy_changes": list()
        }

        vi_optimal_results = {
            "gamma": list(),
            "values": list(),
            "policy": list(),
            "time": list(),
            "steps": list()
        }

        for gamma in self.gammas:  # Gamma
            # need to reinstantiate every change in gamma

            print(f'VI with GAMMA: {gamma}')
            vi = mdptoolbox.mdp.ValueIteration(transition_mtx, reward_mtx, gamma)
            vi.setVerbose()

            step = 0
            start_time = time.clock()
            total_time = 0
            while not vi.step():
                step_time = time.clock() - start_time
                total_time += step_time

                vi_results['gamma'].append(gamma)
                vi_results['step'].append(step)
                vi_results['step_time'].append(step_time)
                vi_results['time'].append(total_time)
                vi_results['values'].append(vi.V)
                vi_results['variation'].append(vi.variation)
                vi_results['policies'].append(vi.policy)
                vi_results['policy_changes'].append(vi.n_policy_changes)

                start_time = time.clock()
                step += 1

            # save optimal of gamma
            vi_optimal_results['gamma'].append(gamma)
            vi_optimal_results['values'].append(vi.V)
            vi_optimal_results['policy'].append(vi.policy)
            vi_optimal_results['time'].append(vi.time)
            vi_optimal_results['steps'].append(step)

            if self.env.env_name == 'blackjack':
                vi_blackjack_states = {
                    'sum_hand': list(),
                    'dealer_hand': list(),
                    'usable_ace': list(),
                    'values': list()
                }

                to_state_dict = self.env.get_to_state_dict()
                for i, j in zip(range(len(vi.V)), vi.V):
                    vi_blackjack_states['sum_hand'].append(to_state_dict.get(i)[0])
                    vi_blackjack_states['dealer_hand'].append(to_state_dict.get(i)[1])
                    vi_blackjack_states['usable_ace'].append(to_state_dict.get(i)[2])
                    vi_blackjack_states['values'].append(j)

                pd.DataFrame.from_dict(vi_blackjack_states).to_csv(f'{OUTPUT_DIRECTORY}/VI/states_values_{self.env_file_name}_g_{gamma}.csv')

            self.logger.info(f'____Value Iteration Results for Gamma: {gamma}____')
            self.logger.info(f'policy: {vi.policy}')
            self.logger.info(f'expected values: {vi.V}')
        if self.env.env_name == 'solly':
            generate_solly_value_function_plots(pd.DataFrame.from_dict(vi_optimal_results))
        pd.DataFrame.from_dict(vi_results).to_csv(OUTPUT_DIRECTORY+'/VI/'+self.env_file_name+'.csv')
        pd.DataFrame.from_dict(vi_optimal_results).to_csv(OUTPUT_DIRECTORY+'/VI/OPT_'+self.env_file_name+'.csv')

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

    solly_vi_vf_pd = df #pd.DataFrame.from_csv(f'{OUTPUT_DIRECTORY}/VI/OPT_solly_VI.csv')
    #solly_pi_vf_pd = df #pd.DataFrame.from_csv(f'{OUTPUT_DIRECTORY}/PI/OPT_solly_PI.csv')
    for gamma in gammas:
        #print(gamma)
        file_name_vi = f"{OUTPUT_DIRECTORY}/images/VI/solly_VI_g_{gamma}_value_func"
        #file_name_pi = f"{OUTPUT_DIRECTORY}/images/PI/solly_PI_g_{gamma}_value_func"

        # plot_solly_value_function(solly_vi_vf_pd.loc[solly_vi_vf_pd['gamma'] == gamma], "Solly VI Value Function",
        #                           file_name_vi, False)
        plot_solly_value_function(solly_vi_vf_pd.loc[solly_vi_vf_pd['gamma'] == gamma], "Solly VI Value Function",
                                  file_name_vi)