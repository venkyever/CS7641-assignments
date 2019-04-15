import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from assignment4.environments.blackjack_env import BlackjackEnv
from assignment4.environments.solly_env import SollyEnv
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import warnings; warnings.simplefilter('ignore')

OUTPUT_DIRECTORY = './output'

class Plot:
    def __init__(self):
        pass

    @staticmethod
    def run():
        blackjack_env = BlackjackEnv()
        solly_env = SollyEnv()

        #solly value function
        #TODO add policy on top
        generate_solly_value_function_plots()

        #black jack value function
        generate_blackjack_value_function_plots()



        # time steps vs time
        solly_vi_pd = pd.DataFrame.from_csv(f'{OUTPUT_DIRECTORY}/VI/solly_VI.csv')
        blackjack_vi_pd = pd.DataFrame.from_csv(f'{OUTPUT_DIRECTORY}/VI/blackjack_VI.csv')

        solly_pi_pd = pd.DataFrame.from_csv(f'{OUTPUT_DIRECTORY}/PI/solly_PI.csv')
        blackjack_pi_pd = pd.DataFrame.from_csv(f'{OUTPUT_DIRECTORY}/PI/blackjack_PI.csv')

        file_name = f"{OUTPUT_DIRECTORY}/images/VI/solly_time_step_gammas"
        plot_time_vs_steps('Solly Steps vs. Time VI', file_name, solly_vi_pd)

        file_name = f"{OUTPUT_DIRECTORY}/images/VI/blackjack_time_step_gammas"
        plot_time_vs_steps('Blackjack Steps vs. Time VI', file_name, blackjack_vi_pd)

        file_name = f"{OUTPUT_DIRECTORY}/images/PI/solly_time_step_gammas"
        plot_time_vs_steps('Solly Steps vs. Time PI', file_name, solly_pi_pd)

        file_name = f"{OUTPUT_DIRECTORY}/images/PI/blackjack_time_step_gammas"
        plot_time_vs_steps('Blackjack Steps vs. Time PI', file_name, blackjack_pi_pd)

        # file_name = f"{OUTPUT_DIRECTORY}/images/Q/blackjack_QL_a_{alpha}_g_{gamma}_edr_{eps_dr}_value_func"


# TODO: SOLLY Value funcction plotsV(S) at each time step with variation, q value at a each time step,
# histogram of time for gamma for pi, vi to converge. Iteration time vs gamma for q learning vi and pi, MORE CONVERGING PLOTS

def plot_solly_value_function(df, title="Value Function", file_name="value_fnc.png", q_learning=True):
    # take action value function df
    value_index = 'q_value' if q_learning else 'values'
    if q_learning:
        new_df = df.groupby(['state'])[value_index].max().reset_index()  # .join(df, on=['state', value_index])
        with plt.style.context('seaborn'):
            fig = plt.figure()
            new_df.plot.bar(x='state', y=value_index)
            plt.title(title)
            plt.xlabel('state')
            plt.ylabel('Value')
            plt.savefig(file_name + '.png')
            # plt.show()
        # new_df.plot(kind='bar', x='state', y=value_index)
        # for i in range(len(r4)):
        #     plt.text(x = r4[i]-0.5 , y = bars4[i]+0.1, s = label[i], size = 6)
    else:
        with plt.style.context('seaborn'):
            plt.bar(np.arange(7), df[value_index][0])
            plt.title(title)
            plt.xlabel('state')
            plt.ylabel('Value')
            plt.savefig(file_name + '.png')


def generate_solly_value_function_plots():
    alphas = [0.1, 0.5, 0.9]
    gammas = np.arange(0.1, 1.0, 0.1)
    eps_drs = [0.995]

    # for q-learning
    for alpha in alphas:
        for gamma in gammas:
            for eps_dr in eps_drs:
                file_name_q = f"{OUTPUT_DIRECTORY}/images/Q/solly_QL_a_{alpha}_g_{gamma}_edr_{eps_dr}_value_func"
                solly_q_pd = pd.DataFrame.from_csv(
                    f'{OUTPUT_DIRECTORY}/Q/FINAL_solly_QL_a_{alpha}_g_{gamma}_edr_{eps_dr}.csv')
                plot_solly_value_function(solly_q_pd, "Solly Q-Learning Value Function", file_name_q, True)
     #TODO this is done itnerally as the nested array gets corrupted when saved to csv
    # solly_vi_vf_pd = pd.DataFrame.from_csv(f'{OUTPUT_DIRECTORY}/VI/OPT_solly_VI.csv')
    # solly_pi_vf_pd = pd.DataFrame.from_csv(f'{OUTPUT_DIRECTORY}/PI/OPT_solly_PI.csv')
    # for gamma in gammas:
    #     file_name_vi = f"{OUTPUT_DIRECTORY}/images/VI/solly_VI_g_{gamma}_value_func"
    #     file_name_pi = f"{OUTPUT_DIRECTORY}/images/PI/solly_PI_g_{gamma}_value_func"
    #
    #     plot_solly_value_function(solly_vi_vf_pd.loc[solly_vi_vf_pd['gamma'] == gamma], "Solly VI Value Function",
    #                               file_name_vi, False)
    #     plot_solly_value_function(solly_pi_vf_pd.loc[solly_pi_vf_pd['gamma'] == gamma], "Solly PI Value Function",
    #                               file_name_pi, False)

    # solly_q_vf_pd = pd.DataFrame.from_csv(f'{OUTPUT_DIR}/Q/OPT_solly_VI.csv')



# TODO need convergence for number of time steps and time in OPT csvs COMPARE PI AND VI TOGETHER


def plot_time_vs_steps(title, file_name, df, xlabel="Steps", ylabel="Time (s)"):
    with plt.style.context('seaborn'):
        df.pivot(index='step',columns='gamma',values='time').plot()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.savefig(file_name + '.png')


def generate_blackjack_value_function_plots():
    alphas = [0.1, 0.5, 0.9]
    gammas = np.arange(0.1, 1.0, 0.1)
    eps_drs = [0.995]

    # for q-learning
    for alpha in alphas:
        for gamma in gammas:
            for eps_dr in eps_drs:
                file_name = f"{OUTPUT_DIRECTORY}/images/Q/blackjack_QL_a_{alpha}_g_{gamma}_edr_{eps_dr}_value_func"
                blackjack_q_pd = pd.DataFrame.from_csv(
                    f'{OUTPUT_DIRECTORY}/Q/FINAL_blackjack_QL_a_{alpha}_g_{gamma}_edr_{eps_dr}.csv')

                plot_blackjack_value_function(blackjack_q_pd,
                                              f"Blackjack with 500,000 steps, Gamma={gamma} Alpha={alpha}", file_name,
                                              True)

    # for VI AND PI
    for gamma in gammas:
        blackjack_vi_vf_pd = pd.DataFrame.from_csv(f'{OUTPUT_DIRECTORY}/VI/states_values_blackjack_VI_g_{gamma}.csv')
        blackjack_pi_vf_pd = pd.DataFrame.from_csv(f'{OUTPUT_DIRECTORY}/PI/states_values_blackjack_PI_g_{gamma}.csv')

        file_name_vi = f"{OUTPUT_DIRECTORY}/images/VI/blackjack_VI_value_func_g_{gamma}"
        file_name_pi = f"{OUTPUT_DIRECTORY}/images/PI/blackjack_PI_value_func_g_{gamma}"

        plot_blackjack_value_function(blackjack_vi_vf_pd, f"Blackjack for VI, Gamma={gamma}", file_name_vi, False)
        plot_blackjack_value_function(blackjack_pi_vf_pd, f"Blackjack for PI, Gamma={gamma}", file_name_pi, False)


def plot_blackjack_value_function(df, title="Value Function", file_name="value_fnc.png", q_learning=True):
    # take action value function df
    value_index = 'q_value' if q_learning else 'values'
    new_df = df.groupby(['sum_hand', 'dealer_hand', 'usable_ace'])[value_index].max().reset_index()

    df_useable_ace = new_df.loc[new_df['usable_ace'] == 1]
    df_no_useable_ace = new_df.loc[new_df['usable_ace'] == 0]

    def plt_value_function(df, title, file_name):
        # 2D-arrays from DataFrame
        x1 = np.linspace(df['sum_hand'].min(), df['sum_hand'].max(), len(np.unique(df['sum_hand'])))
        y1 = np.linspace(df['dealer_hand'].min(), df['dealer_hand'].max(), len(np.unique(df['dealer_hand'])))

        """
        x, y via meshgrid for vectorized evaluation of
        2 scalar/vector fields over 2-D grids, given
        one-dimensional coordinate arrays x1, x2,..., xn.
        """

        x2, y2 = np.meshgrid(x1, y1)
        # Interpolate unstructured D-dimensional data.
        z2 = griddata((df['sum_hand'], df['dealer_hand']), df[value_index], (x2, y2), method='cubic')

        # Ready to plot
        with plt.style.context('seaborn'):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1,
                                   cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
            ax.set_zlim(-1.01, 1.01)
            # ax.set_xlim(12,21)
            ax.set_xlabel('Player Sum')
            ax.set_ylabel('Dealer Showing')
            ax.set_zlabel('Value')

            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            ax.view_init(ax.elev, -120)

            fig.colorbar(surf)
            plt.title(title)

            plt.savefig(file_name + '.png')

    plt_value_function(df_useable_ace, 'Usable Ace ' + title, file_name + '_ace')
    plt_value_function(df_no_useable_ace, 'No Usable Ace ' + title, file_name + '_no_ace')



