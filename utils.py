import datetime
from collections import defaultdict
from time import clock

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

from sklearn.model_selection import learning_curve, GridSearchCV


# Plot distributions of counts
def plot_distributions(pd_df, title, cols_to_get_distr=None):
    if cols_to_get_distr == None:
        cols_to_get_distr = pd_df.columns.values
    with plt.style.context('ggplot'):
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 22))
        axes = axes.flatten()

        for col, ax in zip(cols_to_get_distr, axes):
            ax.hist(pd_df[col], histtype='bar')
            ax.set_title(col + ' Histogram')
            ax.set(xlabel='Distribution', ylabel='count of {0}'.format(col))

        plt.suptitle(title + ' Distributions', fontsize=16, verticalalignment='baseline')
        # plt.subplots_adjust(left=0.5)

        plt.show()


def plot_distribution(pd_df, title, col_to_get_distr):
    with plt.style.context('ggplot'):
        plt.figure(figsize=(18, 15))
        plt.hist(pd_df[col_to_get_distr], histtype='bar')
        plt.title(col_to_get_distr + ' distribution')
        plt.xlabel(col_to_get_distr)
        plt.ylabel('count')

        plt.show()


def compare_counts_boxplots(positive_pd, negative_pd, title, dataset_name, positive_label, cols=None):
    # unique_col_values = df_pd[feature_column].unique()
    # print unique_col_values
    if cols is None:
        cols = positive_pd.columns.values.tolist()

    data = []
    labels = []
    for col in cols:
        data.append(positive_pd[col])
        data.append(negative_pd[col])
        labels.append(f'{positive_label}_{col}')
        labels.append(f'not_{positive_label}_{col}')

    with plt.style.context('ggplot'):
        plt.figure(figsize=(35, 20))
        plt.ylabel('Counts')
        plt.title(title)
        plt.boxplot(data, showfliers=False)
        plt.xticks(np.arange(start=1, stop=len(data) + 1), labels)
        plt.savefig(f"./figs/data/{dataset_name}_barplots.png")


def create_scatterplot_matrix(pd_df, label_column, dataset_name):
    sns.set(style="ticks")
    cols_to_plot = pd_df.columns.values
    cols_to_plot = np.delete(cols_to_plot, np.where(cols_to_plot == label_column)).tolist()
    print(cols_to_plot)
    pairplot = sns.pairplot(pd_df, hue=label_column, markers=["o", "+"], vars=cols_to_plot)
    pairplot.savefig(f"./figs/data/{dataset_name}_scatterplot_matrix.png")
    return pairplot


def plot_violin_distributions(pd_df, title, dataset_name, label_column, cols_to_plot=None):
    if cols_to_plot is None:
        cols_to_plot = pd_df.columns.values
        cols_to_plot = np.delete(cols_to_plot, np.where(cols_to_plot == label_column)).tolist()
        print(cols_to_plot)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 22))
    axes = axes.flatten()

    for col, ax in zip(cols_to_plot, axes):
        ax.set_title(col + ' violinplot')
        create_violin_plot(pd_df, col, label_column, ax)
        ax.set(ylabel='count of {0}'.format(col))

    plt.suptitle(title, fontsize=16, verticalalignment='baseline')
    # plt.subplots_adjust(left=0.5)

    plt.savefig(f"./figs/data/{dataset_name}_violinplot_matrix.png")


def create_violin_plot(pd_df, column, label_column, axes):
    sns.set(style="ticks")
    sns.violinplot(data=pd_df, x=label_column, y=column, ax=axes, kind='violin', cut=0)


def plot_learning_curve(estimator, title, X, y, algorithm, dataset_name, model_name, y_lim=None, cv=None,
                        n_jobs=None,
                        train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Generate a simple plot of the test and training learning curve.
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5)) (changed to np.linspace(0.1, 1.0, 10))
    """

    plt.figure()
    plt.title(title)
    if y_lim is not None:
        plt.ylim(*y_lim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    plt.savefig(f'./figs/learning_curve_{algorithm}_{model_name}_{dataset_name}')
    return plt


def plot_iterative_learning_curve(clfObj, trgX, trgY, tstX, tstY, params, clf_type=None, dataset=None):
    # also adopted from jontays code
    np.random.seed(42)
    if clf_type is None or dataset is None:
        raise
    cv = GridSearchCV(clfObj, n_jobs=1, param_grid=params, refit=True, verbose=10, cv=5, scoring=scorer)
    cv.fit(trgX, trgY)
    regTable = pd.DataFrame(cv.cv_results_)
    regTable.to_csv('./output/ITER_base_{}_{}.csv'.format(clf_type, dataset), index=False)
    d = defaultdict(list)
    name = list(params.keys())[0]
    for value in list(params.values())[0]:
        d['param_{}'.format(name)].append(value)
        clfObj.set_params(**{name: value})
        clfObj.fit(trgX, trgY)
        pred = clfObj.predict(trgX)
        d['train acc'].append(accuracy_score(trgY, pred))
        clfObj.fit(trgX, trgY)
        pred = clfObj.predict(tstX)
        d['test acc'].append(accuracy_score(tstY, pred))
        print(value)
    d = pd.DataFrame(d)
    d.to_csv('./output/ITERtestSET_{}_{}.csv'.format(clf_type, dataset), index=False)
    return cv

def make_timing_curve(X_train, y_train, X_test, y_test, clf, clfName, dataset):
    # 'adopted' from JonTay's code
    timing_df = defaultdict(dict)
    for fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        st = clock()
        np.random.seed(42)
        clf.fit(X_train, y_train)
        timing_df['train'][fraction] = clock() - st
        st = clock()
        clf.predict(X_test)
        timing_df['test'][fraction] = clock() - st
        print(clfName, dataset, fraction)
    timing_df = pd.DataFrame(timing_df)
    timing_df.to_csv(f'./output/{clfName}_{dataset}_timing.csv')
    #todo insert plot function
    return timing_df

def plot_model_timing(title, data_sizes, fit_scores, predict_scores, ylim=None):
    """
    Generate a simple plot of the given model timing data

    Parameters
    ----------
    title : string
        Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    data_sizes : list, array
        The data sizes

    fit_scores : list, array
        The fit/train times

    predict_scores : list, array
        The predict times

    """

    plt.close()
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Data Size (% of total)")
    plt.ylabel("Time (s)")
    fit_scores_mean = np.mean(fit_scores, axis=1)
    fit_scores_std = np.std(fit_scores, axis=1)
    predict_scores_mean = np.mean(predict_scores, axis=1)
    predict_scores_std = np.std(predict_scores, axis=1)
    plt.grid()
    plt.tight_layout()

    plt.fill_between(data_sizes, fit_scores_mean - fit_scores_std,
                     fit_scores_mean + fit_scores_std, alpha=0.2)
    plt.fill_between(data_sizes, predict_scores_mean - predict_scores_std,
                     predict_scores_mean + predict_scores_std, alpha=0.2)
    plt.plot(data_sizes, predict_scores_mean, 'o-', linewidth=1, markersize=4,
             label="Predict time")
    plt.plot(data_sizes, fit_scores_mean, 'o-', linewidth=1, markersize=4,
             label="Fit time")

    plt.legend(loc="best")
    return plt


def _save_cv_results(self):
    # TODO fix this
    regTable = pd.DataFrame(self.dt_model.cv_results_)
    regTable.to_csv(f'./output/cross_validation_{self.model_name}_{self.dataset_name}.csv',
                    index=False)

    results = pd.DataFrame(self.dt_model.cv_results_)
    components_col = 'param___n_components'
    best_clfs = results.groupby(components_col).apply(
        lambda g: g.nlargest(1, 'mean_test_score'))

    ax = plt.figure()
    best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
                   legend=False, ax=ax)
    ax.set_ylabel('Classification accuracy (val)')
    ax.set_xlabel('n_components')
    plt.savefig(f'./figs/cross_validation_{self.model_name}_{self.dataset_name}')
    plt.show()

    # todo make timing curve


def save_model(dataset_name, estimator, file_name):
    file_name = dataset_name + '_' + file_name + '_' + str(datetime.datetime.now().date()) + '.pkl'
    joblib.dump(estimator, f'./models/{file_name}')
