import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import plot_learning_curve, save_model


class DecisionTreeClassifier(tree.DecisionTreeClassifier):
    """
    this class is internal to Decision Tree to allow us to fit a custom DT with post-rule pruning according.
    """

    def __init__(self, criterion="gini", splitter="best", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0., max_features=None, random_state=42,
                 max_leaf_nodes=None,
                 presort=False, alpha=0):
        super().__init__(
            criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_features, random_state, max_leaf_nodes, presort)

        self.alpha = alpha
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        self.test_weights = None
        self.training_weights = None

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.training_weights = sample_weight.copy()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in sss.split(self.X_train, self.y_train):
            self.X_test = self.X_train[test_index]
            self.y_test = self.y_train.iloc[test_index]
            self.X_train = self.X_train[train_index]
            self.y_train = self.y_train.iloc[train_index]
            self.test_weights = sample_weight[test_index]
            self.training_weights = sample_weight[train_index]
        super().fit(self.X_train, self.y_train, sample_weight=self.training_weights, check_input=check_input,
                    X_idx_sorted=X_idx_sorted)
        self.prune()
        return self

        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/helpers.py
        # Based on the pruning technique from Mitchell

    def remove_subtree(self, root):
        """
        Clean up
        :param root:
        :return:
        """
        tmp_tree = self.tree_
        visited, stack = set(), [root]
        while stack:
            v = stack.pop()
            visited.add(v)
            left = tmp_tree.children_left[v]
            right = tmp_tree.children_right[v]
            if left >= 0:
                stack.append(left)
            if right >= 0:
                stack.append(right)
        for node in visited:
            tmp_tree.children_left[node] = -1
            tmp_tree.children_right[node] = -1
        return

    def prune(self):
        c = 1 - self.alpha
        if self.alpha <= -1:  # Early exit
            return self
        tmp_tree = self.tree_
        best_score = self.score(self.X_test, self.y_test)
        candidates = np.flatnonzero(tmp_tree.children_left >= 0)
        for candidate in reversed(candidates):  # Go backwards/leaves up
            if tmp_tree.children_left[candidate] == tmp_tree.children_right[candidate]:  # leaf node. Ignore
                continue
            left = tmp_tree.children_left[candidate]
            right = tmp_tree.children_right[candidate]
            tmp_tree.children_left[candidate] = tmp_tree.children_right[candidate] = -1
            score = self.score(self.X_test, self.y_test)
            if score >= c * best_score:
                best_score = score
                self.remove_subtree(candidate)
            else:
                tmp_tree.children_left[candidate] = left
                tmp_tree.children_right[candidate] = right
        assert (self.tree_.children_left >= 0).sum() == (self.tree_.children_right >= 0).sum()

        return self


class DecisionTree:
    def __init__(self, model_name, dataset_name, best_params=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.best_params = best_params

        self.dt_model = DecisionTreeClassifier()

    def train(self, X_train, y_train, override_best_params=False):
        pipe = Pipeline(steps=[('scale', StandardScaler()),
                               ('dt', self.dt_model)])

        if self.best_params is not None and not override_best_params:
            pipe.set_params(**self.best_params)
            pipe.fit(X_train, y_train)

            save_model(self.dataset_name, pipe, 'dt_estimator')

            self.dt_model = pipe
            estimator = pipe.named_steps['dt']
        else:
            alphas = np.power(10, np.arange(-4, 0, dtype=float))
            alphas = np.append(alphas, 0)

            param_grid = {
                'dt__alpha': alphas,
                'dt__max_depth': np.arange(1, 26, 4),
                'dt__min_samples_leaf': np.arange(1, 10)
            }

            if self.dataset_name is 'speed_dating':
                param_grid['dt__max_depth'] = np.arange(1, 42, 4)

            dt_clf = GridSearchCV(pipe, param_grid, iid=False, cv=5, return_train_score=True, scoring='accuracy',
                                  verbose=0)
            dt_clf.fit(X_train, y_train)

            print("Best parameter (CV score=%0.3f):" % dt_clf.best_score_)
            print(dt_clf.best_params_)

            self.dt_model = dt_clf
            self.best_params = dt_clf.best_params_
            save_model(self.dataset_name, dt_clf.best_estimator_, 'dt_estimator')
            estimator = dt_clf.best_estimator_.named_steps['dt']

        plt = plot_learning_curve(title='Learning Curves (DT)', estimator=estimator, X=X_train, y=y_train,
                                  algorithm='DT', dataset_name=self.dataset_name, model_name=self.model_name,
                                  cv=5)

        plt.show()
        # self._save_cv_results()

    def predict(self, X_test):
        return self.dt_model.predict(X_test)

    def evaluate_model(self, y_test, y_pred):
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        tree.export_graphviz(self.dt_model.named_steps['dt'],
                             out_file=f'./figs/decision_tree_{self.model_name}.dot')  # self.dt_model.best_estimator_.named_steps['dt'],

    def get_model(self):
        return self.dt_model

    def get_best_params(self):
        return self.best_params
