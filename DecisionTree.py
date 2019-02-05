import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import plot_learning_curve


class DecisionTreeClassifier(tree.DecisionTreeClassifier):
    """
    this class is internal to Decision Tree to allow us to fit a custom DT with post-rule pruning according.
    """

    def __init__(self, X_test, y_test, criterion="gini", splitter="best", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0., max_features=None, random_state=42,
                 max_leaf_nodes=None,
                 presort=False, alpha=0):
        super().__init__(
            criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_features, random_state, max_leaf_nodes, presort)

        self.alpha = alpha
        self.X_test = X_test
        self.y_test = y_test

    def fit(self, X_train, y_train, sample_weight=None, **kwargs):
        if sample_weight is None:
            sample_weight = np.ones(X_train.shape[0])
        super().fit(X_train, y_train, sample_weight=sample_weight)
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
    def __init__(self, model_name, dataset_name, X_test, y_test, best_params=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.best_params = best_params

        self.dt_model = DecisionTreeClassifier(X_test=X_test, y_test=y_test)

    def train(self, X_train, y_train, override_best_params=False):
        pipe = Pipeline(steps=[('scale', StandardScaler()),
                               ('dt', self.dt_model)])

        if self.best_params is not None and not override_best_params:
            pipe.set_params(**self.best_params)
            pipe.fit(X_train, y_train)

            self.dt_model = pipe
            estimator = pipe.named_steps['dt']
        else:
            alphas = np.power(10, np.arange(-4, 0, dtype=float))
            alphas = np.append(-alphas, alphas)
            alphas = np.append(alphas, 0)

            param_grid = {
                'dt__alpha': alphas,
                'dt__max_depth': np.arange(1, 32, 4),
                # 'dt__min_samples_split': np.arange(2, 12, 2),
                'dt__min_samples_leaf': np.arange(1, 10)
            }

            dt_clf = GridSearchCV(pipe, param_grid, iid=False, cv=5, return_train_score=True, scoring='accuracy',
                                  verbose=0)
            dt_clf.fit(X_train, y_train)

            print("Best parameter (CV score=%0.3f):" % dt_clf.best_score_)
            print(dt_clf.best_params_)

            self.dt_model = dt_clf
            self.best_params = dt_clf.best_params_
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
                             out_file=f'./figs/decision_tree_{self.model_name}.dot') #self.dt_model.best_estimator_.named_steps['dt'],

    def get_model(self):
        return self.dt_model

    def get_best_params(self):
        return self.best_params


