import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline

### Boosting: do Ada Boost, XGBoost, lightgbm
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from DecisionTree import DecisionTreeClassifier
from utils import plot_learning_curve


class Boosting:
    def __init__(self, model_name, dataset_name, X_test, y_test, best_params=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.best_params = best_params

        self.base_dt_model = DecisionTreeClassifier(X_test, y_test)

        self.base_adabooster = AdaBoostClassifier(algorithm='SAMME.R',
                                                  learning_rate=1.,
                                                  random_state=42,
                                                  base_estimator=self.base_dt_model)  # todo:try SAMME

        self.boosting_model = None

    def train(self, X_train, y_train, override_best_params=False):
        pipe = Pipeline(steps=[('scale', StandardScaler()),
                               ('boost', self.base_adabooster)])

        if self.best_params is not None and not override_best_params:
            pipe.set_params(**self.best_params)
            pipe.fit(X_train, y_train)

            self.boosting_model = pipe
            estimator = pipe.named_steps['boost']
        else:
            param_grid = {
                'boost__n_estimators': [20, 40, 80, 100, 150, 200],  # [1, 2, 5, 10, 20, 30, 45, 60, 80, 100] #,
                'boost__base_estimator__max_depth': [1, 2, 4, 6, 8, 10, 12]
            }

            boosting_clf = GridSearchCV(pipe, param_grid, iid=False, cv=5, return_train_score=True, scoring='accuracy',
                                        verbose=0)
            boosting_clf.fit(X_train, y_train)
            print("Best parameter (CV score=%0.3f):" % boosting_clf.best_score_)
            print(boosting_clf.best_params_)

            self.boosting_model = boosting_clf
            self.best_params = boosting_clf.best_params_
            estimator = boosting_clf.best_estimator_.named_steps['boost']

        plt = plot_learning_curve(title='Learning Curves (Boosting)', estimator=estimator, X=X_train, y=y_train,
                                  algorithm='Boosting', dataset_name=self.dataset_name, model_name=self.model_name,
                                  cv=5)
        plt.show()

    def predict(self, X_test):
        return self.boosting_model.predict(X_test)

    @staticmethod
    def evaluate_model(y_test, y_pred):
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    def get_model(self):
        return self.boosting_model

    def get_best_params(self):
        return self.best_params
