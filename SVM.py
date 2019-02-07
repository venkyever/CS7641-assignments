import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import plot_learning_curve


class SVM:
    def __init__(self, model_name, dataset_name, best_params=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.best_params = best_params

        self.svm_model = SVC()  # default classifier

    def train(self, X_train, y_train, override_best_params=False):
        pipe = Pipeline(steps=[('scale', StandardScaler()),
                               ('svm', self.svm_model)])
        if self.best_params is not None and not override_best_params:
            pipe.set_params(**self.best_params)
            pipe.fit(X_train, y_train)

            self.svm_model = pipe
            estimator = pipe.named_steps['svm']
        else:
            param_grid = {
                'svm__kernel': ['rbf'], #['linear', 'poly', 'rbf', 'sigmoid'],
                'svm__degree': np.arange(1, X_train.shape[1] + 1),
                'svm__gamma': np.power(2, np.arange(-X_train.shape[1], 1, 2, dtype=float)),
                'svm__C': np.power(2, np.arange(-X_train.shape[1],0, 2, dtype=float))
            }

            # scoring = {'Accuracy': make_scorer(accuracy_score)}  # 'AUC': 'roc_auc',
            svm_clf = GridSearchCV(pipe, param_grid, iid=False, cv=5, return_train_score=True, scoring='accuracy',
                                   verbose=0)
            svm_clf.fit(X_train, y_train)
            print("Best parameter (CV score=%0.3f):" % svm_clf.best_score_)
            print(svm_clf.best_params_)

            self.svm_model = svm_clf
            self.best_params = svm_clf.best_params_
            estimator = svm_clf.best_estimator_.named_steps['svm']

        plt = plot_learning_curve(title='Learning Curves (SVM)', estimator=estimator, X=X_train, y=y_train, cv=5,
                                  algorithm='SVM', dataset_name=self.dataset_name, model_name=self.model_name)
        plt.show()

    def predict(self, X_test):
        return self.svm_model.predict(X_test)

    @staticmethod
    def evaluate_model(y_test, y_pred):
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    def get_model(self):
        return self.svm_model

    def get_best_params(self):
        return self.best_params
