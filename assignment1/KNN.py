import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
### Todo Boosting: do Ada Boost, XGBoost, lightgbm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import plot_learning_curve, save_model


class KNN:
    def __init__(self, model_name, dataset_name, best_params=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.best_params = best_params

        self.knn_model = KNeighborsClassifier()  # default classifier

    def train(self, X_train, y_train, override_best_params=False):
        pipe = Pipeline(steps=[('scale', StandardScaler()),
                               ('knn', self.knn_model)])

        if self.best_params is not None and not override_best_params:
            pipe.set_params(**self.best_params)
            pipe.fit(X_train, y_train)

            self.knn_model = pipe
            save_model(self.dataset_name, pipe, 'knn_estimator')
            estimator = pipe.named_steps['knn']
        else:
            param_grid = {
                'knn__metric': ['manhattan', 'euclidean', 'chebyshev'],
                'knn__n_neighbors': np.arange(1, 51, 3),
                'knn__weights': ['uniform', 'distance']
            }
            if self.dataset_name is 'speed_dating':
                param_grid['knn__n_neighbors'] =  np.arange(1, 200, 5)

            knn_clf = GridSearchCV(pipe, param_grid, iid=False, cv=5, return_train_score=True, scoring='accuracy',
                                    verbose=2, n_jobs=3)
            knn_clf.fit(X_train, y_train)
            print("Best parameter (CV score=%0.3f):" % knn_clf.best_score_)
            print(knn_clf.best_params_)

            self.knn_model = knn_clf
            self.best_params = knn_clf.best_params_

            save_model(self.dataset_name, knn_clf.best_estimator_, 'knn_estimator')
            estimator = knn_clf.best_estimator_.named_steps['knn']

        plt = plot_learning_curve(title='Learning Curves (K-NN)', estimator=estimator, X=X_train, y=y_train,
                                  algorithm='KNN', dataset_name=self.dataset_name, model_name=self.model_name,
                                  cv=5)
        plt.show()

    def predict(self, X_test):
        return self.knn_model.predict(X_test)

    @staticmethod
    def evaluate_model(y_test, y_pred):
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    def get_model(self):
        return self.knn_model

    def get_best_params(self):
        return self.best_params