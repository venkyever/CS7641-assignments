import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
### Boosting: do Ada Boost, XGBoost, lightgbm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from DecisionTree import DecisionTreeClassifier
from utils import plot_learning_curve, save_model


class Boosting:
    def __init__(self, model_name, dataset_name, best_params=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.best_params = best_params

        self.base_dt_model = DecisionTreeClassifier()

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

            save_model(self.dataset_name, pipe, 'boosting_estimator')
            self.boosting_model = pipe
            estimator = pipe.named_steps['boost']
        else:
            alphas = np.power(10, np.arange(-4, 0, dtype=float))
            alphas = np.append(alphas, 0)

            param_grid = {
                'boost__n_estimators': [20, 40, 80, 100, 150, 200],  # [1, 2, 5, 10, 20, 30, 45, 60, 80, 100] #,
                # 'boost__learning_rate': [1, 0.5, 0.1, 0.05],
                'boost__base_estimator__alpha': alphas,
                'boost__base_estimator__max_depth': np.arange(1, 42, 4)
            }

            if self.dataset_name is 'speed_dating':
                param_grid['boost__base_estimator__max_depth'] = np.arange(1, 42, 4)

            boosting_clf = GridSearchCV(pipe, param_grid, iid=False, cv=5, return_train_score=True, scoring='accuracy',
                                        verbose=0)
            boosting_clf.fit(X_train, y_train)
            print("Best parameter (CV score=%0.3f):" % boosting_clf.best_score_)
            print(boosting_clf.best_params_)

            self.boosting_model = boosting_clf
            self.best_params = boosting_clf.best_params_
            save_model(self.dataset_name, boosting_clf.best_estimator_, 'boosting_estimator')
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
