# Use scikit-learn to grid search the batch size and epochs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.backend import set_session
from keras.optimizers import SGD, Adam
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, learning_curve
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import plot_learning_curve, save_model

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))


class NN_Keras:
    def __init__(self, model_name, dataset_name, best_params=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.best_params = best_params

        self.nn_model = None

    def create_model_optimizer(self, optimizer='adam'):
        model = Sequential()
        model.add(Dense(12, input_dim=6, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        return model

    def create_model(self, layer_sizes=None, learning_rate=0.01, loss='mean_squared_error'):
        if layer_sizes is None:
            layer_sizes = [12, 1]
        model = Sequential()
        model.add(Dense(layer_sizes[0], input_dim=299, activation='relu'))
        model.add(Dense(1, input_dim=layer_sizes[1], activation='softmax'))

        # Adam was best for twitter data
        optimizer = Adam(lr=learning_rate)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        return model

    def train(self, X_train, y_train, optimizer_tuning=False, override_best_params=False):

        if self.best_params is not None and not override_best_params:
            nn_model = KerasClassifier(build_fn=self.create_model)
            pipe = Pipeline(steps=[('scale', StandardScaler()),
                                   ('dnn', nn_model)])

            pipe.set_params(**self.best_params)

            pipe.fit(X_train, y_train)

            self.nn_model = pipe
            estimator = pipe
            try:
                save_model(self.dataset_name, estimator, 'dnn_estimator')
            except:
                print('unable to save model this way (best param)')
        else:
            if optimizer_tuning:

                nn_model = KerasClassifier(build_fn=self.create_model_optimizer, epochs=100, batch_size=10, verbose=0)

                param_grid = {
                    'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
                }
            else:
                nn_model = KerasClassifier(build_fn=self.create_model, epochs=100, batch_size=10, verbose=0)

                param_grid = {
                    'dnn__layer_sizes': [[180, 50], [150, 2], [100, 100], [50,1], [250, 20], [200, 1], [5,1]],
                    'dnn__learning_rate': [0.0005], #, 0.001, 0.05, 0.1],
                    'dnn__loss': ['mean_squared_error'], #, 'logcosh', 'binary_crossentropy',
                    #               'mean_squared_logarithmic_error'],
                    'dnn__batch_size': [10], #, 50, 100, 200, 500, 1000],
                    'dnn__epochs': [1, 10, 30, 60, 100]
                    # 'momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9] RESULTS SHOWED 0 was best..., pretty useless then
                }

            pipe = Pipeline(steps=[('scale', StandardScaler()),
                                   ('dnn', nn_model)])

            grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, n_jobs=3, verbose=2)
            grid_result = grid.fit(X_train, y_train)

            # summarize results
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))

            self.nn_model = grid
            self.best_params = grid_result.best_params_
            estimator = grid.best_estimator_
            try:
                save_model(self.dataset_name, estimator, 'dnn_estimator')
            except:
                print('unable to save model this way')

        plt = plot_learning_curve(title='Learning Curves (DNN)', estimator=estimator, X=X_train, y=y_train, cv=5,
                                  algorithm='DNN', dataset_name=self.dataset_name, model_name=self.model_name)
        plt.show()

    def predict(self, X_test):
        return self.nn_model.predict(X_test)

    def evaluate(self, X_test, y_test):
        return self.nn_model.evaluate()

    def evaluate_model(self, y_test, y_pred):
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    def get_model(self):
        return self.nn_model

    def get_best_params(self):
        return self.best_params


'''
0.509805 (0.035324) with: {'optimizer': 'SGD'}
0.778487 (0.168809) with: {'optimizer': 'RMSprop'}
0.896292 (0.005016) with: {'optimizer': 'Adagrad'}
0.781861 (0.171186) with: {'optimizer': 'Adadelta'}
0.896443 (0.001033) with: {'optimizer': 'Adam'}
0.780054 (0.169650) with: {'optimizer': 'Adamax'}
0.779481 (0.169423) with: {'optimizer': 'Nadam'}
'''
