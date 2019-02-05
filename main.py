import argparse
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)
random.seed(42)

from DataExploration import DataExploration
from DecisionTree import DecisionTree
from NeuralNetwork import DNN

if __name__ == "__main__":
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', required=True,
                        choices=["get_data","dt", 'dnn'])
    io_args = parser.parse_args()
    method = io_args.method

    if method == "get_data":
        # need to shuffle to respect iid principal for when we split the data

        twitter_df = DataExploration().prepare_twitter_df()
        twitter_df = shuffle(twitter_df)

        X = twitter_df.drop('is_spam', axis=1).drop('created_at', axis=1).drop('collected_at', axis=1).drop(
            'series_of_num_following', axis=1)
        y = twitter_df['is_spam']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train_nn, X_validation_nn, y_train_nn, y_validation_nn = train_test_split(X_train, y_train,
                                                                                    test_size=0.125)  # size = 0.1 of orginal

        # print(X_train)
        # print(twitter_df.head())
        # print(twitter_df.shape)

    if method == "dt":
        dt_clf = DecisionTree(model_name='dt_wip0', dataset_name='spam_twitter')
        dt_clf.train(X_train, y_train)
        y_pred = dt_clf.predict(X_test)
        dt_clf.evaluate_model(y_test=y_test, y_pred=y_pred)

        # dt_clf.get_model().best_estimator_.named_steps['dt'].tree_.max_depth

    if method == "dnn":
        dnn = DNN(n_classes=2, model_name='dnn_wip_1', dataset_name='spam_twitter')
        dnn.train(
            learning_rate=0.05,
            steps=1000,
            batch_size=30,
            hidden_units=[30, 13],
            training_examples=X_train_nn,
            training_targets=y_train_nn,
            validation_examples=X_validation_nn,
            validation_targets=y_validation_nn)

        y_pred = dnn.predict(X_test, y_test)