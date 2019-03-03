from __future__ import print_function

import glob
import math
import os

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

# todo get back to this: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
class DNN:
    # https://colab.research.google.com/notebooks/mlcc/multi-class_classification_of_handwritten_digits.ipynb#scrollTo=kdNTx8jkPQUx
    def __init__(self, n_classes, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.n_classes = n_classes
        self.dnn_model = None

    def train(self,
              learning_rate,
              steps,
              batch_size,
              hidden_units,
              training_examples,
              training_targets,
              validation_examples,
              validation_targets):
        """

        :param learning_rate: A `float`, the learning rate to use.
        :param steps: A non-zero `int`, the total number of training steps. A training step
           consists of a forward and backward pass using a single batch.
        :param batch_size: A non-zero `int`, the batch size.
        :param hidden_units: A `list` of int values, specifying the number of neurons in each layer.
        :param training_examples: A `DataFrame` containing the training features.
        :param training_targets: A `DataFrame` containing the training labels.
        :param validation_examples: A `DataFrame` containing the validation features.
        :param validation_targets: A `DataFrame` containing the validation labels.
        :return: The trained `DNNClassifier` object.
        """

        periods = 10
        # Caution: input pipelines are reset with each call to train.
        # If the number of steps is small, your model may never see most of the data.
        # So with multiple `.train` calls like this you may want to control the length
        # of training with num_epochs passed to the input_fn. Or, you can do a really-big shuffle,
        # or since it's in-memory data, shuffle all the data in the `input_fn`.
        steps_per_period = steps / periods

        # Create the input functions.
        predict_training_input_fn = tf.estimator.inputs.pandas_input_fn(
            x=training_examples, y=training_targets, batch_size=batch_size, num_epochs=1, shuffle=False)
        predict_validation_input_fn = tf.estimator.inputs.pandas_input_fn(
            x=validation_examples, y=validation_targets, batch_size=batch_size, num_epochs=1, shuffle=False)
        training_input_fn = tf.estimator.inputs.pandas_input_fn(
            x=training_examples, y=training_targets, batch_size=batch_size, num_epochs=1, shuffle=False)

        # Create feature columns.
        # TODO CHANGE IF NON NUMERIC FEATURE COLUMNS
        feature_columns = [tf.feature_column.numeric_column(key=feature_name) for feature_name in
                           training_examples.columns.values]

        # Create a DNNClassifier object.
        my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
        classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            n_classes=self.n_classes,
            hidden_units=hidden_units,
            activation_fn=tf.nn.relu,
            optimizer=my_optimizer,
            config=tf.contrib.learn.RunConfig(keep_checkpoint_max=0),
            model_dir='./dnn_dir/'
        )

        # Train the model, but do so inside a loop so that we can periodically assess
        # loss metrics.
        print("Training model...")
        print("LogLoss error (on validation data):")
        training_errors = []
        validation_errors = []
        for period in range(0, periods):
            # Train the model, starting from the prior state.
            classifier.train(
                input_fn=training_input_fn,
                steps=steps_per_period
            )

            # Take a break and compute probabilities.
            training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
            training_probabilities = np.array([item['probabilities'] for item in training_predictions])
            training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
            training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 2)

            validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
            validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
            validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
            validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 2)

            # Compute training and validation errors.
            training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
            validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
            # Occasionally print the current loss.
            print("  period %02d : %0.2f" % (period, validation_log_loss))
            # Add the loss metrics from this period to our list.
            training_errors.append(training_log_loss)
            validation_errors.append(validation_log_loss)
        print("Model training finished.")
        # Remove event files to save disk space.
        _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

        # Calculate final predictions (not probabilities, as above).
        final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
        final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

        accuracy = metrics.accuracy_score(validation_targets, final_predictions)
        print("Final accuracy (on validation data): %0.2f" % accuracy)

        # Output a graph of loss metrics over periods.
        plt.ylabel("LogLoss")
        plt.xlabel("Periods")
        plt.title("LogLoss vs. Periods")
        plt.plot(training_errors, label="training")
        plt.plot(validation_errors, label="validation")
        plt.legend()
        plt.savefig(f'./figs/{self.dataset_name}_{self.model_name}')
        plt.show()

        # Output a plot of the confusion matrix.
        cm = metrics.confusion_matrix(validation_targets, final_predictions)
        # Normalize the confusion matrix by row (i.e by the number of samples
        # in each class).
        # cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        # ax = sns.heatmap(cm_normalized, cmap="bone_r")
        # ax.set_aspect(1)
        print(cm)

        self.dnn_model = classifier

    def predict(self, X_test, y_test=None):
        if self.dnn_model is None:
            raise FileNotFoundError

        predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
            X_test, batch_size=100, shuffle=False)
        prediction = self.dnn_model.predict(input_fn=predict_test_input_fn)
        prediction = np.array([item['class_ids'][0] for item in prediction])

        if y_test is not None:
            accuracy = metrics.accuracy_score(y_test, prediction)
            print("Accuracy on test data: %0.2f" % accuracy)

        return prediction

    def get_model(self):
        return self.dnn_model

    # todo: add restoration from checkpoints, tensorboard interface, #
    # todo: cv with hidden layers, input sizes, and epochs, fix confusion matrix
