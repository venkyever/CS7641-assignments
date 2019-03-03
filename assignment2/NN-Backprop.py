"""
Backprop NN training
"""
# Adapted from https://github.com/JonathanTay/CS-7641-assignment-2/blob/master/NN0.py

import sys

#
# if you are using windows you have to include full path of ABAGAIL.jar. Use backward slash ("/") instead of forward slash ("\") that you are used to in windows.
#
# by using the command sys.path.append("....")
# in case of other OSs need to append the relative path where ABAGAIL.jar . normally "../ABAGAIL/ABAGIL.jar"
sys.path.append("./ABAGAIL.jar")
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
from func.nn.activation import RELU
from base import *

# Network parameters found "optimal" in Assignment 1

# dating
# INPUT_LAYER = 299
# HIDDEN_LAYER1 = 200
# HIDDEN_LAYER2 = 50

# twitter
INPUT_LAYER = 7
HIDDEN_LAYER1 = 50
HIDDEN_LAYER2 = 5

OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 501
OUTFILE = OUTPUT_DIRECTORY + '/NN_OUTPUT/NN_{}_LOG.csv'


# {'dnn__batch_size': 100, 'dnn__epochs': 3, 'dnn__layer_sizes': [50, 5], 'dnn__learning_rate': 0.05, 'dnn__loss': 'binary_crossentropy'}

def main():
    """Run this experiment"""
    training_ints = initialize_instances(TRAIN_DATA_FILE)
    testing_ints = initialize_instances(TEST_DATA_FILE)
    validation_ints = initialize_instances(VALIDATE_DATA_FILE)
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    relu = RELU()
    # 50 and 0.000001 are the defaults from RPROPUpdateRule.java
    rule = RPROPUpdateRule(0.05, 50, 0.001)
    oa_names = ["Backprop"]
    classification_network = factory.createClassificationNetwork(
        [INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER], relu)
    train(BatchBackPropagationTrainer(data_set, classification_network, measure, rule), classification_network,
          'Backprop', training_ints, validation_ints, testing_ints, measure, TRAINING_ITERATIONS,
          OUTFILE.format('Backprop'))


if __name__ == "__main__":
    with open(OUTFILE.format('Backprop'), 'a+') as f:
        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format('iteration', 'MSE_trg', 'MSE_val', 'MSE_tst', 'acc_trg',
                                                            'acc_val', 'acc_tst', 'f1_trg', 'f1_val', 'f1_tst',
                                                            'elapsed'))
    main()
