import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import numpy as np

import pickle

batch_size = 0


def load_data():
    """Get data with labels, split into training, validation and test set."""
    with open("/home/loringit/Bulat/neuron/train.pickle", 'rb') as pickle_file:
        x_train = np.array(pickle.load(pickle_file))
    y_train = np.array(x_train)

    # with open("val.pickle", 'rb') as pickle_file:
    #   X_valid = pickle.load(pickle_file)
    # y_valid = np.array(X_valid)

    # with open("test.pickle", 'rb') as pickle_file:
    #   X_test = pickle.load(pickle_file)
    # y_test = np.array(X_test)

    global batch_size
    batch_size = x_train.shape[1]

    return dict(
        X_train=x_train,
        y_train=y_train,
        num_examples_train=x_train.shape[0],
        input_dim=batch_size,
        output_dim=batch_size,
    )


def learn_net(data):
    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=(None, batch_size),
        hidden_num_units=batch_size,  # number of units in 'hidden' layer
        hidden_nonlinearity=lasagne.nonlinearities.sigmoid,
        output_nonlinearity=lasagne.nonlinearities.elu,
        output_num_units=batch_size,  # 10 target values for the digits 0, 1, 2, ..., 9

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.1,
        update_momentum=0.9,

        max_epochs=2000,
        verbose=1,

        regression=True,
        objective_loss_function=lasagne.objectives.squared_error,
        # custom_score=("validation score", lambda x, y: np.mean(np.abs(x - y)))
    )
    # Train the network
    net1.fit(data['X_train'], data['y_train'])

    return net1


# Try the network on new data
def train():
    data = load_data()
    print("Got %i testing datasets." % len(data['X_train']))
    net = learn_net(data)
    print("NET DONE")
    # print(dir(net))
    net.save_params_to("/home/loringit/Bulat/neuron/bulik_nn")
    return {"result": "Neural Network trained"}
