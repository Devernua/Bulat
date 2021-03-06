"""
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import numpy as np
from neuron import data_creation

import matplotlib.pyplot as plt

data = data_creation.get_data()
batch_size = len(data)

attempts = 20

thres = 0.4

net = NeuralNet(
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
    objective_loss_function=lasagne.objectives.squared_error
    # custom_score=("validation score", lambda x, y: np.mean(np.abs(x - y)))
)
net.load_params_from("lev_nn")

good_answers = []
bad_answers = []
net_answer = net.predict([data])
totall2 = np.linalg.norm(data - net_answer)
good_answers.append(totall2)
print("l2: ", totall2)

for i in range(attempts):
    data = data_creation.get_data()
    if batch_size != len(data):
        print("BAD!!1!")
        i -= 1
        continue
    net_answer = net.predict([data])
    totall2 = np.linalg.norm(data - net_answer)
    if totall2 < thres:
        good_answers.append(totall2)
    else:
        good_answers.append(thres)
    print("sample ", i + 1, " of ", attempts, ", l2: ", totall2)

print("GOOD gode, now bad pls")
for i in range(attempts - 1):
    data = data_creation.get_data()
    if batch_size != len(data):
        print("BAD!!1!")
        i -= 1
        continue
    net_answer = net.predict([data])
    totall2 = np.linalg.norm(data - net_answer)
    if totall2 < thres:
        bad_answers.append(totall2)
    else:
        bad_answers.append(thres)
    print("sample ", i + 1, " of ", attempts, ", l2: ", totall2)

plt.hist(good_answers, bins='auto', alpha=0.5, facecolor='red')  # plt.hist passes it's arguments to np.histogram
plt.hist(bad_answers, bins='auto', alpha=0.5, facecolor='green')  # plt.hist passes it's arguments to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()
"""

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import numpy as np


def operate(data):
    # data = [0.0592330098, 0.140761971, 0.0757750273, 0.119381011, 0.0651519895, 0.120247006, 0.0454769731]

    batch_size = len(data)

    thres = 0.4

    net = NeuralNet(
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
        objective_loss_function=lasagne.objectives.squared_error
        # custom_score=("validation score", lambda x, y: np.mean(np.abs(x - y)))
    )
    net.load_params_from("/home/loringit/Bulat/neuron/bulik_nn")

    net_answer = net.predict([data])
    result = np.linalg.norm(data - net_answer)
    # return result < thres
    if result < thres:
        return "true"
    else:
        return "false"
