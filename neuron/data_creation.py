'''
from pynput.keyboard import Key, Listener
import numpy as np
import time

import pickle

data = []

num_samples = 50


def data_parser(data):
    time_series = [0]
    for i in range(1, len(data)):
        time_series.append(data[i] - data[i - 1])

    return time_series


def on_press(key):
    if key == Key.backspace:
        raise
    data.append(time.perf_counter())


def on_release(key):
    data.append(time.perf_counter())
    if key == Key.enter:
        return False


def get_data():
    with Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()
    # input()
    global data
    time_series = data_parser(data)
    data = []

    return time_series


if __name__ == '__main__':
    print(num_samples, " samples")
    print("go!")
    data_created = []
    for i in range(num_samples):
        print("sample number ", i)
        data_created.append(get_data())

    sample_size = len(data_created[0])
    for sample in data_created:
        if len(sample) != sample_size:
            raise

    print(np.array(data_created).shape)
    with open("train.pickle", 'wb') as pickle_file:
        pickle.dump(data_created, pickle_file)
'''

import pickle

def makeDataFile(data):
    with open("/home/loringit/Bulat/neuron/train.pickle", 'wb') as pickle_file:
        pickle.dump(data, pickle_file)
