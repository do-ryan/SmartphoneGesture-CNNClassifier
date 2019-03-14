import matplotlib.pyplot as plt
import numpy as np

from util import *

import pandas as pd

from scipy.signal import savgol_filter
'''
3.5 Plotting training and validation accuracy as a function of training steps
'''

def load_csv(config):
    """
    loads the appropriate accuracy CSV files to plot

    :param type: string denoting the type of files to load ('err' or 'loss')
    :param config: configuration dictionary
    :return: Numpy arrays for the train and test value
    """
    model_name = get_model_name(config)
    train_file = './train_val_results/train_acc_{}.csv'.format(model_name)
    val_file = './train_val_results/val_acc_{}.csv'.format(model_name)

    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)

    return train_data, val_data

def plot_accuracy(train_acc_data, val_acc_data, config):

    model_name = get_model_name(config)

    WINDOW_LENGTH = 25
    POLY_ORDER = 5

    train_acc_data = train_acc_data[train_acc_data["train_acc"] != 0]
    val_acc_data = val_acc_data[val_acc_data["val_acc"] != 0]
    # drop all rows with 0

    plt.figure()
    plt.title("Accuracy over training steps: {}".format(model_name))
    plt.plot(train_acc_data["step"], savgol_filter(train_acc_data["train_acc"], WINDOW_LENGTH, POLY_ORDER), label="Train")
    plt.plot(val_acc_data["step"], savgol_filter(val_acc_data["val_acc"], WINDOW_LENGTH, POLY_ORDER), label="Validation")
    plt.xlabel("Training Step")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig("./train_val_results/train_val_acc_{}.png".format(model_name))

    plt.show()

    return

def main():

    config, batch_size, lr, epochs, eval_every, val_set_portion = load_config('configuration.json')

    train_acc_data, val_acc_data = load_csv(config) # read accuracy csv into pd dataframes

    plot_accuracy(train_acc_data, val_acc_data, config)

    return
if __name__ == "__main__":
    main()