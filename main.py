import argparse
from time import time

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

torch.manual_seed(1) # set seed for torch RNG
np.random.seed(1)

from model import CNN
from dataset import GestureDataset

import pandas as pd

from util import *
from test import *
from augment_data import *
from fractions import Fraction

'''
    The entry into your code. This file should include a training function and an evaluation function.
'''

def train_val_split(instances_npy_file, labels_npy_file, test_size):
    # Part 2.5
    instances = np.load(instances_npy_file)
    labels = np.load(labels_npy_file)

    return train_test_split(instances, labels, test_size=test_size, random_state=1)


def load_data(train_data, val_data, train_labels, val_labels, batch_size):
    #see assignment 2 for load_data function and dataset class

    train_dataset = GestureDataset(train_data, train_labels)
    validation_dataset = GestureDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def load_model(lr):
    model = CNN()

    loss_fnc = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer

def evaluate(model, val_loader):
    total_corr = 0

    for i, val_batch in enumerate(val_loader):

        feats, labels = val_batch
        predictions = model(feats.float())

        corr = (labels == predictions.max(1)[1])

        total_corr += int(corr.sum())

    return float(total_corr) / len(val_loader.dataset)

def main():

    config, batch_size, lr, epochs, eval_every, val_set_portion = load_config('configuration.json')
    # load hyperparameters

    instances_train_split, instances_val_split, labels_train_split, labels_val_split = train_val_split("./data/normalized_data.npy", "./data/labels.npy", test_size=val_set_portion)
    # split data into training and validation, crop out gyroscope data

    instances_train_split, labels_train_split = time_warp(instances_train_split, labels_train_split, time_warp_ratio=Fraction(4/11), slice_size=11, percent_augment=1)
    instances_train_split, labels_train_split = time_warp(instances_train_split, labels_train_split,time_warp_ratio=Fraction(9 / 11), slice_size=11, percent_augment=0.4)
    instances_train_split, labels_train_split = jitter(instances_train_split, labels_train_split, percent_augment=0.4)
    # AUGMENT DATA

    train_loader, val_loader = load_data(instances_train_split, instances_val_split, labels_train_split, labels_val_split, batch_size=batch_size)
    model, loss_fnc, optimizer = load_model(lr=lr)
    # initialize tools

    num_acc_points = int(len(train_loader.dataset) / batch_size * epochs / eval_every) + 3
    # one plot point for every eval_every steps. num_error_points is the total number of plot points

    train_acc = np.zeros(num_acc_points)
    val_acc = np.zeros(num_acc_points)
    # initialize accuracy series data arrays

    #training loop

    step = 0 #tracks batch number

    for epoch in range(epochs):
        total_train_loss = 0.0
        tot_corr = 0

        for i, batch in enumerate(train_loader, 0):
            feats, labels = batch
            #feats: batchsizex6x100, labels: batchsizex26

            step +=1

            #zero the gradients
            optimizer.zero_grad()

            predictions = model(feats.float())
            # generate predictions using current model, batchsizex26

            batch_loss = loss_fnc(input=predictions.squeeze(), target=labels.long()) # had a couple issues here with the shape of the labels, predictions
            # compute current batch loss function based on label data

            total_train_loss += batch_loss
            # update total train loss

            batch_loss.backward()
            optimizer.step()
            # adjust parameters based on gradient

            corr = (labels == predictions.max(1)[1]) # correct if label letter (a-z) matches the index of the max value from the CNN output
            tot_corr += int(corr.sum())

            if (step + 1) % eval_every == 0: # evaluate model on validation data every eval_every steps
                this_val_acc = evaluate(model, val_loader)
                print("Epoch: {}, Step {} | Loss: {} | Test acc: {}".format(epoch + 1, step + 1, float(total_train_loss) / (i + 1), this_val_acc, float(tot_corr) / (i+1)))
                # log losses, accuracies and epoch/step

                train_acc[int(step/eval_every)] = float(tot_corr) / ((i+1) * batch_size)
                val_acc[int(step/eval_every)] = this_val_acc
                # add to train plot arrays every eval_every steps

        print("train acc: ", float(tot_corr) / len(train_loader.dataset))
        # log training accuracy for each epoch

    print("Finished training")

    model_name = get_model_name(config) # create a model name based on current config
    steps_vector = np.arange(step=eval_every, start=eval_every, stop=eval_every*(val_acc.size+1)) # create step vector

    df = pd.DataFrame({"step": steps_vector, "train_acc": train_acc})
    df.to_csv("./train_val_results/train_acc_{}.csv".format(model_name), index = False)
    df = pd.DataFrame({"step": steps_vector, "val_acc": val_acc})
    df.to_csv("./train_val_results/val_acc_{}.csv".format(model_name, index = False))
    # export train/val accuracy plot arrays

    torch.save(model, 'model_CNN_doryan.pt')
    #save model

    test(model, "./assign3part3/test_data.npy")

    return

if __name__ == "__main__":
        main()