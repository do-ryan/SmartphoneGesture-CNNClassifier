import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

torch.manual_seed(1) # set seed for torch RNG
np.random.seed(1)

from model import CNN
from dataset import GestureDataset

def test(model, test_data_path):
    test_data = np.load(test_data_path)
    test_dataset = GestureDataset(test_data, np.loadtxt("./assign3part3/predictions.txt"))
    test_loader = DataLoader(test_dataset, batch_size=test_data.shape[0], shuffle=False)

    total_corr = 0

    for i, test_batch in enumerate(test_loader):

        feats, labels = test_batch
        feats = np.swapaxes(feats, 1, 2) # flip around 6 and 100 to be in proper form for model input
        predictions = model(feats.float())

    predictions = predictions.max(1)[1]
    np.savetxt("./assign3part3/predictions.txt", predictions)

    return

def main():
    test(torch.load("model_CNN_doryan.pt"), "./assign3part3/test_data.npy")

if __name__ == "__main__":
        main()