import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer

import glob

'''
Save the data in the .csv file, save as a .npy file in ./data

2.1
'''
def one_hot_encode(cat_data):
    oneh_encoder = LabelBinarizer()

    x = oneh_encoder.fit_transform(cat_data)

    return x

def main():

    file_paths = glob.glob('unnamed_train_data/*/*.csv')
    instances = []
    labels = []

    for file in file_paths:
        print("parsing ", file)
        list.append(instances, np.genfromtxt(file, delimiter=","))
        list.append(labels, file[-7])

    instances = np.asarray(instances)
    labels = np.asarray(labels)

    label_encoder = LabelEncoder()
    #labels = one_hot_encode(labels)
    labels = label_encoder.fit_transform(labels)

    np.save('./data/instances.npy', instances)
    np.save('./data/labels.npy', labels)
    print('Instances and labels saved.')

if __name__ == "__main__":
    main()