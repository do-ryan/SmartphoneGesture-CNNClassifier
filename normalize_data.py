import numpy as np
'''
    Normalize the data, save as ./data/normalized_data.npy
    
2.4
'''

def local_normalization(instances):
# takes in instances, a .npy file of size (number of files/ instances)x(number of samples across time)x(number of channels+time)

    instances = np.load(instances)
    normalized_instances = np.empty((instances.shape[0], instances.shape[2]-1,instances.shape[1])) #don't need time channel, need number of channels as second dimension

    for i in range(instances.shape[0]): # loop through all instances

        for j in range(normalized_instances.shape[1]): #number of channels
            normalized_instances[i,j,:] = (instances[i,:,j+1] - np.average(instances[i,:,j+1]))/np.std(instances[i,:,j+1])
            # locally normalize each value in a channel of an instance

    np.save("./data/normalized_data.npy", normalized_instances)

def main():

    local_normalization('./data/instances.npy')


if __name__ == "__main__":
    main()