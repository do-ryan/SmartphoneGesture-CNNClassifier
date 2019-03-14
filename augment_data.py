import numpy as np
from fractions import Fraction
import scipy as sp

'''

Load up an .npy file and save a data augmented file
By default: take in data output of normalize_data.py

'''

def time_warp(gesture_data, gesture_data_labels, time_warp_ratio, slice_size, percent_augment):
    # takes in data of shape (number of gesture instances x num channels x num time samples)
    # returns data of same shape but time warped. takes time slices from gesture_data and alternates speed up/ slow down based on time_warp_ratio.
    # slice_size is size of alternative slices to be sped up/ slowed down
    # time_warp_ratio is a Fraction that represents the speed up/slow down multiplier. Fraction * slice_size must be an integer

    curr_index = 0
    curr_warp_index = 0
    speed_up = True
    time_warped_data = np.empty(gesture_data.shape)

    for i in range(int(gesture_data.shape[2]/(slice_size))):
        # repeats for every time slice in original data
        # speed/slow period is 2* time slice

        x = np.arange(0, slice_size+1, 1)
        # create x 3D array with z size slice_size for interpolation function. (x,y) = (num samples, num channels)
        y = gesture_data[:, :, curr_index:(curr_index + slice_size)+1]
        f = sp.interpolate.interp1d(x, y, axis = 2)
        # create interpolation function of original time slice

        if speed_up:
            x= np.arange(0, float(slice_size+1/time_warp_ratio), float(1/time_warp_ratio))
            # create x 1D array with length slice_size*time_warp_ratio (creates a new smaller slice)
            time_warped_data[:,:,curr_index:int(curr_index+slice_size*time_warp_ratio)+1] = f(x)
            # compress (speed up) using new x.

            curr_warp_index += slice_size*time_warp_ratio

        else:
            x = np.arange(0, float(slice_size+1/(2-time_warp_ratio)), float(1/(2-time_warp_ratio)))
            # create x 1d array with bigger length (slice*(2-time_warpratio). size(big slice + small slice) = size(2* orig slice)
            time_warped_data[:, :, int(curr_index - slice_size *(1 - time_warp_ratio)):int(curr_index + slice_size)+1] = f(x)
            # dilate(slow down) using new x.

            curr_warp_index += slice_size *(2 - time_warp_ratio)


        speed_up = not speed_up #alternates speed up and slow down
        curr_index += slice_size # curr_index tracks original data
    return np.vstack((gesture_data, time_warped_data[0:int(percent_augment*time_warped_data.shape[0])])), np.concatenate((gesture_data_labels, gesture_data_labels[0:int(percent_augment*gesture_data_labels.shape[0])]))

def jitter(gesture_data, gesture_data_labels, percent_augment):
    # takes in data of shape (number of gesture instances x num channels x num time samples)
    # takes corresponding labels
    # returns gesture_data concatenated with jittered data of same shape, size = percent_augment* size of gesture_data

    noise = np.random.normal(loc=0, scale=0.01, size=gesture_data.shape)
    jittered_data = gesture_data + noise

    return np.vstack((gesture_data, jittered_data[0:int(percent_augment*jittered_data.shape[0])])), np.concatenate((gesture_data_labels, gesture_data_labels[0:int(percent_augment*gesture_data_labels.shape[0])]))
