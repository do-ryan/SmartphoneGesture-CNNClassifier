import numpy as np
import string
import matplotlib.pyplot as plt

'''
    Visualize some basic statistics of our dataset.

2.3
'''

def compute_avg_sensorvals(instances_npy_file, labels_npy_file):
    # instances takes in a .npy file of size (number of files/ instances)x(number of samples across time)x(number of channels+time)
    # returns 26x6 np array with average value for each of 6 channels for all 26 letters of alphabet

    NUM_CHANNELS = 6
    instances = np.load(instances_npy_file) #5590x100x7
    labels = np.load(labels_npy_file) # 1D array: 5590 of "a" -> "z"
    avg_sensor_vals = np.zeros((len(string.ascii_lowercase), NUM_CHANNELS)) #26x6 array containing avg value/ letter/ channel

    for i in range(len(string.ascii_lowercase)):
        temp_list = []
        for j in range(instances.shape[0]):
            if labels[j] == string.ascii_lowercase[i]:
                list.append(temp_list, instances[j])
        # at this point temp_list is 215x100x7
        for k in range(instances.shape[2]-1): # number of channels = num columns minus time col
            avg_sensor_vals[i,k] = np.average(np.asarray(temp_list)[:,:,k+1]) # for current letter, average all samples for each channel

    return avg_sensor_vals #26x6

def compute_std_sensorvals(instances_npy_file, labels_npy_file):
    # instances takes in a .npy file of size (number of files/ instances)x(number of samples across time)x(number of channels+time)
    # returns 26x6 np array with standard deviation for each of 6 channels for all 26 letters of alphabet

    NUM_CHANNELS = 6
    instances = np.load(instances_npy_file)  # 5590x100x7
    labels = np.load(labels_npy_file)  # 1D array: 5590 of "a" -> "z"
    std_sensor_vals = np.zeros((len(string.ascii_lowercase), NUM_CHANNELS))  # 26x6 array containing std value/ letter/ channel

    for i in range(len(string.ascii_lowercase)):
        temp_list = []
        for j in range(instances.shape[0]):
            if labels[j] == string.ascii_lowercase[i]:
                list.append(temp_list, instances[j])
        # at this point temp_list is 215x100x7
        for k in range(instances.shape[2] - 1):  # number of channels = num columns minus time col
            std_sensor_vals[i, k] = np.std(np.asarray(temp_list)[:, :, k+1])  # for current letter, take std of all samples for each channel

    return std_sensor_vals #26x6

def bar_plot_sensorval_stats(gesture, avg_sensor_vals, std_sensor_vals, fig_file_name):
    labels = ["avg $a_x$", "avg $a_y$", "avg $a_z$", "avg pitch", "avg roll", "avg yaw"]
    plt.figure()
    plt.title("Average and std of each sensory channel for gesture " + gesture)
    plt.bar(x = labels, height = avg_sensor_vals[string.ascii_lowercase.index(gesture)], tick_label=labels)
    plt.errorbar(x = labels, y = avg_sensor_vals[string.ascii_lowercase.index(gesture)], yerr=std_sensor_vals[string.ascii_lowercase.index(gesture)], ecolor = "red", ls='none')
    plt.ylabel("Acceleration $(m/s^2)$ or Angular Speed (rad/s)")
    plt.legend(loc='best')

    plt.savefig(fig_file_name)

def main():
    avg_sensor_vals = compute_avg_sensorvals("instances.npy", "labels.npy") #26x6
    std_sensor_vals = compute_std_sensorvals("instances.npy", "labels.npy") #26x6

    bar_plot_sensorval_stats(gesture="a", avg_sensor_vals=avg_sensor_vals, std_sensor_vals=std_sensor_vals, fig_file_name="gesture_a_stats.png")
    bar_plot_sensorval_stats(gesture="f", avg_sensor_vals=avg_sensor_vals, std_sensor_vals=std_sensor_vals, fig_file_name="gesture_f_stats.png")
    bar_plot_sensorval_stats(gesture="z", avg_sensor_vals=avg_sensor_vals, std_sensor_vals=std_sensor_vals, fig_file_name="gesture_z_stats.png")



if __name__ == "__main__":
    main()


