import matplotlib.pyplot as plt
import numpy as np

'''
2.2 Visualize some samples.
'''

def plot_sensor_data(csv_path , fig_file_name):

    #csv must have 7 columns: first is time, last 6 are acc(x,y,z) and angular speed(x,y,z)

    sensor_data = np.genfromtxt(csv_path, delimiter=',')
    NUM_CHANNELS = 6
    TIME_COLUMN = 0
    labels = ["$a_x$", "$a_y$", "$a_z$", "pitch", "roll", "yaw"]

    plt.figure()
    plt.title("Visualizing " + csv_path)
    for i in range(NUM_CHANNELS):
        plt.plot(sensor_data[:,TIME_COLUMN], sensor_data[:, i+1], label=labels[i])

    plt.xlabel("Time (ms)")
    plt.ylabel("Acceleration $(m/s^2)$ or Angular Speed (rad/s)")
    plt.legend(loc='best')
    plt.savefig(fig_file_name)

    return

def main():
    plot_sensor_data('unnamed_train_data/student0/a_1.csv', "plot_s0_a1.png")
    plot_sensor_data('unnamed_train_data/student1/a_2.csv', "plot_s1_a2.png")
    plot_sensor_data('unnamed_train_data/student2/a_3.csv', "plot_s2_a3.png")
    plot_sensor_data('unnamed_train_data/student3/z_1.csv', "plot_s3_z1.png")
    plot_sensor_data('unnamed_train_data/student4/z_2.csv', "plot_s4_z2.png")
    plot_sensor_data('unnamed_train_data/student5/z_3.csv', "plot_s5_z3.png")

if __name__ == "__main__":
    main()