from dataclasses import dataclass
from scipy.interpolate import interp1d
import imufusion
import matplotlib.pyplot as pyplot
import numpy as np
import datetime
import librosa


def imu_loading_example(imu_file):
    sample_rate = 400  # 400 Hz
    data = np.genfromtxt(imu_file, delimiter=",", skip_header=1)
    timestamp = data[:, 0]
    gyroscope = data[:, 1:4]
    accelerometer = data[:, 4:7]
    # Process sensor data
    delta_time = np.diff(timestamp, prepend=timestamp[0])
    return timestamp, gyroscope, accelerometer, delta_time
def imu_loading(imu_file):
    sample_rate = 50
    if imu_file.endswith('.npy'):
        imu_data = np.load(imu_file)
        gyroscope = imu_data[:, 0:3] / np.pi * 180
        accelerometer = imu_data[:, 3:6] / 9.81
        timestamp = np.linspace(0, len(accelerometer)/sample_rate, len(accelerometer))
        delta_time = np.diff(timestamp, prepend=timestamp[0])
        return timestamp, gyroscope, accelerometer, delta_time
    else:
        imu_data = np.loadtxt(imu_file, delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6,))
        imu_timestamp = np.loadtxt(imu_file, delimiter=',', skiprows=1, usecols=(7,), dtype=str)
        imu_timestamp = [datetime.datetime.strptime(time, '%Y%m%d_%H%M%S_%f') for time in imu_timestamp]
        # set the start time to 0 and convert to seconds
        imu_timestamp = np.array([(time - imu_timestamp[0]).total_seconds() for time in imu_timestamp])
        imu_sr = len(imu_timestamp)/imu_timestamp[-1]
        imu_data = librosa.resample(imu_data[:, :6].T, orig_sr=imu_sr, target_sr=50).T
        accelerometer = imu_data[:, 0:3] / 9.81
        gyroscope = imu_data[:, 3:6] / np.pi * 180
        timestamp = np.linspace(0, len(accelerometer)/sample_rate, len(accelerometer))
        delta_time = np.diff(timestamp, prepend=timestamp[0])
    return timestamp, gyroscope, accelerometer, delta_time

def imu_fusion(timestamp, gyroscope, accelerometer, delta_time, sample_rate=50):

    # Instantiate AHRS algorithms
    offset = imufusion.Offset(sample_rate)
    ahrs = imufusion.Ahrs()

    ahrs.settings = imufusion.Settings(imufusion.CONVENTION_NWU,
                                    0.5,  # gain
                                    2000,  # gyroscope range
                                    10,  # acceleration rejection
                                    0,  # magnetic rejection
                                    5 * sample_rate)  # rejection timeout = 5 seconds

    # Process sensor data
    delta_time = np.diff(timestamp, prepend=timestamp[0])

    euler = np.empty((len(timestamp), 3))
    internal_states = np.empty((len(timestamp), 3))
    acceleration = np.empty((len(timestamp), 3))

    for index in range(len(timestamp)):
        gyroscope[index] = offset.update(gyroscope[index])

        ahrs.update_no_magnetometer(gyroscope[index], accelerometer[index], delta_time[index])

        euler[index] = ahrs.quaternion.to_euler()

        ahrs_internal_states = ahrs.internal_states
        internal_states[index] = np.array([ahrs_internal_states.acceleration_error,
                                            ahrs_internal_states.accelerometer_ignored,
                                            ahrs_internal_states.acceleration_recovery_trigger])

        acceleration[index] = 9.81 * ahrs.earth_acceleration  # convert g to m/s/s

    return timestamp, gyroscope, accelerometer, euler, internal_states, acceleration

# Plot sensor data
def plot_sensor(timestamp, gyroscope, accelerometer, euler, internal_states):
    figure, axes = pyplot.subplots(nrows=3, sharex=True)

    figure.suptitle("Sensors data, Euler angles, and AHRS internal states")

    axes[0].plot(timestamp, gyroscope[:, 0], "tab:red", label="Gyroscope X")
    axes[0].plot(timestamp, gyroscope[:, 1], "tab:green", label="Gyroscope Y")
    axes[0].plot(timestamp, gyroscope[:, 2], "tab:blue", label="Gyroscope Z")
    axes[0].set_ylabel("Degrees/s")
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(timestamp, accelerometer[:, 0], "tab:red", label="Accelerometer X")
    axes[1].plot(timestamp, accelerometer[:, 1], "tab:green", label="Accelerometer Y")
    axes[1].plot(timestamp, accelerometer[:, 2], "tab:blue", label="Accelerometer Z")
    axes[1].set_ylabel("g")
    axes[1].grid()
    axes[1].legend()


    # Plot Euler angles
    axes[2].plot(timestamp, euler[:, 0], "tab:red", label="Roll")
    axes[2].plot(timestamp, euler[:, 1], "tab:green", label="Pitch")
    axes[2].plot(timestamp, euler[:, 2], "tab:blue", label="Yaw")
    axes[2].set_ylabel("Degrees")
    axes[2].grid()
    axes[2].legend()

    # # Plot internal states
    # axes[3].plot(timestamp, internal_states[:, 0], "tab:olive", label="Acceleration error")
    # axes[3].set_ylabel("Degrees")
    # axes[3].grid()
    # axes[3].legend()

    # axes[4].plot(timestamp, internal_states[:, 1], "tab:cyan", label="Accelerometer ignored")
    # pyplot.sca(axes[4])
    # pyplot.yticks([0, 1], ["False", "True"])
    # axes[4].grid()
    # axes[4].legend()

    # axes[5].plot(timestamp, internal_states[:, 2], "tab:orange", label="Acceleration recovery trigger")
    # axes[5].set_xlabel("Seconds")
    # axes[5].grid()
    # axes[5].legend()
    pyplot.savefig("sensors_data.png")

def imu_tracking(timestamp, acceleration, sample_rate=50):
    delta_time = np.diff(timestamp, prepend=timestamp[0])
    # Identify moving periods
    is_moving = np.empty(len(timestamp))

    for index in range(len(timestamp)):
        is_moving[index] = np.sqrt(acceleration[index].dot(acceleration[index])) > 3  # threshold = 3 m/s/s

    margin = int(0.1 * sample_rate)  # 100 ms

    for index in range(len(timestamp) - margin):
        is_moving[index] = any(is_moving[index:(index + margin)])  # add leading margin

    for index in range(len(timestamp) - 1, margin, -1):
        is_moving[index] = any(is_moving[(index - margin):index])  # add trailing margin

    # Calculate velocity (includes integral drift)
    velocity = np.zeros((len(timestamp), 3))

    for index in range(len(timestamp)):
        if is_moving[index]:  # only integrate if moving
            velocity[index] = velocity[index - 1] + delta_time[index] * acceleration[index]

    # Find start and stop indices of each moving period
    is_moving_diff = np.diff(is_moving, append=is_moving[-1])
    
    
    @dataclass
    class IsMovingPeriod:
        start_index: int = -1
        stop_index: int = -1


    is_moving_periods = []
    is_moving_period = IsMovingPeriod()

    for index in range(len(timestamp)):
        if is_moving_period.start_index == -1:
            if is_moving_diff[index] == 1:
                is_moving_period.start_index = index

        elif is_moving_period.stop_index == -1:
            if is_moving_diff[index] == -1:
                is_moving_period.stop_index = index
                is_moving_periods.append(is_moving_period)
                is_moving_period = IsMovingPeriod()

    # Remove integral drift from velocity
    velocity_drift = np.zeros((len(timestamp), 3))

    for is_moving_period in is_moving_periods:
        start_index = is_moving_period.start_index
        stop_index = is_moving_period.stop_index

        t = [timestamp[start_index], timestamp[stop_index]]
        x = [velocity[start_index, 0], velocity[stop_index, 0]]
        y = [velocity[start_index, 1], velocity[stop_index, 1]]
        z = [velocity[start_index, 2], velocity[stop_index, 2]]

        t_new = timestamp[start_index:(stop_index + 1)]

        velocity_drift[start_index:(stop_index + 1), 0] = interp1d(t, x)(t_new)
        velocity_drift[start_index:(stop_index + 1), 1] = interp1d(t, y)(t_new)
        velocity_drift[start_index:(stop_index + 1), 2] = interp1d(t, z)(t_new)

    velocity = velocity - velocity_drift

        
    # Calculate position
    position = np.zeros((len(timestamp), 3))

    for index in range(len(timestamp)):
        position[index] = position[index - 1] + delta_time[index] * velocity[index]
    return is_moving, velocity, position

def plot_tracking(timestamp, acceleration, is_moving, velocity, position):
      # Plot acceleration
    _, axes = pyplot.subplots(nrows=4, sharex=True, gridspec_kw={"height_ratios": [6, 1, 6, 6]})
    axes[0].plot(timestamp, acceleration[:, 0], "tab:red", label="X")
    axes[0].plot(timestamp, acceleration[:, 1], "tab:green", label="Y")
    axes[0].plot(timestamp, acceleration[:, 2], "tab:blue", label="Z")
    axes[0].set_title("Acceleration")
    axes[0].set_ylabel("m/s/s")
    axes[0].grid()
    axes[0].legend()


    # Plot moving periods
    axes[1].plot(timestamp, is_moving, "tab:cyan", label="Is moving")
    pyplot.sca(axes[1])
    pyplot.yticks([0, 1], ["False", "True"])
    axes[1].grid()
    axes[1].legend()


    # Plot velocity
    axes[2].plot(timestamp, velocity[:, 0], "tab:red", label="X")
    axes[2].plot(timestamp, velocity[:, 1], "tab:green", label="Y")
    axes[2].plot(timestamp, velocity[:, 2], "tab:blue", label="Z")
    axes[2].set_title("Velocity")
    axes[2].set_ylabel("m/s")
    axes[2].grid()
    axes[2].legend()

    # Plot position
    axes[3].plot(timestamp, position[:, 0], "tab:red", label="X")
    axes[3].plot(timestamp, position[:, 1], "tab:green", label="Y")
    axes[3].plot(timestamp, position[:, 2], "tab:blue", label="Z")
    axes[3].set_title("Position")
    axes[3].set_xlabel("Seconds")
    axes[3].set_ylabel("m")
    axes[3].grid()
    axes[3].legend()

    # Print error as distance between start and final positions
    print("Error: " + "{:.3f}".format(np.sqrt(position[-1].dot(position[-1]))) + " m")
    pyplot.savefig("moving.png")
