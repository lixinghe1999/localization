import numpy as np
import datetime
import librosa
import matplotlib.pyplot as plt

def quaternion_to_euler_batch(quaternions):
    """
    Convert an array of quaternions into euler angles (yaw, pitch, roll).
    
    quaternions: numpy array of shape [N, 4] where each row is [w, x, y, z]
    return: numpy array of shape [N, 3] with euler angles (yaw, pitch, roll) in radians
    """
    # Ensure input is a numpy array
    quaternions = np.asarray(quaternions)
    
    # Extract components
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]
    
    # Yaw (ψ)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    # Pitch (θ)
    pitch = np.arcsin(np.clip(2 * (w * y - x * z), -1.0, 1.0))
    
    # Roll (φ)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

    # Stack the results into an [N, 3] array
    euler = np.column_stack((yaw, pitch, roll))
    euler = np.degrees(euler)
    return euler
def pyIMU(data, frequency=50):
    from pyIMU.madgwick import Madgwick
    from pyIMU.quaternion import Quaternion, Vector3D 
    from pyIMU.motion import Motion

    madgwick = Madgwick(frequency=frequency, gain=0.033)
    estimator = Motion(declination=9.27, latitude=32.253460, altitude=730, magfield=47392.3)
    timestamp = 0
    quaternions = []; positions = []
    for i in range(len(data)):
        gyro_data = data[i, 3:6]
        acc_data = data[i, 0:3]
        timestamp += 1/frequency
        gyro_data = Vector3D(gyro_data)
        acc_data = Vector3D(acc_data)
        # provide time increment dt based on time expired between each sensor reading
        madgwick.update(gyr=gyro_data, acc=acc_data, dt=1/frequency)
        # access the quaternion
        quaternions.append(madgwick.q.q)

        estimator.update(q=madgwick.q, acc=acc_data, timestamp=timestamp, moving=True)
        positions.append(estimator.worldPosition.v)
        # or take the return value of the update function
    quaternions = np.array(quaternions)
    positions = np.array(positions)
    # convert quaternion to euler angles
    eulers = quaternion_to_euler_batch(quaternions)
    return eulers, positions

def imu_loading(imu_file, sample_rate=50):
    '''
    Output
    accelerometer: [T, 3], M/s^2
    gyroscope: [T, 3], rad/s
    '''
    if imu_file.endswith('.npy'):
        imu_data = np.load(imu_file)
        gyroscope = imu_data[:, 0:3] 
        accelerometer = imu_data[:, 3:6]
    else:
        imu_data = np.loadtxt(imu_file, delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6,))
        imu_timestamp = np.loadtxt(imu_file, delimiter=',', skiprows=1, usecols=(7,), dtype=str)
        imu_timestamp = [datetime.datetime.strptime(time, '%Y%m%d_%H%M%S_%f') for time in imu_timestamp]
        # set the start time to 0 and convert to seconds
        imu_timestamp = np.array([(time - imu_timestamp[0]).total_seconds() for time in imu_timestamp])
        imu_sr = len(imu_timestamp)/imu_timestamp[-1]
        imu_data = librosa.resample(imu_data[:, :6].T, orig_sr=imu_sr, target_sr=sample_rate).T
        # accelerometer = imu_data[:, 0:3] / 9.81
        # gyroscope = imu_data[:, 3:6] / np.pi * 180
        accelerometer = imu_data[:, 0:3] 
        gyroscope = imu_data[:, 3:6] 
    imu = np.hstack((accelerometer, gyroscope))
    return imu
if __name__ == '__main__':
    imu_file = 'test_example.csv'
    imu = imu_loading(imu_file, sample_rate=50)
    eulers, positions = pyIMU(imu, frequency=50)

    plt.figure()
    plt.plot(eulers[:, 0], label='yaw')
    plt.plot(eulers[:, 1], label='pitch')
    plt.plot(eulers[:, 2], label='roll')
    plt.legend()
    plt.title('Euler Angles')
    plt.savefig('euler_angles.png')
