import numpy as np
import os
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
def pyIMU(data, plot=False):
    from pyIMU.madgwick import Madgwick
    from pyIMU.quaternion import Quaternion, Vector3D 
    from pyIMU.motion import Motion

    madgwick = Madgwick(frequency=50.0, gain=0.033)

    estimator = Motion(declination=9.27, latitude=32.253460, altitude=730, magfield=47392.3)
    timestamp = 0
    quaternions = []; positions = []
    for i in range(len(data)):
        gyro_data = data[i, 3:6]
        acc_data = data[i, 0:3]
        timestamp += 0.02
        gyro_data = Vector3D(gyro_data)
        acc_data = Vector3D(acc_data)
        # provide time increment dt based on time expired between each sensor reading
        madgwick.update(gyr=gyro_data, acc=acc_data, dt=0.02)
        # access the quaternion
        quaternions.append(madgwick.q.q)

        estimator.update(q=madgwick.q, acc=acc_data, timestamp=timestamp, moving=True)
        positions.append(estimator.worldPosition.v)
        # or take the return value of the update function
    quaternions = np.array(quaternions)
    positions = np.array(positions)
    # convert quaternion to euler angles
    eulers = quaternion_to_euler_batch(quaternions)
    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        axs[0].plot(positions)
        axs[1].plot(eulers, label=['yaw', 'pitch', 'roll'])
        axs[1].legend()
        plt.savefig('pyIMU.png')
    return eulers


if __name__ == '__main__':
    data_dir = 'dataset/earphone/test'
    data_files = os.listdir(data_dir)
    data_files.sort()

    for data_file in data_files[4:]:
        data = np.loadtxt(os.path.join(data_dir, data_file), delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5))
        break
    # kalman_filter(data, dt=0.02, plot=True)
    pyIMU(data, plot=True)
