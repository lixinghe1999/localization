import math
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self):
        # 状态向量 [位置, 速度]
        self.x = np.zeros((6, 1))  # [roll, roll_rate, pitch, pitch_rate, yaw, yaw_rate]
        
        # 状态转移矩阵
        self.A = np.eye(6)
        
        # 观测矩阵
        self.H = np.eye(6)
        
        # 过程噪声协方差
        self.Q = np.eye(6) * 0.01
        
        # 测量噪声协方差
        self.R = np.eye(6) * 0.1
        
        # 估计误差协方差
        self.P = np.eye(6)
    
    def predict(self, gyro, dt):
        # 更新状态向量
        self.x[::2] += gyro.reshape(-1, 1) * dt
        
        # 状态转移
        self.x = self.A @ self.x
        
        # 更新误差协方差
        self.P = self.A @ self.P @ self.A.T + self.Q
        
    def update(self, acc):
        # 将加速度计数据转换为测量值
        z = np.zeros((6, 1))
        z[0, 0] = self.x[0, 0]  # roll
        z[1, 0] = self.x[1, 0]  # roll rate
        z[2, 0] = self.x[2, 0]  # pitch
        z[3, 0] = self.x[3, 0]  # pitch rate
        z[4, 0] = self.x[4, 0]  # yaw
        z[5, 0] = self.x[5, 0]  # yaw rate
        
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        self.x += K @ (acc.reshape(-1, 1) - z)
        
        # 更新误差协方差
        self.P = (np.eye(6) - K @ self.H) @ self.P
    
    def get_euler_angles(self):
        # 返回当前的欧拉角
        return self.x[0, 0], self.x[2, 0], self.x[4, 0]  # roll, pitch, yaw


def kalman_filter(data, dt=0.01, plot=False):
    # 读取CSV文件
    df = pd.DataFrame(data, columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])

    kf = KalmanFilter()  # 初始化卡尔曼滤波器

    for index, row in df.iterrows():
        gyro_x = row['gyro_x']
        gyro_y = row['gyro_y']
        gyro_z = row['gyro_z']
        acc_x = row['acc_x']
        acc_y = row['acc_y']
        acc_z = row['acc_z']
        
        # 预测和更新卡尔曼滤波器
        kf.predict(np.array([gyro_x, gyro_y, gyro_z]) * np.pi / 180, dt)  # 角速度单位转换为弧度
        kf.update(np.array([acc_x, acc_y, acc_z]))

        # 获取欧拉角
        roll, pitch, yaw = kf.get_euler_angles()

        # 更新数据框
        df.at[index, 'roll'] = math.degrees(roll)
        df.at[index, 'pitch'] = math.degrees(pitch)
        df.at[index, 'yaw'] = math.degrees(yaw)
        # 保存结果到新的CSV文件
        euler = df[['roll', 'pitch', 'yaw']].values
    if plot:
        plt.plot(euler, label=['roll', 'pitch', 'yaw'])
        plt.legend()
        plt.savefig('euler.png')

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
