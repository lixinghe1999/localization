import time
from BMI160_i2c import Driver

map_rate_acc = {
    1600: 12,
    800: 11,
    400: 10,
    200: 9,
    100: 8,
    50: 7,
    25: 6,
}
map_rate_gyro = {
    25: 6,
    50: 7,
    100: 8,
    200: 9,
    400: 10,
    800: 11,
    1600: 12,
    3200: 13,
}
def bmi160_accsave(dataset_folder, sample_rate=1600, t=5, port=0):
    sensor = Driver(0x69, port) #change address if needed
    sensor.set_accel_rate(map_rate_acc[sample_rate])
    num = sample_rate * t
    a = 0; b = 0 # good number, bad number
    time_start = time.time()
    acc = ''
    writer = open(dataset_folder + f'/{str(time_start)}_acc_{str(port)}.txt', 'w')
    while (a < num):
        if sensor.getIntACCDataReadyStatus():
            try:
                data = sensor.getAcceleration()
                a = a + 1
                acc += str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + ' ' + str(time.time()) + '\n'
            except:
                b = b + 1
        if b > 100:
            print('too many bad data') 
            break
    real_sample_rate = num / (time.time() - time_start)
    print('ACC port:', port, 'sample rate:', num / (time.time() - time_start))
    writer.write(acc)      
    return real_sample_rate
def bmi160_gyrosave(dataset_folder, sample_rate=1600, t=5, port=0):
    sensor = Driver(0x69, port)# change address if needed
    sensor.set_gyro_rate(map_rate_gyro[sample_rate])
    a = 0; b = 0
    num = sample_rate * t
    time_start = time.time()
    gyro = ''
    writer = open(dataset_folder + f'/{str(time_start)}_gyro_{str(port)}.txt', 'w')
    while (a < num):
        if sensor.getIntGYRODataReadyStatus():
            try:
                a = a + 1
                data = sensor.getRotation()
                gyro += str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + ' ' + str(time.time()) + '\n'
            except:
                b = b + 1
        if b > 100:
            print('too many bad data, gyro')
            break
    real_sample_rate = num / (time.time() - time_start)
    print('GYRO port:', port, 'sample rate:', num/(time.time() - time_start))
    writer.write(gyro)
    return real_sample_rate

def bmi160(dataset_folder, sample_rate=1600, t=5, port=1):
    sensor = Driver(0x69, port) # change address if needed
    sensor.set_gyro_rate(map_rate_gyro[sample_rate])
    sensor.set_accel_rate(map_rate_acc[sample_rate])
    i = 0
    writer = open(dataset_folder + f'/bmi160_{str(port)}.txt', 'w')
    imu = ''
    t_start = time.time()
    # sensor.setIntDataReadyEnabled(True)
    while i < sample_rate * t:
        if sensor.getIntACCDataReadyStatus() and sensor.getIntGYRODataReadyStatus():
        # if sensor.getIntDataReadyEnabled():
            data = sensor.getMotion6() # gx, gy, gz, ax, ay, az
            imu += str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + ' ' + str(data[3]) + ' ' \
                    + str(data[4]) + ' ' + str(data[5]) + ' ' + str(time.time()) + '\n'
            i += 1
    writer.write(imu)
    real_sample_rate = sample_rate * t / (time.time() - t_start)
    print('IMU port:', port, 'sample rate:', real_sample_rate)
    return real_sample_rate
def vis(file):
    import matplotlib.pyplot as plt
    import numpy as np
    data = np.loadtxt(file)
    fig, axs = plt.subplots(2, 1)
    timestamp = data[:, -1] - data[0, -1]
    axs[0].plot(timestamp, data[:, 0], 'r', label='gx')
    axs[0].plot(timestamp, data[:, 1], 'g', label='gy')
    axs[0].plot(timestamp, data[:, 2], 'b', label='gz')
    plt.legend()

    axs[1].plot(timestamp, data[:, 3], 'r', label='ax')
    axs[1].plot(timestamp, data[:, 4], 'g', label='ay')
    axs[1].plot(timestamp, data[:, 5], 'b', label='az')
    plt.legend()
    plt.savefig('bmi160.png')
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--duration", type=int, default=3)
    parser.add_argument("-f", "--folder", type=str, default='.')
    args = parser.parse_args()
    sr = bmi160(args.folder, 400, args.duration, 1)
    vis('bmi160_1.txt')
    #sr1 = bmi160_accsave('.', 400, 3, port=1)
    # sr2 = bmi160_gyrosave('.', 400, 3, port=1)
