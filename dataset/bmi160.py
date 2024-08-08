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
    writer = open(dataset_folder + f'/acc_{str(port)}.txt', 'w')
    while (a < num):
        if sensor.getIntACCDataReadyStatus():
            try:
                data = sensor.getAcceleration()
                a = a + 1
                acc = str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + ' ' + str(time.time()) + '\n'
                writer.write(acc)      
            except:
                b = b + 1
        if b > 100:
            print('too many bad data') 
            break
    real_sample_rate = num / (time.time() - time_start)
    # print('ACC port:', port, 'sample rate:', num / (time.time() - time_start))
    return real_sample_rate
def bmi160_gyrosave(name, sample_rate=1600, t=5, port=0):
    sensor = Driver(0x69, port)# change address if needed
    sensor.set_gyro_rate(12)
    a = 0; b = 0
    num = sample_rate * t
    time_start = time.time()
    writer = open(name + '_' + str(port) + '.txt', 'w')
    while (a < num):
        if sensor.getIntGYRODataReadyStatus():
            try:
                a = a + 1
                data = sensor.getRotation()
                gryo = gryo + str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + ' ' + str(time.time()) + '\n'
                writer.write(gryo)
            except:
                b = b + 1
        if b > 100:
            print('too many bad data')
            break
    real_sample_rate = num / (time.time() - time_start)
    #print('GYRO port:', port, num/(time.time() - time_start))
    return real_sample_rate
if __name__ == "__main__":
    sr1 = bmi160_accsave('', 400, 3, port=0)
    sr2 = bmi160_accsave('', 400, 3, port=1)