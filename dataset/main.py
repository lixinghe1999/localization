from FrameSaver_DCAM710 import record_video
from bmi160 import bmi160, bmi160_accsave, bmi160_gyrosave
from mic import receive_audio
import datetime
import os
import argparse
import multiprocessing as mp
import sys
sys.path.append('./BaseSDK_python_wrapper')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imu", type=int, default=0) # 0: no imu, 1: left, 2: right, 3: both
    parser.add_argument("--camera", type=int, default=0) # 0: no camera, 1: yes
    parser.add_argument("--audio", type=int, default=0) # 0: no audio, 1: Binaural
    parser.add_argument("--time", type=str, default=None)
    parser.add_argument("-d", "--duration", type=int, default=5)
    args = parser.parse_args()

    if args.time is not None:
        pre_date = args.time.split('_')[0]
        post_date = args.time.split('_')[1]
    else:
        pre_date = datetime.datetime.now().strftime("%Y-%m-%d")
        post_date = datetime.datetime.now().strftime("%H-%M-%S")
    dataset_folder = os.path.join(pre_date, post_date)
    os.makedirs(dataset_folder, exist_ok=True)

    process_list = []
    if args.imu != 0:
        if args.imu == 1 or args.imu == 3:
            p1 = mp.Process(target=bmi160, args=(dataset_folder, 200, args.duration, 0))
            process_list.append(p1)
        if args.imu == 2 or args.imu == 3:
            p2 = mp.Process(target=bmi160, args=(dataset_folder, 200, args.duration, 1))
            process_list.append(p2)
    if args.camera == 1:
        p3 = mp.Process(target=record_video, args=(dataset_folder, 30, args.duration))
        process_list.append(p3)
    if args.audio == 1:
        p4 = mp.Process(target=receive_audio, args=(dataset_folder, 48000, args.duration, 2))
        process_list.append(p4)
    else:
        p5 = mp.Process(target=receive_audio, args=(dataset_folder, 48000, args.duration, 8))
        process_list.append(p5)
    

    for p in process_list:
        p.start()
    for p in process_list:
        p.join()    
    # os.system('python FrameSaver_DCAM710.py --folder ' + dataset_folder)
    # use subprocess to run time in parrellel
    # subprocess.run(['python', 'FrameSaver_DCAM710.py', '--folder', dataset_folder])
    # subprocess.run(['python', 'bmi160.py', '--folder', dataset_folder, '--duration', str(args.duration)])

    # t1 = threading.Thread(target=record, args=(dataset_folder, camera, 100))
    # t1.start()
    # t1.join()
    # if ret == 0:
    #     print("Camera found")
    #     p1 = mp.Process(target=record, args=(dataset_folder, camera, args.duration * 30))
    #     p2 = mp.Process(target=bmi160, args=(dataset_folder, args.duration, 1))
    #     p3 = mp.Process(target=receive_audio, args=(dataset_folder, args.duration))

    #     p1.start()
    #     p2.start()
    #     p3.start()

    #     p1.join()
    #     p2.join()
    #     p3.join()
    # else:
    #     print("No camera found")
