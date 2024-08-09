from FrameSaver_DCAM710 import record
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
    parser.add_argument("-d", "--duration", type=int, default=2)
    args = parser.parse_args()

    pre_date = datetime.datetime.now().strftime("%Y-%m-%d")
    post_date = datetime.datetime.now().strftime("%H-%M-%S")
    dataset_folder = os.path.join(pre_date, post_date)
    os.makedirs(dataset_folder, exist_ok=True)
    p1 = mp.Process(target=record, args=(dataset_folder, args.duration * 30))
    # p2 = mp.Process(target=bmi160, args=(dataset_folder, args.duration, 1))
    p2 = mp.Process(target=bmi160_accsave, args=(dataset_folder, 400, args.duration, 1))
    p3 = mp.Process(target=receive_audio, args=(dataset_folder, args.duration))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()
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
