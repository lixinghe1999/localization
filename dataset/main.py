from FrameSaver_DCAM710 import scan_camera, record
from bmi160 import bmi160_accsave, bmi160_gyrosave
from mic import receive_audio

if __name__ == "__main__":
    import datetime
    import os
    pre_date = datetime.datetime.now().strftime("%Y-%m-%d")
    post_date = datetime.datetime.now().strftime("%H-%M-%S")
    dataset_folder = os.path.join(pre_date, post_date)
    os.makedirs(dataset_folder, exist_ok=True)

    camera, ret = scan_camera()
    if ret == 0:
        record(dataset_folder, camera, 100)
    else:
        print("No camera found")
