from pickle import FALSE, TRUE
import sys
sys.path.append('./BaseSDK_python_wrapper')

from BaseSDK_python_wrapper.DCAM710.API.Vzense_api_710 import *
import cv2
import time
import multiprocessing as mp
from tqdm import tqdm

def scan_camera():
    camera = VzenseTofCam()
    camera_count = camera.Ps2_GetDeviceCount()
    retry_count = 100
    while camera_count==0 and retry_count > 0:
        retry_count = retry_count-1
        camera_count = camera.Ps2_GetDeviceCount()
        time.sleep(1)
        print("scaning......   ",retry_count)
    if camera_count > 1:
        ret,device_infolist=camera.Ps2_GetDeviceListInfo(camera_count)
        if ret==0:
            device_info = device_infolist[0]
            for info in device_infolist: 
                print('cam uri:  ' + str(info.uri))
        else:
            print(' failed:' + ret)  
            exit()  
    elif camera_count == 1:
        ret,device_info=camera.Ps2_GetDeviceInfo()
        if ret==0:
            print('cam uri:' + str(device_info.uri))
        else:
            print(' failed:' + ret)   
            exit() 
    else: 
        print("there are no camera found")
        exit()
    print("uri: "+str(device_info.uri))
    ret = camera.Ps2_OpenDevice(device_info.uri)
    return camera, ret

def write_frame(frame, writer, value_max):
    frametmp = numpy.ctypeslib.as_array(frame.pFrameData, (1, frame.width * frame.height * 2))
    frametmp.dtype = numpy.uint16
    frametmp.shape = (frame.height, frame.width)

    #convert ushort value to 0xff is just for display
    img = numpy.int32(frametmp)
    img = img*255/value_max
    img = numpy.clip(img, 0, 255)
    img = numpy.uint8(img)
    # img = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)
    writer.write(img)
    
def record(name, camera, max_frames = 0):
    ret = camera.Ps2_StartStream()
    if  ret == 0:
        print("start stream successful")
    else:
        print("Ps2_StartStream failed:",ret)
    ret = camera.Ps2_SetDataMode(PsDataMode.PsDepthAndIR_30)
    if  ret != 0:  
        print("Ps2_SetDataMode failed:",ret)
    ret, depthrange = camera.Ps2_GetDepthRange()
    if  ret == 0:
        print("Ps2_GetDepthRange :",depthrange.value)
    else:
        print("Ps2_GetDepthRange failed:",ret)

    ret, depth_max, value_min, value_max = camera.Ps2_GetMeasuringRange(PsDepthRange(depthrange.value))
    if  ret == 0:
        print("Ps2_GetMeasuringRange: ",depth_max,",",value_min,",",value_max)
    else:
        print("Ps2_GetMeasuringRange failed:",ret)

    # rgb_writer = cv2.VideoWriter(os.path.join(name, 'rgb.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 360))
    depth_writer = cv2.VideoWriter(os.path.join(name, 'depth.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480), isColor=False)
    ir_writer = cv2.VideoWriter(os.path.join(name, 'ir.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480),  isColor=False)
    time_start = time.time()
    depthframes = []; irframes = []
    try:
        for frames in tqdm(range(max_frames)):
            ret, frameready = camera.Ps2_ReadNextFrame()
            if  ret !=0:
                print("Ps2_ReadNextFrame failed:",ret)
                time.sleep(1)
                continue
            ret,depthframe = camera.Ps2_GetFrame(PsFrameType.PsDepthFrame)
            depthframes.append(depthframe)
            ret,irframe = camera.Ps2_GetFrame(PsFrameType.PsIRFrame)
            irframes.append(irframe)
            # if  frameready.rgb:      
            #     ret,rgbframe = camera.Ps2_GetFrame(PsFrameType.PsRGBFrame)
            #     if  ret == 0:
            #         frametmp = numpy.ctypeslib.as_array(rgbframe.pFrameData, (1, rgbframe.width * rgbframe.height * 3))
            #         frametmp.dtype = numpy.uint8
            #         frametmp.shape = (rgbframe.height, rgbframe.width,3)
            #         rgb_writer.write(frametmp)
            #     else:
            #         print("---end---")            
    except Exception as e :
        print(e)
    finally:
        print("frames: ", frames, "frames per second: ", frames/(time.time()-time_start))
        time_start = time.time()
        print("start writing")
        for (depthframe, irframe) in tqdm(zip(depthframes, irframes), total=len(depthframes)):
            write_frame(depthframe, depth_writer, value_max)
            write_frame(irframe, ir_writer, 3840)
        print("finished writing, takes: ", time.time()-time_start)
    
            

if __name__ == "__main__":
    camera, ret = scan_camera()
    if ret == 0:
        record('', camera, 100)
    else:
        print('Ps2_OpenDevice failed: ' + str(ret))  