from pickle import FALSE, TRUE
import sys
sys.path.append('./BaseSDK_python_wrapper')

from BaseSDK_python_wrapper.DCAM710.API.Vzense_api_710 import *
import cv2
import time
import copy
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
    #convert ushort value to 0xff is just for display
    img = numpy.int32(frame)
    img = img*255/value_max
    img = numpy.clip(img, 0, 255)
    img = numpy.uint8(img)
    # img = cv2.equalizeHist(img)
    # img = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)
    writer.write(img)
    return img

def record_ir(name):
    camera, ret = scan_camera()
    if ret != 0:
        print('Ps2_OpenDevice failed: ' + str(ret))
        return
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
    time_start = time.time()
    ret, frameready = camera.Ps2_ReadNextFrame()
    print(frameready.ir)
    while True:
        if frameready.ir:
            ret,irframe = camera.Ps2_GetFrame(PsFrameType.PsIRFrame)

            frametmp = numpy.ctypeslib.as_array(irframe.pFrameData, (1, irframe.width * irframe.height * 2))
            frametmp.dtype = numpy.uint16
            frametmp.shape = (irframe.height, irframe.width)

            img = numpy.int32(frametmp)
            img = img*255/value_max
            img = numpy.clip(img, 0, 255)
            img = numpy.uint8(img)
            cv2.imwrite(name, img)
            break
    print("finished writing, takes: ", time.time()-time_start)


def record_video(name, fps=30, duration = 1):
    camera, ret = scan_camera()
    if ret != 0:
        print('Ps2_OpenDevice failed: ' + str(ret))
        return
    ret = camera.Ps2_StartStream()
    if  ret == 0:
        print("start stream successful")
    else:
        print("Ps2_StartStream failed:",ret)
    ret = camera.Ps2_SetDataMode(PsDataMode.PsDepthAndIR_30)
    if  ret != 0:  
        print("Ps2_SetDataMode failed:",ret)

    ret = camera.Ps2_SetDepthRange(PsDepthRange.PsMidRange)
    if  ret != 0:  
        print("Ps2_SetDepthRange failed:",ret)

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
    time_start = time.time()
    # rgb_writer = cv2.VideoWriter(os.path.join(name, 'rgb.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 360))
    depth_writer = cv2.VideoWriter(os.path.join(name, str(time_start) + '_depth.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480), isColor=False)
    ir_writer = cv2.VideoWriter(os.path.join(name, str(time_start) + '_ir.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480),  isColor=False)
    depthframes = []; irframes = []
    max_frames = duration * fps
    try:
        for frames in range(max_frames):
            ret, frameready = camera.Ps2_ReadNextFrame()
            if  ret !=0:
                print("Ps2_ReadNextFrame failed:",ret)
                time.sleep(1)
                continue
            if  frameready.depth:    
                ret,depthframe = camera.Ps2_GetFrame(PsFrameType.PsDepthFrame)

                frametmp = numpy.ctypeslib.as_array(depthframe.pFrameData, (1, depthframe.width * depthframe.height * 2))
                frametmp.dtype = numpy.uint16
                frametmp.shape = (depthframe.height, depthframe.width)
                frametmp = copy.deepcopy(frametmp) 
                depthframes.append(frametmp)
            if frameready.ir:
                ret,irframe = camera.Ps2_GetFrame(PsFrameType.PsIRFrame)

                frametmp = numpy.ctypeslib.as_array(irframe.pFrameData, (1, irframe.width * irframe.height * 2))
                frametmp.dtype = numpy.uint16
                frametmp.shape = (irframe.height, irframe.width)
                frametmp = copy.deepcopy(frametmp) 
                irframes.append(frametmp)

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
        print("start writing video")
        for (depthframe, irframe) in zip(depthframes, irframes):
            write_frame(depthframe, depth_writer, value_max)
            write_frame(irframe, ir_writer, 3840)
        print("finished writing, takes: ", time.time()-time_start)
    
            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--duration", type=int, default=5)
    parser.add_argument("-f", "--folder", type=str, default='.')
    parser.add_argument("-n", "--name", type=int, default=1)
    args = parser.parse_args()
    #record_video('', 50)   
    record_ir(f'ir{args.name}.png')