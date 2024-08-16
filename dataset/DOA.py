import numpy as np
import cv2
import matplotlib.pyplot as plt


def undistorted_img(image,camera_matrix,dist_coeffs):
    # 校正图像
    height, width = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))
    # 去畸变
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # 裁剪图像
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]

    # 保存结果
    cv2.imwrite('./distorted_image.jpg',image)
    cv2.imwrite('./undistorted_image.jpg', undistorted_image)
    return undistorted_image

def calculate_angles(image, fx, fy, cx, cy):
    height, width = image.shape[:2]
    angles = np.zeros((height, width, 3), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            # 归一化坐标
            xn = (x - cx) / fx
            yn = (y - cy) / fy
            zn = 1.0
            
            # 计算角度 (弧度)
            theta_x = np.degrees(np.arctan2(xn, zn))
            theta_y = np.degrees(np.arctan2(yn, zn))
            angle = np.degrees(np.arccos(zn / np.sqrt(xn**2 + yn**2 + zn**2)))

            angles[y, x, 0] = theta_x
            angles[y, x, 1] = theta_y
            angles[y, x ,2] = angle

    return angles
#camera_matrix = np.array([[517.306408,0.000000,318.643040],
                #  [0.000000,516.469215,255.313989],
                #  [0.000000,0.000000,1.000000]])
camera_matrix = np.array([[470.10501443925796,0.000000,316.2297282272395],
                 [0.000000,469.85695908479204,233.96838482821087],
                 [0.000000,0.000000,1.000000]])
#dist_coeffs = np.array([0.262383,-0.953104,-0.005358,0.002628,1.163314])
dist_coeffs = np.array([-0.02928646756844723,0.10886972361640557,-0.00037308099853059434,0.0016085357675248854, -0.09406327705239186])
#example_file = './data/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png'
example_file = './calibrate/ir1.png'
image = cv2.imread(example_file)
undistorted_image = undistorted_img(image,camera_matrix,dist_coeffs)

fx = camera_matrix[0,0]
fy = camera_matrix[1,1]
cx = camera_matrix[0,2]
cy = camera_matrix[1,2]
angle_map = calculate_angles(image, fx, fy, cx, cy)
# 可视化
# plt.imshow(angle_map[:,:,0], cmap='jet')
# plt.colorbar(label='Angle with X-axis')
# plt.title('Pixel Angles with X-axis')
# plt.savefig('./test_x.png')
# plt.close()
# plt.imshow(angle_map[:,:,1], cmap='jet')
# plt.colorbar(label='Angle with Y-axis')
# plt.title('Pixel Angles with Y-axis')
# plt.savefig('./test_y.png')
# plt.close()
# plt.imshow(angle_map[:,:,2], cmap='jet')
# plt.colorbar(label='Angle with Z-axis')
# plt.title('Pixel Angles with Z-axis')
# plt.savefig('./test_z.png')
# plt.close()
