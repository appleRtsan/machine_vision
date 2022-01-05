import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('dumptruck1_360x270.bmp')
img2 = cv2.imread('dumptruck2_360x270.bmp')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# calculate optical flow
flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                                      
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv = np.zeros_like(img1)
hsv[..., 1] = 255
hsv[..., 0] = ang*180/np.pi/2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

step = 9
plt.quiver(np.arange(0, flow.shape[1], step), np.arange(flow.shape[0], 0, -step), 
           flow[::step, ::step, 0], flow[::step, ::step, 1])
plt.show()
