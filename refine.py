import cv2
import matplotlib.pyplot as plt
import numpy as np

# img1 = cv2.imread('dumptruck1_360x270.bmp')
# img2 = cv2.imread('dumptruck2_360x270.bmp')
img1 = cv2.imread('basketball1_360x270.bmp')
img2 = cv2.imread('basketball2_360x270.bmp')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#################################################################
#                                                               #                
#                      histogram specification                  #
#                                                               #
#################################################################


hist,bins = np.histogram(img1.flatten(),256,[0,256])
# plt.hist(img1.flatten(),256,[0,256], color = 'r')
# plt.show()

equ1 = cv2.equalizeHist(gray1)
equ2 = cv2.equalizeHist(gray2)

res1 = np.hstack((gray1,equ1))
res2 = np.hstack((gray2,equ2))

#################################################################
#                                                               #                
#                       Bit-plane Slicing                       #
#                                                               #
#################################################################
lst=[]
for i in range(equ2.shape[0]):
    for j in range(equ2.shape[1]):
         lst.append(np.binary_repr(gray2[i][j] ,width=8)) # width = no. of bits

eight_bit_img2 = (np.array([int(i[0]) for i in lst],dtype = np.uint8) * 128).reshape(gray1.shape[0],gray1.shape[1])
seven_bit_img2 = (np.array([int(i[1])+255 for i in lst],dtype = np.uint8) * 64).reshape(gray1.shape[0],gray1.shape[1])
six_bit_img = (np.array([int(i[2])+128 for i in lst],dtype = np.uint8) * 32).reshape(gray1.shape[0],gray1.shape[1])
five_bit_img = (np.array([int(i[3])+128 for i in lst],dtype = np.uint8) * 16).reshape(gray1.shape[0],gray1.shape[1])
four_bit_img = (np.array([int(i[4])+128 for i in lst],dtype = np.uint8) * 8).reshape(gray1.shape[0],gray1.shape[1])
three_bit_img = (np.array([int(i[5])+128 for i in lst],dtype = np.uint8) * 4).reshape(gray1.shape[0],gray1.shape[1])
two_bit_img = (np.array([int(i[6])+128 for i in lst],dtype = np.uint8) * 2).reshape(gray1.shape[0],gray1.shape[1])
one_bit_img = (np.array([int(i[7])+128 for i in lst],dtype = np.uint8) * 1).reshape(gray1.shape[0],gray1.shape[1])

finalr = cv2.hconcat([eight_bit_img2,seven_bit_img2,six_bit_img,five_bit_img])
finalv =cv2.hconcat([four_bit_img,three_bit_img,two_bit_img,one_bit_img])

# Vertically concatenate
final = cv2.vconcat([finalr,finalv])
# cv2.imshow('o',final)
# cv2.imwrite('ori.bmp',final)

lst=[]
for i in range(equ1.shape[0]):
    for j in range(equ1.shape[1]):
         lst.append(np.binary_repr(equ2[i][j] ,width=8)) # width = no. of bits

eight_bit_img1 = (np.array([int(i[0]) for i in lst],dtype = np.uint8) * 128).reshape(equ1.shape[0],equ1.shape[1])
seven_bit_img1 = (np.array([int(i[1])+255 for i in lst],dtype = np.uint8) * 64).reshape(equ1.shape[0],equ1.shape[1])
six_bit_img = (np.array([int(i[2])+128 for i in lst],dtype = np.uint8) * 32).reshape(equ1.shape[0],equ1.shape[1])
five_bit_img = (np.array([int(i[3])+128 for i in lst],dtype = np.uint8) * 16).reshape(equ1.shape[0],equ1.shape[1])
four_bit_img = (np.array([int(i[4])+128 for i in lst],dtype = np.uint8) * 8).reshape(equ1.shape[0],equ1.shape[1])
three_bit_img = (np.array([int(i[5])+128 for i in lst],dtype = np.uint8) * 4).reshape(equ1.shape[0],equ1.shape[1])
two_bit_img = (np.array([int(i[6])+128 for i in lst],dtype = np.uint8) * 2).reshape(equ1.shape[0],equ1.shape[1])
one_bit_img = (np.array([int(i[7])+128 for i in lst],dtype = np.uint8) * 1).reshape(equ1.shape[0],equ1.shape[1])

finalr = cv2.hconcat([eight_bit_img1,seven_bit_img1,six_bit_img,five_bit_img])
finalv =cv2.hconcat([four_bit_img,three_bit_img,two_bit_img,one_bit_img])
bruh = cv2.hconcat([seven_bit_img1,seven_bit_img2])
bruh_8 = cv2.hconcat([eight_bit_img1,eight_bit_img2])
cv2.imwrite('7_slicing.bmp',bruh)
cv2.imwrite('8_slicing.bmp',bruh_8)
# Vertically concatenate
final = cv2.vconcat([finalr,finalv])
# cv2.imshow('oo',final)
# cv2.imwrite('equ.bmp',final)
# cv2.waitKey(0)


#################################################################
#                                                               #                
#                       Optical Flow                            #
#                                                               #
#################################################################
flow = cv2.calcOpticalFlowFarneback(seven_bit_img1, seven_bit_img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                                      
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
