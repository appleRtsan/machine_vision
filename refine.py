import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import ttest_ind
import csv

img1 = cv2.imread('dumptruck1_360x270.bmp')
img2 = cv2.imread('dumptruck2_360x270.bmp')
# img1 = cv2.imread('basketball1_360x270.bmp')
# img2 = cv2.imread('basketball2_360x270.bmp')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#################################################################
#                                                               #                
#                      histogram specification                  #
#                                                               #
#################################################################


hist,bins = np.histogram(img1.flatten(),256,[0,256])
# plt.hist(img1.flatten(),256,[0,256], color = 'r', label = 'ori')


equ1 = cv2.equalizeHist(gray1)
equ2 = cv2.equalizeHist(gray2)

res1 = np.hstack((gray1,equ1))
res2 = np.hstack((gray2,equ2))
# plt.hist(equ1.flatten(),256,[0,256], color = 'b',  label = 'equalize')
# plt.xlabel("intensity", size = 20)
# plt.ylabel("amount", size = 20)
# plt.legend(loc = 'best')
# plt.show()

#################################################################
#                                                               #                
#                      Edge  Sharpening                         #
#                                                               #
#################################################################

# src_gray = cv2.GaussianBlur(gray1, (3, 3), 0)
# kernal = np.array([[ -1, -1, -1] ,[-1, 9, -1], [-1 , -1 , -1] ])
# identity1 = cv2.filter2D(src=equ1, ddepth=-1, kernel=kernal)
# identity2 = cv2.filter2D(src=equ2, ddepth=-1, kernel=kernal)
# r = cv2.hconcat([equ1, identity1])
# cv2.imwrite('after_kernal_car.bmp',r)
# dst = cv2.Laplacian(src_gray, cv2.CV_16S, ksize=3)
# abs_dst1 = cv2.convertScaleAbs(dst)
# src_gray = cv2.GaussianBlur(gray2, (3, 3), 0)
# dst = cv2.Laplacian(src_gray, cv2.CV_16S, ksize=3)
# abs_dst2 = cv2.convertScaleAbs(dst)
# # cv2.imshow('a',abs_dst2)
# cv2.waitKey(0)


#################################################################
#                                                               #                
#                       Bit-plane Slicing                       #
#                                                               #
#################################################################
lst=[]
for i in range(equ2.shape[0]):
    for j in range(equ2.shape[1]):
         lst.append(np.binary_repr(equ2[i][j] ,width=8)) # width = no. of bits

eight_bit_img2 = (np.array([int(i[0]) for i in lst],dtype = np.uint8) * 128).reshape(gray1.shape[0],gray1.shape[1])
seven_bit_img2 = (np.array([int(i[1])+255 for i in lst],dtype = np.uint8) * 64).reshape(gray1.shape[0],gray1.shape[1])
six_bit_img = (np.array([int(i[2])+255 for i in lst],dtype = np.uint8) * 32).reshape(gray1.shape[0],gray1.shape[1])
five_bit_img = (np.array([int(i[3])+255 for i in lst],dtype = np.uint8) * 16).reshape(gray1.shape[0],gray1.shape[1])
four_bit_img = (np.array([int(i[4])+255 for i in lst],dtype = np.uint8) * 8).reshape(gray1.shape[0],gray1.shape[1])
three_bit_img = (np.array([int(i[5])+255 for i in lst],dtype = np.uint8) * 4).reshape(gray1.shape[0],gray1.shape[1])
two_bit_img = (np.array([int(i[6])+255 for i in lst],dtype = np.uint8) * 2).reshape(gray1.shape[0],gray1.shape[1])
one_bit_img = (np.array([int(i[7])+255 for i in lst],dtype = np.uint8) * 1).reshape(gray1.shape[0],gray1.shape[1])

finalr = cv2.hconcat([eight_bit_img2,seven_bit_img2,six_bit_img,five_bit_img])
finalv =cv2.hconcat([four_bit_img,three_bit_img,two_bit_img,one_bit_img])

# Vertically concatenate
final = cv2.vconcat([finalr,finalv])
# cv2.imshow('o',final)
cv2.imwrite('kernal_8_slice.bmp',final)

lst=[]
for i in range(equ1.shape[0]):
    for j in range(equ1.shape[1]):
         lst.append(np.binary_repr(equ1[i][j] ,width=8)) # width = no. of bits

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
# cv2.imshow('7_slicing.bmp',bruh)
# cv2.imshow('8_slicing.bmp',bruh_8)
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

flow2 = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
flow = cv2.calcOpticalFlowFarneback(seven_bit_img1, seven_bit_img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

mag2, ang2 = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])


hsv = np.zeros_like(img1)
hsv[..., 1] = 255
hsv[..., 0] = ang*180/np.pi/2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# step = 9
# plt.imshow(gray1)
# plt.quiver(np.arange(0, flow.shape[1], step), np.arange( 0,flow.shape[0], step), 
#            flow[::step, ::step, 0], flow[::step, ::step, 1])

# plt.legend(loc = 'best')
# plt.show()

#################################################################
#                                                               #                
#                        Analysis                               #
#                                                               #
#################################################################


# plt.scatter(ang2*180/math.pi, mag2, color = 'black', alpha= 0.2, label = 'refine')
# plt.scatter(ang*180/math.pi, mag, color = 'cyan', alpha= 0.2, label = 'original')
# print(mag.flatten())
mag_flat = mag.flatten()
a = np.isfinite(mag.flatten())
for i in range(len(a)):
    if not a[i]:
        mag_flat[i] = 0
mag_flat2 = mag2.flatten()
print("mean = ",mag_flat.mean() )
print("std = ",mag_flat.std() )

a2 = np.isfinite(mag2.flatten())
for i in range(len(a)):
    if not a2[i]:
        mag_flat2[i] = 0

print("mean2 = ",mag_flat2.mean() )
print("std2 = ",mag_flat2.std() )
print(ttest_ind(mag_flat,mag_flat2))
# plt.scatter(mag_flat, mag_flat2, color = 'red', alpha= 0.1, label = 'original')

with open('output.csv',"a", newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(a2)):
        writer.writerow([mag_flat[i], mag_flat2[i]])
# ttest,pval = ttest_ind(mag_flat,mag_flat2)
# print('pval = ',pval)
# plt.legend(loc = 'best')
# plt.show()


