import os
import scipy
from scipy import optimize
from skimage import io, color
import numpy as np
import cv2
from skimage.transform import resize
from utils import *
# img = cv2.imread('111MSDCF_DSC07998.png')
# img = cv2.resize(img,(256,256))
# h,w,c = img.shape
# input_img = img.copy()
# ref_img = cv2.resize(input_img, (w//4, h//4))
# ref_img = np.array([2.5781,1.0,1.7617]) * ref_img[...,::-1]
# cv2.imwrite('input.png', input_img)
# cv2.imwrite('ref.png', ref_img)
# raise

file0 = 'input.png'
file1 = 'ref.png'
img0 = io.imread(file0)
scale = 4
img0 = special_downsampling(img0, scale)
img1 = io.imread(file1)
#
# h,w,c = img1.shape
# img0 = cv2.resize(img0, (w,h))
# # print (img1 / img0)
# # raise
#
# extrem_h = h // 4
# extrem_w = w // 4
# img0 = cv2.resize(img0, (extrem_w, extrem_h), interpolation=cv2.INTER_CUBIC)
# img1 = cv2.resize(img1, (extrem_w, extrem_h), interpolation=cv2.INTER_CUBIC)
# img_0 = color.rgb2lab(img_0)
# img_1 = color.rgb2lab(img_1)

#
# img0 = np.arange(6).reshape(3,2)
# img1 = img0 * np.array([2,3])
# print (img0)
# print (img1)


# img1 = np.array([2.578,1.0,1.7617]) * img0

def f(x,img0,img1):
    x = np.reshape(x,-1)
    img0 = x * img0
    # img0 = color.rgb2lab(img0)
    # img1 = color.rgb2lab(img1)
    loss = np.sum((img0 - img1)**2) / 2
    return loss

# print (img1/ img0)
a = optimize.fmin(f,[1.0,1.0,1.0], args=(img0,img1))
print (a)
print (angular_error(np.array(a), np.array([2.578,1.0,1.7617])))
