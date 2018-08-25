import os
import scipy
from scipy import optimize
from skimage import io, color
import numpy as np
import cv2
from skimage.transform import resize
from utils import *

import matplotlib.pyplot as plt

data_root = 'data/multi_source_results/'

# est_img = np.load(data_root+'000_00.npy')
# print(np.amin(est_img), np.amax(est_img))
# plt.imshow(np.clip(est_img, 0, 255))
# plt.show()

img_id = '000_01'
concat_img = cv2.imread(data_root+img_id+'.png')
h,w,c = concat_img.shape
w_cut = w //4
img0 = concat_img[:,0:w_cut,:]
img2 = concat_img[:,2*w_cut:3*w_cut,:]

def f(x,img_src,img_target):
    x = np.reshape(x,-1)
    img_src = np.clip(img_src * x[::-1], 0, 255)
    # img0 = color.rgb2lab(img0)
    # img1 = color.rgb2lab(img1)
    loss = np.sum((img_src - img_target)**2) / 2
    return loss

a = optimize.fmin(f, [1.0, 1.0, 1.0], args= (img0, img2))
print(a)
a = np.reshape(a,-1)
img_result=np.clip(img0 * a[::-1], 0, 255)
cv2.imwrite(data_root+img_id+'_global.png', img_result)

##################################
# Clustering for blocks WB
##################################
from sklearn.cluster import DBSCAN
self_eps = 1e-5

gain_map = (img2+self_eps) / (img0+self_eps)
width = gain_map.shape[0]
height = gain_map.shape[1]

gain_map.resize((width*height, gain_map.shape[2]))

db = DBSCAN(eps=0.5, min_samples=10).fit(gain_map)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
        for each in np.linspace(0, 1, len(unique_labels))]

final_image = np.zeros((gain_map.shape[0],4), np.float32)
all_idx_ = np.arange(len(gain_map))
for k, col in zip(unique_labels, colors):
    class_member_mask = (labels == k)

    idx_core_k_bool = class_member_mask & core_samples_mask
    idx_out_k_bool = class_member_mask & ~core_samples_mask

    idx_core_k = all_idx_[idx_core_k_bool]
    idx_out_k = all_idx_[idx_out_k_bool]

    for i in idx_core_k:
        final_image[i] = col

    for i in idx_out_k:
        final_image[i] = col

final_image.resize((width, height, 4))
# final_image*=255
# cv2.imwrite(workroot+result_subroot+'/DBSCAN_'+str(eps)+'.png', final_image.astype(np.uint8))
cv2.imshow('DBSCAN clustering', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# a = optimize.fmin(f, [1.0, 1.0, 1.0], args= (img0, img2))
# print(a)
# a = np.reshape(a,-1)
# img_result=np.clip(img0 * a[::-1], 0, 255)
# cv2.imwrite(data_root+img_id+'_blocks.png', img_result)

