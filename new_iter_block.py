import os
import scipy
from scipy import optimize
from skimage import io, color
import numpy as np
import cv2
from skimage.transform import resize
from utils import *
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import glob
import time


# est_img = np.load(data_root+'000_00.npy')
# print(np.amin(est_img), np.amax(est_img))
# plt.imshow(np.clip(est_img, 0, 255))
# plt.show()

# file_name = './000_10.png'
file_name = './000_01.png'
concat_img = cv2.imread(file_name)
h,w,c = concat_img.shape
w_cut = w //4
final_concat = np.zeros((h,w+w_cut,c), dtype=np.uint8)
final_concat[:,:4*w_cut,:] = concat_img
img0 = concat_img[:,0:w_cut,:].astype(np.float64)
img2 = concat_img[:,2*w_cut:3*w_cut,:].astype(np.float64)
# img0, img2 = filter_img(img0, img2)

# def f(x,img_src,img_target):
#     x = np.reshape(x,-1)
#     img_src = np.clip(img_src * x[::-1], 0, 255)
#     # img0 = color.rgb2lab(img0)
#     # img1 = color.rgb2lab(img1)
#     loss = np.sum((img_src - img_target)**2) / 2
#     return loss

# a = optimize.fmin(f, [1.0, 1.0, 1.0], args= (img0, img2))
# print("global gain: ", a)
# a = np.reshape(a,-1)
# img_result=np.clip(img0 * a[::-1], 0, 255)
# cv2.imwrite(data_root+img_id+'_global.png', img_result)

##################################
# Clustering for blocks WB
##################################
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
self_eps = 1e-5

gain_map = (img2+self_eps) / (img0+self_eps)
width = gain_map.shape[0]
height = gain_map.shape[1]

gain_map.resize((width*height, gain_map.shape[2]))
# gain_map = gain_map/np.linalg.norm(gain_map)
print("gain map min: ", np.amin(gain_map), " max: ", np.amax(gain_map))

print ('start clustering')
start_time = time.time()
# db = SpectralClustering(n_clusters=3).fit(gain_map)
# db = SpectralClustering(n_clusters=3, n_init = 3, n_jobs = -1).fit(gain_map)
# db = SpectralClustering(n_clusters=3, n_jobs = -1).fit(gain_map)
db = KMeans(n_clusters=3).fit(gain_map)
# db = AgglomerativeClustering(n_clusters=3).fit(gain_map)
elapsed_time = time.time() - start_time
print ('elapsed_time: ', elapsed_time)
print ('finish clustering')
labels = db.labels_

############################################
# Compute gain iteratively for each cluster
############################################
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
        for each in np.linspace(0, 1, len(unique_labels))]

def errPerCluster(gain_old, pixs_src, pixs_target):
    gain_old = np.reshape(gain_old, -1)
    pixs_src = np.clip(pixs_src * gain_old[::-1], 0, 255)
    loss = np.sum((pixs_src - pixs_target)**2) / 2
    return loss

final_image = np.zeros((gain_map.shape[0],4), np.float32)
final_box = np.zeros((gain_map.shape[0],3), np.float32)
all_idx_ = np.arange(len(gain_map))
for k, col in zip(unique_labels, colors):
    class_member_mask = (labels == k)

    final_image[class_member_mask] = col

    print('Cluster ', k, ':')
    mask_k = np.reshape(class_member_mask, (width, height))
    print('Num of pixel contains: ', np.sum(mask_k))
    optim_gain_k = optimize.fmin(errPerCluster, [1.0, 1.0, 1.0], args=(img0[mask_k], img2[mask_k]))
    # print('loss: ', errPerCluster(optim_gain_k, img0[mask_k], img2[mask_k]))
    optim_gain_k = np.reshape(optim_gain_k,-1)
    final_box[class_member_mask] = optim_gain_k
    img0[mask_k] *= optim_gain_k[::-1]

img_result_local = np.clip(img0, 0, 255)
cv2.imwrite('spectral.png', img_result_local.astype(np.uint8))

final_concat[:,4*w_cut:,:] = img_result_local.astype(np.uint8)
cv2.imwrite('final_concat.png', final_concat)
#
final_image.resize((width, height, 4))
final_image*=255.0
cv2.imwrite('clustering.png', final_image.astype(np.uint8))
