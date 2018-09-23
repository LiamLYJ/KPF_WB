import os
import time
import numpy as np
import operator
import math
from scipy import optimize
from scipy.stats import gmean
from skimage.util import random_noise
from skimage import transform
import random
from skimage.io import imread
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import cv2
import json


def data_loader_np(data_folder, data_txt, patch_size, start_index, batch_size, use_ms = False):
    file_names, labels = read_label_file(data_txt, start_index, batch_size)
    def _read(filename):
        _img = imread(filename)
        _img = cv2.resize(_img, (patch_size, patch_size))
        return _img

    imgs = list(map(_read, [os.path.join(data_folder, item) for item in file_names]))
    imgs_gt = [apply_gain(img_item, label_item) for img_item, label_item in zip(imgs,labels)]
    if not use_ms:
        imgs = np.stack(imgs, axis = 0)
        imgs_gt = np.stack(imgs_gt, axis = 0)
        return imgs, imgs_gt, labels, file_names
    else:
        imgs_concat = []
        imgs_concat_gt = []
        labels_couple = []
        files_names_couple = []
        configs = []
        for _ in range(batch_size):
            img1_index, img2_index = np.random.randint(0, batch_size), np.random.randint(0, batch_size)
            coin = np.random.randint(0,2)
            split = np.random.uniform(0.2,0.8)
            offset_1 = np.random.uniform(0, 1-split)
            offset_2 = np.random.uniform(0, split)
            config = {'coin':coin, 'split':split, 'offset_1': offset_1, 'offset_2':offset_2}
            img1 = imgs[img1_index]
            img2 = imgs[img2_index]
            img1_gt = imgs_gt[img1_index]
            img2_gt = imgs_gt[img2_index]
            img_concat_tmp, config = get_concat_config(img1, img2, config)
            imgs_concat.append(img_concat_tmp)
            img_concat_tmp, _ = get_concat_config(img1_gt, img2_gt, config)
            imgs_concat_gt.append(img_concat_tmp)
            labels_couple.append((labels[img1_index], labels[img2_index]))
            files_names_couple.append((file_names[img1_index], file_names[img2_index]))
            config['file1_2'] = [file_names[img1_index], file_names[img2_index]]
            config['label1_2'] = [labels[img1_index], labels[img2_index]]
            configs.append(config)
        imgs_concat = np.stack(imgs_concat, axis = 0)
        imgs_concat_gt = np.stack(imgs_concat_gt, axis = 0)
        return imgs_concat, imgs_concat_gt, labels_couple, files_names_couple, configs

def apply_gain(img, label):
    h,w,c = img.shape
    img_after = np.clip(np.reshape(img,[-1, c]) * np.reshape(label,[-1, c]), 0.0, 255.0)
    img_after = np.reshape(img_after, [h,w,c])
    return img_after

def apply_gain_box(img, gain_box, h_scale, w_scale):
    img = img.copy()
    h,w,c = img.shape
    h_count = h // h_scale
    w_count = w //w_scale
    assert (h_count == gain_box.shape[0] and w_count == gain_box.shape[1])
    for j in range(h_count):
        for i in range(h_count):
            img[j*h_scale:(j+1)*h_scale, i*w_scale:(i+1)*w_scale,:] = \
                apply_gain(img[j*h_scale:(j+1)*h_scale, i*w_scale:(i+1)*w_scale,:], gain_box[j,i,:])
    return img

def get_concat_config(img1, img2, config):
    # 1: left,top, 2:right down
    coin = config['coin']
    split = config['split']
    offset_1 = config['offset_1']
    offset_2 = config['offset_2']
    h, w, _ = img1.shape
    config['height']= h
    config['width'] = w
    if coin == 0:
        # concat left right '
        left_start = int(w * offset_1)
        left_end = int(left_start + split * w)
        right_start = int(w * offset_2)
        right_end = int(right_start + w  - split * w)
        img1 = img1[:, left_start:left_end, :]
        img2 = img2[:, right_start:(right_end+1), :]
        config['left'] = '%d_%d'%(left_start, left_end)
        config['right'] = '%d_%d'%(right_start, right_end + 1)
        img = np.concatenate([img1, img2], axis = 1)
    else:
        # concat top down
        top_start = int(h * offset_1)
        top_end = int(top_start + split * h)
        down_start = int(h * offset_2)
        down_end = int(down_start + h  - split * h)
        img1 = img1[top_start: top_end, :, :]
        img2 = img2[down_start: (down_end+1), :, :]
        config['top'] = '%d_%d'%(top_start, top_end)
        config['down'] = '%d_%d'%(down_start, down_end+1)
        img = np.concatenate([img1, img2], axis = 0)
    return img, config


def batch_stable_process(img_batch, use_crop, use_clip, use_flip, use_rotate, use_noise):
    b,h,w,_ = img_batch.shape
    img_batch_after = []
    for index in range(b):
        img = img_batch[index,...]
        if use_crop:
            img = random_crop(img)
        if use_clip:
            img = random_clip(img)
        if use_flip:
            img =  random_flip(img)
        if use_rotate:
            img = random_rotate(img)
        if use_noise:
            img = random_add_noise(img)
        img_batch_after.append(img)
    img_batch_after = np.stack(img_batch_after, axis = 0)
    return img_batch_after

def random_crop(img, size = None):
    h,w, _ = img.shape
    if size is None:
        size = int(np.random.uniform(0.6, 1) * h)
    start_y = np.random.randint(0, h-size)
    start_x = np.random.randint(0, w-size)
    img_tmp = img[start_y: size+ start_y, start_x: start_x + size]
    return cv2.resize(img_tmp, (h,w))

def random_clip(img, rate = None):
    if rate is None:
        rate = np.random.uniform(0,0.2)
    img_max = img.max()
    img_min = img.min()
    img = np.clip(img, img_min * (1+rate), img_max * (1 - rate))
    return img

def random_flip(img, flag = None):
    if flag is None:
        flag = np.random.randint(0,2)
    if flag > 0:
        img = img[:, ::-1]
    return img

def random_rotate(img, angle = None):
    if angle is None:
        angle = np.random.randint(10,60)
    h,w,_ = img.shape
    img = transform.rotate(img, angle)
    return img

def random_add_noise(img, var = None):
    # in skiiamge_utils
    if var is None:
        var = 0.005
    mean = 0.0
    gau_img = random_noise(img, mean = mean, var = var) # defalu mode is gaussian
    pos_img = random_noise(gau_img, mode = 'poisson')
    return pos_img

def encode_label(label):
    r_gain,g_gain,b_gain = [float(item) for item in label.split(',')]
    return [r_gain, g_gain, b_gain]

def read_label_file(file, start_index = None, batch_size = None):
    f = open(file, "r")
    filepaths = []
    labels = []
    tog = True
    for line in f:
        if tog:
            # -1 for earse '\n'
            filepaths.append(line[:-1])
            tog = not tog
        else:
            labels.append(encode_label(line))
            tog = not tog
    if start_index is None and batch_size is None:
        return filepaths, labels
    else:
        return filepaths[start_index:(start_index + batch_size)], labels[start_index:(start_index + batch_size)]

def get_concat(input,gt,est):
    concat = np.concatenate([input, gt, est], axis = 2) / 255.0
    concat = np.clip(concat, 0, 1)
    # concat = np.clip(np.power(concat, 1/2.2), 0, 1)
    return concat

def np_convolve(input, filts, final_K, final_W, spatial=True):
    kpad = final_K//2
    sh = input.shape
    ch = sh[-1]
    initial_W = ch
    h, w = sh[1], sh[2]
    input = np.pad(input, [[0, 0], [kpad, kpad], [
                   kpad, kpad], [0, 0]], mode='constant')
    img_stack = []
    for i in range(final_K):
        for j in range(final_K):
            img_stack.append(input[:, i:h+i, j:w+j, :])
    img_stack = np.stack(img_stack, axis=-2)  # [batch, h, w, K**2, ch]

    A = np.reshape(img_stack, [sh[0], h, w, final_K**2 * ch, 1])

    fsh = filts.shape
    x = np.reshape(filts, [fsh[0], fsh[1] if spatial else 1, fsh[2] if spatial else 1,
                           final_K ** 2 * initial_W, final_W])

    return np.sum(A * x, axis=-2)


def compute_rate_confidence(filts, img, final_K, final_W, sel_ch, ref_ch, is_spatial = False):
    img_sh = img.shape
    input_ch = img_sh[-1]
    h,w = img_sh[1], img_sh[2]
    kpad = final_K // 2
    img_pad = np.pad(img, [[0,0], [kpad, kpad], [kpad, kpad], [0,0]], mode = 'constant')
    img_stack = []
    for i in range(final_K):
        for j in range(final_K):
            img_stack.append(img_pad[:,i:h+i, j:w+j,:])
    img_stack = np.stack(img_stack, axis = -2) #[batch, h,w, K**2 ,ch]
    img_stack = np.reshape(img_stack, [img_sh[0], h, w, final_K**2, input_ch])
    filts = np.reshape(filts, [img_sh[0],h,w,final_K**2, input_ch, final_W])
    filts = filts[...,sel_ch]
    dot_results = img_stack * filts # [batch, h, w, final_K**2 , input_c]
    dot_sel = dot_results[..., sel_ch] # [batch, h, w, final_K**2, 1]
    dot_ref = dot_results[..., ref_ch] # [batch, h ,w, final_k **2 ,[ref_ch]]
    dot_sel_sum = np.mean(np.abs(dot_sel), axis = -1)
    dot_ref_sum = np.mean(np.mean(np.abs(dot_ref), axis = -1), axis = -1) + 0.00001
    batch_confidence = dot_sel_sum / dot_ref_sum # [batch , h, w]
    if not is_spatial:
        batch_confidence = np.mean(np.mean(batch_confidence, axis = -1), axis = -1) # [batch,1]

    return batch_confidence


def compute_var_confidence(filts, time_std = 2, rate = 0.8):
    filts_sh = filts.shape
    filts = np.reshape(filts, [filts_sh[0],filts_sh[1],-1])
    final_val = 0
    for i in range(filts.shape[-1]):
        matrix = filts[...,i].reshape(-1)
        matrix_var = np.nanvar(matrix)
        matrix_std = np.nanstd(matrix)

        # find cluster in matrix
        cluster = []
        for index, item in enumerate(matrix):
            low_index = matrix > (item - time_std * matrix_std)
            tmp = matrix[low_index]
            high_index = tmp < (item + time_std * matrix_std)
            tmp = tmp[high_index]
            count = len(tmp)
            if count >= rate * len(matrix):
                cluster.append(item)
        # compute var regrad to cluster
        candidate = []
        for value in cluster:
            candidate.append(matrix - value)
        candidate = np.stack(candidate, axis = -1)
        candidate = np.abs(candidate)
        pending_matrix = np.amin(candidate, axis = -1)
        pending_matrix_var = np.sum(pending_matrix * pending_matrix)
        final_val += pending_matrix_var
    return final_val


def angular_error(estimation, ground_truth):
    return 180 * (1/math.pi) * math.acos(
        np.clip(
            np.dot(estimation, ground_truth) / np.linalg.norm(estimation) /
            np.linalg.norm(ground_truth), -1, 1))

def special_downsampling(img, scale):
    h,w,c = img.shape
    h_down, w_down = h // scale, w // scale
    img_down = np.ones([h_down, w_down, c])
    for j in range(h_down):
        for i in range(w_down):
            cut_out = img[j* scale: (j+1) *scale, i*scale:(i+1)*scale,:]
            value = np.mean(np.mean(cut_out, axis = 0), axis = 0)
            img_down[j,i,:] = value
    return img_down

def summary_angular_errors(errors):
    errors = sorted(errors)

    def g(f):
        return np.percentile(errors, f * 100)

    median = g(0.5)
    mean = np.mean(errors)
    gm = gmean(errors)
    trimean = 0.25 * (g(0.25) + 2 * g(0.5) + g(0.75))
    results = {
        '25': np.mean(errors[:int(0.25 * len(errors))]),
        '75': np.mean(errors[int(0.75 * len(errors)):]),
        '95': g(0.95),
        'tri': trimean,
        'med': median,
        'mean': mean,
        'gm': gm
    }
    return results


def just_print_angular_errors(results):
    print ("25: %5.3f," % results['25'],)
    print ("med: %5.3f" % results['med'],)
    print ("tri: %5.3f" % results['tri'],)
    print ("avg: %5.3f" % results['mean'],)
    print ("75: %5.3f" % results['75'],)
    print ("95: %5.3f" % results['95'])
    print ("gm: %5.3f" % results['gm'])


def print_angular_errors(errors):
    print ("%d images tested. Results:" % len(errors))
    results = summary_angular_errors(errors)
    just_print_angular_errors(results)
    return results


def solve_gain(img, img_ref, is_single = False):
    def f(x,img0,img1):
        x = np.reshape(x,-1)
        img0 = np.clip(img0 * x, 0, 255)
        # img0 = color.rgb2lab(img0)
        # img1 = color.rgb2lab(img1)
        loss = np.sum((img0 - img1)**2) / 2
        return loss
    img = np.clip(img, 0 , 255.0)
    img_ref = np.clip(img_ref, 0 , 255.0)
    # gain_mean = np.mean(np.mean(img_ref / img, axis = 0), axis= 0)
    # gain = optimize.fmin(f, gain_mean, args= (img, img_ref))
    if not is_single:
        gain = optimize.fmin(f, [1.0,1.0,1.0], args= (img, img_ref))
    else:
        gain = optimize.fmin(f, [1.0], args = (img, img_ref))
    return gain


def filter_img(input_img, ref_img, filter_value = [0, 255]):
    input_img = input_img.copy()
    ref_img = ref_img.copy()

    h,w,c = ref_img.shape
    img = np.mean(ref_img, axis = -1)
    valid_matrix = np.logical_and(img > filter_value[0], img < filter_value[1])
    valid_points = np.where(valid_matrix)
    valid_points = np.stack([valid_points[0],valid_points[1]], axis=-1)
    non_valid_x, non_valid_y = np.where(np.logical_not(valid_matrix))
    how_many = len(non_valid_x)
    for non_valid_point in zip(non_valid_x, non_valid_y):
        print ('how many remian:', how_many)
        how_many -= 1
        non_valid_point = np.expand_dims(np.array(non_valid_point), axis =0) # 1x2
        tmp = np.concatenate([non_valid_point, valid_points]) # (all_num+1)*2
        dist_all = squareform(pdist(tmp))[0] # conpute a matrix norm, and only use one line
        min_dist = sorted(dist_all)[1] # 1 for consider the non_valid_point
        adjust_index = np.where(dist_all == min_dist)[0][0]
        adjust_x = adjust_index // h
        adjust_y = adjust_index % h
        input_img[non_valid_point[0][0], non_valid_point[0][1],:] = input_img[adjust_x, adjust_y, :]
        ref_img[non_valid_point[0][0], non_valid_point[0][1],:] = ref_img[adjust_x, adjust_y, :]
    return input_img, ref_img


def gain_fitting(img_input, img_ref, is_pure = False, is_local = True, n_clusters = 3, gamma = 0.8, with_clus = False):
    if is_local:
        self_eps = 1e-5
        gain_map = (img_ref+self_eps) / (img_input+self_eps)
        if is_pure:
            return gain_map
        width = gain_map.shape[0]
        height = gain_map.shape[1]

        gain_map = np.clip(gain_map, 0, 500)
        gain_map.resize((width*height, gain_map.shape[2]))
        print("gain map min: ", np.amin(gain_map), " max: ", np.amax(gain_map))

        print ('start clustering')
        start_time = time.time()
        db = SpectralClustering(n_clusters = n_clusters, n_jobs = -1, gamma = gamma).fit(gain_map)
        # db = KMeans(n_clusters=3).fit(gain_map)
        elapsed_time = time.time() - start_time
        print ('elapsed_time: ', elapsed_time)
        print ('finish clustering')
        labels = db.labels_

        # Compute gain iteratively for each cluster
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                for each in np.linspace(0, 1, len(unique_labels))]
        clus_img = np.zeros((gain_map.shape[0],4), np.float32)
        gain_box = np.zeros((gain_map.shape[0],3), np.float32)
        all_idx_ = np.arange(len(gain_map))
        for k, col in zip(unique_labels, colors):
            class_member_mask = (labels == k)
            clus_img[class_member_mask] = col
            print('Cluster ', k, ':')
            mask_k = np.reshape(class_member_mask, (height, width))
            print('Num of pixel contains: ', np.sum(mask_k))
            optim_gain_k = solve_gain(img_input[mask_k], img_ref[mask_k])
            # print('loss: ', errPerCluster(optim_gain_k, img0[mask_k], img2[mask_k]))
            optim_gain_k = np.reshape(optim_gain_k,-1)
            gain_box[class_member_mask] = optim_gain_k
        gain_box = gain_box.reshape([height, width, 3])
        clus_img = clus_img.reshape([height, width, -1])
        if with_clus:
            return gain_box, clus_img, labels
        else:
            return gain_box
    else:
        gain = solve_gain(img_input, img_ref)
        return gain

def gain_fitting_sep(img_input, img_ref, is_local = True, n_clusters = 2, with_clus = False):
    if is_local:
        self_eps = 1e-5
        gain_map = (img_ref+self_eps) / (img_input+self_eps)
        width = gain_map.shape[0]
        height = gain_map.shape[1]
        gain_map = np.clip(gain_map, 0, 500)
        gain_map.resize((width*height, gain_map.shape[2]))

        start_time = time.time()
        gain_box = []
        clus_box = []
        for i in range(2):
            if i == 1:
                i += 1
            print ('start clustering', i)
            db = SpectralClustering(n_clusters = n_clusters, n_jobs = -1).fit(np.expand_dims(gain_map[...,i],axis= -1))
            # db = KMeans(n_clusters = n_clusters, n_jobs = -1).fit(np.expand_dims(gain_map[...,i],axis= -1))
            elapsed_time = time.time() - start_time
            print ('elapsed_time: ', elapsed_time)
            print ('finish clustering', i)
            labels = db.labels_

            # Compute gain iteratively for each cluster
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each)
                    for each in np.linspace(0, 1, len(unique_labels))]
            clus_item = np.zeros((gain_map.shape[0],4), np.float32)
            gain_item = np.zeros((gain_map.shape[0],1), np.float32)
            for k, col in zip(unique_labels, colors):
                class_member_mask = (labels == k)
                clus_item[class_member_mask] = col
                print('Cluster ', k, ':')
                mask_k = np.reshape(class_member_mask, (height, width))
                print('Num of pixel contains: ', np.sum(mask_k))
                optim_gain_k = solve_gain(img_input[mask_k, i], img_ref[mask_k, i], is_single = True)
                optim_gain_k = np.reshape(optim_gain_k,-1)
                gain_item[class_member_mask] = optim_gain_k
            clus_item = np.reshape(clus_item, (height, width, -1))
            gain_item = np.reshape(gain_item, (height, width, -1))
            clus_box.append(clus_item)
            gain_box.append(gain_item)
        green = np.ones_like(gain_box[0])
        gain_box = np.concatenate([gain_box[0], green, gain_box[1]], axis = 2)
        if with_clus:
            return gain_box, clus_box, labels
        else:
            return gain_box
    else:
        gain = solve_gain(img_input, img_ref)
        return gain

def get_original(data_dir, json_file, original_size = 1024, with_gt = False):
    with open(json_file) as f:
        data = json.load(f)
    height_ref = data['height']
    width_ref = data['width']
    height_org = width_org = original_size

    file_name_1 = data['file1_2'][0]
    file_name_2 = data['file1_2'][1]

    img_org_1 = imread(os.path.join(data_dir, file_name_1))
    img_org_2 = imread(os.path.join(data_dir, file_name_2))

    img_org_1 = cv2.resize(img_org_1, (height_org, width_org))
    img_org_2 = cv2.resize(img_org_2, (height_org, width_org))

    scale_h = height_org // height_ref
    scale_w = width_org // width_ref

    coin = data['coin']
    if coin == 0:
        left_start = int(data['left'].split('_')[0])
        left_end = int(data['left'].split('_')[1])
        right_start = int(data['right'].split('_')[0])
        right_end = int(data['right'].split('_')[1])

        left_start *= scale_w
        right_start *= scale_w
        left_end *= scale_w
        right_end *= scale_w

        concat_left = img_org_1[:, left_start:left_end,:]
        concat_right = img_org_2[:, right_start:right_end,:]
        concat = np.concatenate([concat_left, concat_right], axis = 1)
        if with_gt:
            label1_2 = data['label1_2']
            concat_left_gt = apply_gain(concat_left, label1_2[0])
            concat_right_gt = apply_gain(concat_right, label1_2[1])
            concat_gt = np.concatenate([concat_left_gt, concat_right_gt], axis = 1)

    elif coin == 1:
        top_start = int(data['top'].split('_')[0])
        top_end = int(data['top'].split('_')[1])
        down_start = int(data['down'].split('_')[0])
        down_end = int(data['down'].split('_')[1])

        top_start *= scale_h
        top_end *= scale_h
        down_start *= scale_h
        down_end *= scale_h

        concat_top = img_org_1[top_start:top_end, :, :]
        concat_down = img_org_2[down_start:down_end, :, :]
        concat = np.concatenate([concat_top, concat_down], axis = 0)
        if with_gt:
            label1_2 = data['label1_2']
            concat_top_gt = apply_gain(concat_top, label1_2[0])
            concat_down_gt = apply_gain(concat_down, label1_2[1])
            concat_gt = np.concatenate([concat_top_gt, concat_down_gt], axis = 0)
            
    else:
        raise ValueError('bad coin')

    if with_gt:
        return concat, concat_gt
    else:
        return concat, scale_h, scale_w

def get_confi_multi(clus_labels, confi, label):
    # confi hxw
    # clus_labels: h*w
    mask = clus_labels == label
    confi = np.reshape(confi, -1)
    return np.mean(confi[mask])

if __name__ == '__main__':
    # pass
    #########################
    # test for data_loader
    #########################
    dataset_dir = "./data/sony"
    dataset_file_name = './data_txt_file/file_train.txt'
    imgs, imgs_gt, labels, file_names = data_loader_np(data_folder = dataset_dir, patch_size=128,
                            data_txt=dataset_file_name, start_index=0, batch_size=10,
                            use_ms = False)
    print (imgs[0])
    print (imgs.shape)
    print (imgs_gt.shape)
    print (labels)
    print (file_names)

    ##############################
    # test for gain fit
    ##############################
    # img1 = np.ones([10,10,3])
    # # img1[0,0,:] = np.array([255,255,255])
    # img2 = img1.copy()
    # # tmp1, tmp2 = filter_img(img1, img2)
    # # print (img1)
    # # print (np.sum(tmp1 == img1))
    # # print (np.sum(tmp2 == img2))
    # gain_box = gain_fitting(img1, img2)
    # print(gain_box)


    ###############################
    # test for compute confidence
    ###############################
    # filts = np.random.randn(5,5,3)
    # filts = np.arange(18).reshape(3,3,2)
    # final_val = compute_confidence(filts)
    # print (final_val)

    # # filts = np.arange(2*5*5*3*3*3*3).reshape(2,5,5,3*3*3,3)
    # filts = np.ones([2,5,5,3*3*3,3])
    # # img = np.arange(2*5*5*3).reshape(2,5,5,3)
    # img = np.ones([2,5,5,3])
    # val = compute_rate_confidence(filts,img,3,3,0,1)
    # print (val)
