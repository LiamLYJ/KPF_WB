import os
import numpy as np
import operator
import math

def get_concat(input,gt,est):
    concat = np.concatenate([input, gt, est], axis = 2) / 255.0
    concat = np.clip(concat, 0, 1)
    # concat = np.clip(np.power(concat, 1/2.2), 0, 1)
    return concat

def np_convolve(img_stack, filts, final_K, final_W, spatial=True):
    noisy = img_stack
    initial_W = img_stack.shape[-1]
    kpad = final_K//2
    ch = noisy.shape[-1]
    ch1 = final_W
    sh = noisy.shape
    h, w = sh[1], sh[2]
    noisy = np.pad(noisy, [[0, 0], [kpad, kpad], [
                   kpad, kpad], [0, 0]], mode='constant')
    img_stack = []
    for i in range(final_K):
        for j in range(final_K):
            img_stack.append(noisy[:, i:h+i, j:w+j, :])
    img_stack = np.stack(img_stack, axis=-2)  # [batch, h, w, K**2, ch]

    A = np.reshape(img_stack, [sh[0], h, w, final_K**2 * ch, 1])

    fsh = filts.shape
    x = np.reshape(filts, [fsh[0], fsh[1] if spatial else 1, fsh[2] if spatial else 1,
                           final_K ** 2 * initial_W, final_W])

    return np.sum(A * x, axis=-2)


def compute_rate_confidence(filts, img, final_K, final_W, sel_ch, ref_ch):
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
    filts = np.squeeze(filts[...,sel_ch])
    dot_results = img_stack * filts # [batch, h, w, final_K**2 , input_c]
    dot_sel = dot_results[..., sel_ch] # [batch, h, w, final_K**2, 1]
    dot_ref = dot_results[..., ref_ch]
    dot_sel = np.squeeze(dot_sel)
    dot_ref = np.squeeze(dot_ref)
    dot_sel_sum = np.sum(dot_sel, axis = -1)
    dot_ref_sum = np.sum(dot_ref, axis = -1)
    batch_confidence = dot_sel_sum / dot_ref_sum # [batch , h, w]
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

if __name__ == '__main__':
    # filts = np.random.randn(5,5,3)
    # filts = np.arange(18).reshape(3,3,2)
    # final_val = compute_confidence(filts)
    # print (final_val)

    # filts = np.arange(2*5*5*3*3*3*3).reshape(2,5,5,3*3*3,3)
    filts = np.ones([2,5,5,3*3*3,3])
    # img = np.arange(2*5*5*3).reshape(2,5,5,3)
    img = np.ones([2,5,5,3])
    val = compute_rate_confidence(filts,img,3,3,0,1)
    print (val)
