import os
import numpy as np
import operator

def get_concat(input,gt,est):
    concat = np.concatenate([input, gt, est], axis = 2) / 255.0
    concat = np.clip(concat, 0, 1)
    # concat = np.clip(np.power(concat, 1/2.2), 0, 1)
    return concat

def convolve(img_stack, filts, final_K, final_W, spatial=True):
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


def compute_confidence(filts, time_std = 2, rate = 0.8):
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

if __name__ == '__main__':
    # filts = np.random.randn(5,5,3)
    filts = np.arange(18).reshape(3,3,2)
    final_val = compute_confidence(filts)
    print (final_val)
