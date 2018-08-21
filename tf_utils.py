import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import random
import math

# mimic a multi_source
def cut_and_concat(img_0,img_1,img_2,img_3, height, width, cut_range, coin):
    if coin == 0:
        # vertical cut & concat
        height_0 = tf.cast(height * cut_range, tf.int32)
        height_1 = height - height_0

        offset_height_0 = tf.cast(tf.random_uniform([],0,(1-cut_range)) * height , tf.int32)
        crop_0 = tf.image.crop_to_bounding_box(img_0,
                offset_height=offset_height_0, offset_width=0,target_width=width,target_height=height_0)
        crop_2 = tf.image.crop_to_bounding_box(img_2,
                offset_height=offset_height_0, offset_width=0,target_width=width,target_height=height_0)
        offset_height_1 = tf.cast(tf.random_uniform([],0,cut_range) * height , tf.int32)
        crop_1 = tf.image.crop_to_bounding_box(img_1,
                offset_height=offset_height_1, offset_width=0,target_width=width,target_height=height_1)
        crop_3 = tf.image.crop_to_bounding_box(img_3,
                offset_height=offset_height_1, offset_width=0,target_width=width,target_height=height_1)
        return tf.concat([crop_0, crop_1], axis = 0), tf.concat([crop_2, crop_3], axis = 0)
    else:
        # horizontal cut & concat
        width_0 = tf.cast(width * cut_range, tf.int32)
        width_1 = width - width_0

        offset_width_0 = tf.cast(tf.random_uniform([],0,(1-cut_range)) * width , tf.int32)
        crop_0 = tf.image.crop_to_bounding_box(img_0,
                offset_height=0, offset_width=offset_width_0, target_width=width_0,target_height=height)
        crop_2 = tf.image.crop_to_bounding_box(img_2,
                offset_height=0, offset_width=offset_width_0, target_width=width_0,target_height=height)
        offset_width_1 = tf.cast(tf.random_uniform([],0,cut_range) * width , tf.int32)
        crop_1 = tf.image.crop_to_bounding_box(img_1,
                offset_height=0, offset_width=offset_width_1, target_width=width_1,target_height=height)
        crop_3 = tf.image.crop_to_bounding_box(img_3,
                offset_height=0, offset_width=offset_width_1, target_width=width_1,target_height=height)
        return tf.concat([crop_0, crop_1], axis =1 ), tf.concat([crop_2, crop_3], axis =1 )


def sRGBforward(x):
    b = .0031308
    gamma = 1./2.4
    # a = .055
    # k0 = 12.92
    a = 1./(1./(b**gamma*(1.-gamma))-1.)
    k0 = (1+a)*gamma*b**(gamma-1.)

    def gammafn(x): return (1+a)*tf.pow(tf.maximum(x, b), gamma)-a
    # gammafn = lambda x : (1.-k0*b)/(1.-b)*(x-1.)+1.
    srgb = tf.where(x < b, k0*x, gammafn(x))
    k1 = (1+a)*gamma
    srgb = tf.where(x > 1, k1*x-k1+1, srgb)
    return srgb


# batch Downsample
def batch_down2(img):
    return (img[:, ::2, ::2, ...]+img[:, 1::2, ::2, ...]+img[:, ::2, 1::2, ...]+img[:, 1::2, 1::2, ...])/4


# Loss
def gradient(imgs):
    return tf.stack([.5*(imgs[..., 1:, :-1,:]-imgs[..., :-1, :-1,:]),
                     .5*(imgs[..., :-1, 1:,:]-imgs[..., :-1, :-1,:])], axis=-1)


def gradient_loss(guess, truth):
    return tf.reduce_mean(tf.abs(gradient(guess)-gradient(truth)))


def basic_img_loss(img, truth):
    l2_pixel = tf.reduce_mean(tf.square(img - truth))
    l1_grad = gradient_loss(img, truth)
    return l2_pixel + l1_grad


def filts_reg_loss(filts, input_ch, final_K, final_W, use_l1 = False):
    filts_sh = tf.shape(filts)
    filts = tf.reshape(filts, filts_sh[0], filts_sh[1], filts_sh[2], final_K**2, input_ch, final_W)
    loss = 0
    # surpress the filter not aligned
    for i in range(input_ch):
        for j in range(final_W):
            if i == j :
                continue
            else:
                if use_L1:
                    loss += tf.sum(tf.abs(filts[..., i, j]))
                else:
                    loss += tf.sum(tf.square(filts[...,i,j]))
    return loss


def get_angular_loss(vec1, vec2, length_regularization=0.0, ANGULAR_LOSS = True ):
        with tf.name_scope('angular_error'):
            safe_v = 0.999999
            if len(vec1.get_shape()) == 2:
                illum_normalized = tf.nn.l2_normalize(vec1, 1)
                _illum_normalized = tf.nn.l2_normalize(vec2, 1)
                dot = tf.reduce_sum(illum_normalized * _illum_normalized, 1)
                dot = tf.clip_by_value(dot, -safe_v, safe_v)
                length_loss = tf.reduce_mean(
                    tf.maximum(tf.log(tf.reduce_sum(vec1**2, axis=1) + 1e-7), 0))
            else:
                assert len(vec1.get_shape()) == 4
                illum_normalized = tf.nn.l2_normalize(vec1, 3)
                _illum_normalized = tf.nn.l2_normalize(vec2, 3)
                dot = tf.reduce_sum(illum_normalized * _illum_normalized, 3)
                dot = tf.clip_by_value(dot, -safe_v, safe_v)
                length_loss = tf.reduce_mean(
                    tf.maximum(tf.log(tf.reduce_sum(vec1**2, axis=3) + 1e-7), 0))
            angle = tf.acos(dot) * (180 / math.pi)

            if ANGULAR_LOSS:
                return tf.reduce_mean(angle) + length_loss * length_regularization
            else:
                dot = tf.reduce_sum(
                    (illum_normalized - _illum_normalized)**2,
                    axis=len(illum_normalized.get_shape()) - 1)
                return tf.reduce_mean(dot) * 1000 + length_loss * length_regularization


def get_gain_from_filter(filters, final_W):
    filters_sh = tf.shape(filters)
    filters = tf.reshape(filters, [filters_sh[0], filters_sh[1], filters_sh[2], final_W])
    gain = tf.reduce_mean(filters, axis = [1,2])
    return gain


def convolve(img_stack, filts, final_K, final_W):
    initial_W = img_stack.get_shape().as_list()[-1]
    imgsh = tf.shape(img_stack)
    fsh = tf.shape(filts)
    filts = tf.reshape(filts, [fsh[0],fsh[1],fsh[2],-1])
    img_stack = tf.cond(tf.less(fsh[1], imgsh[1]), lambda: batch_down2(img_stack), lambda: img_stack)
    # print ('filts shape: ', filts.shape)
    filts = tf.reshape(
        filts, [fsh[0], fsh[1], fsh[2], final_K ** 2 * initial_W, final_W])

    kpad = final_K//2
    imgs = tf.pad(img_stack, [[0, 0], [kpad, kpad], [kpad, kpad], [0, 0]])
    ish = tf.shape(img_stack)
    img_stack = []
    for i in range(final_K):
        for j in range(final_K):
            img_stack.append(
                imgs[:, i:tf.shape(imgs)[1]-2*kpad+i, j:tf.shape(imgs)[2]-2*kpad+j, :])
    img_stack = tf.stack(img_stack, axis=-2)
    img_stack = tf.reshape(
        img_stack, [ish[0], ish[1], ish[2], final_K**2 * initial_W, 1])
    # removes the final_K**2*initial_W dimension but keeps final_W
    img_net = tf.reduce_sum(img_stack * filts, axis=-2)
    return img_net


def sep_convolve(img, filts, final_K, final_W):
    pre_img = img * filts
    return pre_img

def convolve_per_layer(input_stack, filts, final_K, final_W):
    initial_W = input_stack.get_shape().as_list()[-1]
    img_net = []
    for i in range(initial_W):
        img_net.append(
            convolve(input_stack[..., i:i+1], filts[..., i:i+1, :], final_K, final_W))
    img_net = tf.concat(img_net, axis=-1)
    return img_net



# For separable stuff
def convolve_aniso(img_stack, filts, final_Kh, final_Kw, final_W, layerwise=False):
    initial_W = img_stack.get_shape().as_list()[-1]

    fsh = tf.shape(filts)
    if layerwise:
        filts = tf.reshape(
            filts, [fsh[0], fsh[1], fsh[2], final_Kh * final_Kw,           initial_W])
    else:
        filts = tf.reshape(
            filts, [fsh[0], fsh[1], fsh[2], final_Kh * final_Kw * initial_W, final_W])

    kpadh = final_Kh//2
    kpadw = final_Kw//2
    imgs = tf.pad(img_stack, [[0, 0], [kpadh, kpadh], [kpadw, kpadw], [0, 0]])
    ish = tf.shape(img_stack)
    img_stack = []
    for i in range(final_Kh):
        for j in range(final_Kw):
            img_stack.append(
                imgs[:, i:tf.shape(imgs)[1]-2*kpadh+i, j:tf.shape(imgs)[2]-2*kpadw+j, :])
    img_stack = tf.stack(img_stack, axis=-2)
    if layerwise:
        img_stack = tf.reshape(
            img_stack, [ish[0], ish[1], ish[2], final_Kh * final_Kw, initial_W])
    else:
        img_stack = tf.reshape(
            img_stack, [ish[0], ish[1], ish[2], final_Kh * final_Kw * initial_W, 1])
    # removes the final_K**2*initial_W dimension but keeps final_W
    img_net = tf.reduce_sum(img_stack * filts, axis=-2)
    return img_net
