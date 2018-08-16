import numpy as np
import tensorflow as tf
slim = tf.contrib.slim


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
    return tf.stack([.5*(imgs[..., 1:, :-1]-imgs[..., :-1, :-1]), .5*(imgs[..., :-1, 1:]-imgs[..., :-1, :-1])], axis=-1)


def gradient_loss(guess, truth):
    return tf.reduce_mean(tf.abs(gradient(guess)-gradient(truth)))


def basic_img_loss(img, truth):
    l2_pixel = tf.reduce_mean(tf.square(img - truth))
    l1_grad = gradient_loss(img, truth)
    return l2_pixel + l1_grad




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
