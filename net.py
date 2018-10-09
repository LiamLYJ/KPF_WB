import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from tf_utils import *


def convolve_subset(inputs, ch, N, D=3):
    inputs0 = inputs
    print ('Downsample')
    inputs = batch_down2(inputs)
    for d in range(D):
        print ('Pre-Layer with {} channels at N={}'.format(ch, N))
        inputs = tf.layers.conv2d(
            inputs, ch, 3, padding='same', activation=tf.nn.relu)

    if N > 0:
        ch1 = ch*2 if ch < 512 else ch
        print ('Recursing to ch={} and N={}'.format(ch1, N-1))
        inputs, core = convolve_subset(inputs, ch1, N-1)
        for d in range(D):
            print ('Post-Layer with {} channels at N={}'.format(ch, N))
            inputs = tf.layers.conv2d(
                inputs, ch, 3, padding='same', activation=tf.nn.relu)
    else:
        core = inputs

    print ('Upsample and Skip')
    inputs = tf.image.resize_images(
        inputs,
        [tf.shape(inputs)[1]*2, tf.shape(inputs)[2]*2],
        method=tf.image.ResizeMethod.BILINEAR)
    inputs = tf.concat([inputs, inputs0], axis=-1)
    return inputs, core


def convolve_net(input_stack,  final_K, final_W, ch0=64, N=4, D=3, scope='cnet2',
                  equiv=False, separable=False, bonus=False, reuse = False, ):
    with tf.variable_scope(scope) as net_scope:
        if reuse:
            net_scope.reuse_variables()
        initial_W = input_stack.get_shape().as_list()[-1]
        inputs = input_stack
        if not separable:
            ch_final = final_K ** 2 * initial_W * final_W
        else:
            ch_final = final_K * 2 * initial_W * final_W
        # ch = 2**(10-N)
        ch = ch0
        for d in range(D):
            print ('Pre-Layer with {} channels at N={}'.format(ch, N))
            inputs = tf.layers.conv2d(
                inputs, ch, 3, padding='same', activation=tf.nn.relu)

        inputs_at_0 = inputs

        for i in range(1):
            print ('Downsample')
            inputs = batch_down2(inputs)
            ch = ch * 2
            N = N-1
            for d in range(D):
                print ('Pre-Layer with {} channels at N={}'.format(ch, N))
                inputs = tf.layers.conv2d(
                    inputs, ch, 3, padding='same', activation=tf.nn.relu)

        inputs, core = convolve_subset(inputs, ch=ch*2, N=N-1, D=D)

        N = N+1
        ch = ch_final
        for d in range(2):
            print ('Post-Layer with {} channels at N={}'.format(ch, N))
            inputs = tf.layers.conv2d(
                inputs, ch, 3, padding='same', activation=tf.nn.relu)

        if not equiv:
            ch = ch_final
            print ('Final-Layer with {} channels at N={}'.format(ch, N))
            inputs = tf.layers.conv2d(
                inputs, ch, 3, padding='same', activation=None)
            # print ('final inputs: ', inputs)

            if False:
                inputs = tf.nn.relu(inputs)
                print ('Upsample')
                inputs = tf.image.resize_images(
                    inputs,
                    [tf.shape(input_stack)[1], tf.shape(input_stack)[2]],
                    method=tf.image.ResizeMethod.BILINEAR)
                N = N+1
                inputs = tf.concat([inputs, inputs_at_0], axis=-1)
                for d in range(2):
                    print ('bonus/Post-Layer with {} channels at N={}'.format(ch, N))
                    inputs = tf.layers.conv2d(
                        inputs, ch, 3, padding='same', activation=tf.nn.relu)

                print ('bonus/Final-Layer with {} channels at N={}'.format(ch, N))
                inputs = tf.layers.conv2d(
                    inputs, ch, 3, padding='same', activation=None)

            else:
                print ('Upsample')
                inputs = tf.image.resize_images(
                    inputs,
                    [tf.shape(input_stack)[1], tf.shape(input_stack)[2]],
                    method=tf.image.ResizeMethod.BILINEAR)
            # inputs = tf.nn.relu(inputs)
        return inputs

def convolve_net_v1(input_stack,  final_K, final_W, ch0=64, N=4, D=3, scope='cnet2'):
    with tf.variable_scope(scope):
        initial_W = input_stack.get_shape().as_list()[-1]
        inputs = input_stack
        ch_final = final_K ** 2 * initial_W * final_W
        # ch = 2**(10-N)
        ch = ch0
        for d in range(D):
            print ('Pre-Layer with {} channels at N={}'.format(ch, N))
            inputs = tf.layers.conv2d(
                inputs, ch, 3, padding='same', activation=tf.nn.relu)

        inputs_at_0 = inputs

        for i in range(1):
            print ('Downsample')
            inputs = batch_down2(inputs)
            ch = ch * 2
            N = N-1
            for d in range(D):
                print ('Pre-Layer with {} channels at N={}'.format(ch, N))
                inputs = tf.layers.conv2d(
                    inputs, ch, 3, padding='same', activation=tf.nn.relu)

        inputs, core = convolve_subset(inputs, ch=ch*2, N=N-1, D=D)

        N = N+1
        ch = ch_final
        for d in range(2):
            print ('Post-Layer with {} channels at N={}'.format(ch, N))
            inputs = tf.layers.conv2d(
                inputs, ch, 3, padding='same', activation=tf.nn.relu)

        ch = ch_final
        print ('Final-Layer with {} channels at N={}'.format(ch, N))
        inputs = tf.layers.conv2d(
            inputs, ch, 3, padding='same', activation=None)
        # print ('final inputs: ', inputs)

        print ('Upsample')
        filts = tf.image.resize_images(
            inputs,
            [tf.shape(input_stack)[1], tf.shape(input_stack)[2]],
            method=tf.image.ResizeMethod.BILINEAR)

        print ('additional conv for get rgb pixel wise gain')
        ch = final_W
        inputs = tf.nn.relu(inputs)
        filts_gain = tf.layers.conv2d(
            inputs, ch, 3, padding = 'same', activation = None)
        filts_gain = tf.image.resize_images(
            filts_gain,
            [tf.shape(input_stack)[1], tf.shape(input_stack)[2]],
            method=tf.image.ResizeMethod.BILINEAR)
        return filts, filts_gain
