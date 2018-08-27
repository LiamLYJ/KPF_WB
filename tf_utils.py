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


def filts_reg_loss(filts, input_ch, final_K, final_W, use_L1 = False):
    filts_sh = tf.shape(filts)
    filts = tf.reshape(filts, [filts_sh[0], filts_sh[1], filts_sh[2],
                               final_K**2, input_ch, final_W])
    loss = 0
    # surpress the filter not aligned
    for i in range(input_ch):
        for j in range(final_W):
            if i == j :
                continue
            else:
                if use_L1:
                    loss += tf.reduce_sum(tf.abs(filts[..., i, j]))
                else:
                    loss += tf.reduce_sum(tf.square(filts[...,i,j]))
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


def data_augment(input, random_crop=False, random_flip=False, random_rotate = False):
    if random_flip:
        input = tf.image.random_flip_left_right(input)

    return input

def color_augment(image, illum, aug_color, aug_color_offdiag):
    # image [h,w,3], illum [3,]
    image = tf.cast(image, tf.float32)
    image_sh = tf.shape(image)
    h,w = image_sh[0], image_sh[1]

    color_aug_matrix = []
    for i in range(3):
        for j in range(3):
            if i == j:
                value = 1 + tf.random_uniform([]) * aug_color - 0.5 * aug_color
            else:
                value = tf.random_uniform([]) * aug_color_offdiag - 0.5 * aug_color_offdiag
            color_aug_matrix.append(value)
    color_aug_matrix = tf.concat([color_aug_matrix], axis = 0)
    color_aug_matrix = tf.reshape(color_aug_matrix, [3,3])

    illum = tf.matmul(tf.reshape(illum, [-1,3]), color_aug_matrix)
    image_tmp = tf.reshape(image, [-1,3])
    image = tf.reshape(tf.matmul(image_tmp, color_aug_matrix), [h , w, 3])
    image = tf.cast(image, tf.uint8)
    illum = tf.reshape(illum, [3,])
    return image, illum

def data_augment(images,
            horizontal_flip=False,
            vertical_flip=False,
            rotate=0, # Maximum rotation angle in degrees
            crop_probability=0, # How often we do crops
            crop_min_percent=0.6, # Minimum linear dimension of a crop
            crop_max_percent=1.,  # Maximum linear dimension of a crop
            mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf

  # if images.dtype != tf.float32:
  #   images = tf.image.convert_image_dtype(images, dtype=tf.float32)
  #   images = tf.subtract(images, 0.5)
  #   images = tf.multiply(images, 2.0)

  with tf.name_scope('augmentation'):
    shp = tf.shape(images)
    batch_size, height, width = shp[0], shp[1], shp[2]
    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)

    # The list of affine transformations that our image will go under.
    # Every element is Nx8 tensor, where N is a batch size.
    transforms = []
    identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
    if horizontal_flip:
      coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
      flip_transform = tf.convert_to_tensor(
          [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
      transforms.append(
          tf.where(coin,
                   tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                   tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

    if vertical_flip:
      coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
      flip_transform = tf.convert_to_tensor(
          [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
      transforms.append(
          tf.where(coin,
                   tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                   tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

    if rotate > 0:
      angle_rad = rotate / 180 * math.pi
      angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
      transforms.append(
          tf.contrib.image.angles_to_projective_transforms(
              angles, height, width))

    if crop_probability > 0:
      crop_pct = tf.random_uniform([batch_size], crop_min_percent,
                                   crop_max_percent)
      left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
      top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
      crop_transform = tf.stack([
          crop_pct,
          tf.zeros([batch_size]), top,
          tf.zeros([batch_size]), crop_pct, left,
          tf.zeros([batch_size]),
          tf.zeros([batch_size])
      ], 1)

      coin = tf.less(
          tf.random_uniform([batch_size], 0, 1.0), crop_probability)
      transforms.append(
          tf.where(coin, crop_transform,
                   tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

    if transforms:
      images = tf.contrib.image.transform(
          images,
          tf.contrib.image.compose_transforms(*transforms),
          interpolation='BILINEAR') # or 'NEAREST'

    def cshift(values): # Circular shift in batch dimension
      return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

    if mixup > 0:
      mixup = 1.0 * mixup # Convert to float, as tf.distributions.Beta requires floats.
      beta = tf.distributions.Beta(mixup, mixup)
      lam = beta.sample(batch_size)
      ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
      images = ll * images + (1 - ll) * cshift(images)

  return images


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
