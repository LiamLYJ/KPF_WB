import os
import tensorflow as tf
slim = tf.contrib.slim
dataset_data_provider = slim.dataset_data_provider
dataset = slim.dataset
queues = slim.queues
gfile = tf.gfile
from scipy.misc import imsave

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tf_utils import *
from utils import *


def load_batch(dataset_dir, dataset_file_name, batch_size=32, height=64, width=64, channel = 3, shuffle = True, use_ms = False,
                    with_only_gain = False, with_file_name_gain = False, aug_color =0.0, aug_color_offdiag = 0.0, is_16bit = False):
    file_names, labels = read_label_file(dataset_file_name)
    file_names = [os.path.join(dataset_dir, fp) for fp in file_names]

    file_names = tf.convert_to_tensor(file_names, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    file_name_tmp, label = tf.train.slice_input_producer([file_names, labels],
                                                 shuffle=shuffle)
    file_name = tf.read_file(file_name_tmp)
    if is_16bit:
        image = tf.image.decode_png(file_name, dtype = tf.uint16, channels=channel)
        image = image / 65535.0 * 255.0
    else:
        image = tf.image.decode_png(file_name, channels=channel)

    image, label = color_augment(image, label, aug_color, aug_color_offdiag)
    image =  tf.image.resize_images(image, [height, width])

    image_after = tf.clip_by_value(tf.reshape(image,[-1, channel]) * tf.reshape(label,[-1, channel]), 0, 255)
    image_after = tf.reshape(image_after, [height, width, channel])

    if use_ms:
        file_name_ms_tmp, label_ms = tf.train.slice_input_producer([file_names, labels],
                                                     shuffle=True)
        file_name_ms = tf.read_file(file_name_ms_tmp)
        if is_16bit:
            image_ms = tf.image.decode_png(file_name_ms, dtype = tf.uint16, channels=channel)
            image = image / 65535.0 * 255.0
        else:
            image_ms = tf.image.decode_png(file_name_ms, channels=channel)

        image_ms, label_ms = color_augment(image_ms, label_ms, aug_color, aug_color_offdiag)
        image_ms =  tf.image.resize_images(image_ms, [height, width])
        image_after_ms = tf.clip_by_value(tf.reshape(image_ms,[-1, channel]) * tf.reshape(label_ms,[-1, channel]), 0, 255)
        image_after_ms = tf.reshape(image_after_ms, [height, width, channel])

        cut_range = tf.random_uniform([],0.1,0.9)
        coin = tf.random_uniform([])
        # below will cause gt and est use differnt random crop parmeters
        # def f1(img0,img1):
        #     width_0 = tf.cast(width * crop_range, tf.int32)
        #     width_1 = width - width_0
        #     img0 = tf.random_crop(img0,[height, width_0,3])
        #     img1 = tf.random_crop(img1,[height, width_1,3])
        #     return tf.concat([img0,img1],axis = 1 )
        # def f2(img0,img1):
        #     height_0 = tf.cast(height * crop_range, tf.int32)
        #     height_1 = height - height_0
        #     img0 = tf.random_crop(img0,[height_0, width,3])
        #     img1 = tf.random_crop(img1,[height_1, width,3])
        #     return tf.concat([img0,img1],axis = 0 )

        # image = tf.cond(tf.less(coin,0.5), lambda : f1(image, image_ms),lambda : f2(image, image_ms))
        # image_after = tf.cond(tf.less(coin,0.5),lambda : f1(image_after, image_after_ms),lambda : f2(image_after, image_after_ms))

        image,image_after = tf.cond(tf.less(coin,0.5),
                    lambda : cut_and_concat(image, image_ms, image_after, image_after_ms, height, width, cut_range, coin = 1),
                    lambda : cut_and_concat(image, image_ms, image_after, image_after_ms, height, width, cut_range, coin = 0) )


        # for feed into queue, need to explicit the shape of data
        image =  tf.reshape(image, [height, width, channel])
        image_after = tf.reshape(image_after, [height, width, channel])

    if with_only_gain:
        # if use gain, it can not be ms
        assert (not use_ms and not with_file_name_gain)
        # label = tf.reshape(label, [-1]) / tf.norm(label)
        label = tf.reshape(label, [-1])
        X, Y, gt_gain = tf.train.batch([image, image_after, label], batch_size=batch_size,
                        capacity = batch_size*8, num_threads=2)
        return X,Y,gt_gain

    elif with_file_name_gain:
        assert (not use_ms and not with_only_gain)
        label = tf.reshape(label, [-1])
        # label = tf.reshape(label, [-1]) / tf.norm(label)
        file_name_tmp = tf.reshape(file_name_tmp, [-1])
        X, Y, gt_gain, img_name = tf.train.batch([image, image_after, label, file_name_tmp], batch_size=batch_size,
                        capacity= batch_size*8, num_threads=2)
        return X, Y, gt_gain, img_name

    else:
        X, Y = tf.train.batch([image, image_after], batch_size=batch_size,
                  capacity=batch_size * 8,
                  num_threads=2)
        return X,Y


if __name__ == '__main__':
    dataset_dir = './data/gehler'
    dataset_file_name = 'data_txt_file/gehler_train.txt'
    batch_size = 2
    height = width = 256
    # image, image_after = load_batch(dataset_dir, dataset_file_name, batch_size, height, width, use_ms = True)
    # image, image_after, gt = load_batch(dataset_dir, dataset_file_name, batch_size, height, width, with_only_gain = True)
    image, image_after, gt, img_name = load_batch(dataset_dir, dataset_file_name, batch_size, height, width, with_file_name_gain = True)
    save_path = './check_dump'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for i in range(4):
            # image_, image_after_ = sess.run([image, image_after])
            image_, image_after_, gt_, img_name_ = sess.run([image, image_after, gt, img_name])
            for j in range(batch_size):
                # imsave(os.path.join(save_path, '%03d_%02d_input.png'%(i,j)), image_[j])
                current_file_name = img_name_[j][0].decode('utf-8').split('/')[-1]
                imsave(os.path.join(save_path, current_file_name), image_[j])
                imsave(os.path.join(save_path, current_file_name), image_after_[j])
                # imsave(os.path.join(save_path, '%03d_%02d_output.png'%(i,j)), image_after_[j])
        coord.request_stop()
        coord.join(threads)
