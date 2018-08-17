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

def encode_label(label):
    r_gain,g_gain,b_gain = [float(item) for item in label.split(',')]
    return [r_gain, g_gain, b_gain]


def read_label_file(file):
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
    return filepaths, labels



def load_batch(dataset_dir, dataset_file_name, batch_size=32, height=64, width=64, channel = 3):
    file_names, labels = read_label_file(dataset_file_name)
    file_names = [os.path.join(dataset_dir, fp) for fp in file_names]

    file_names = tf.convert_to_tensor(file_names, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    file_name, label = tf.train.slice_input_producer([file_names, labels],
                                                 shuffle=True)
    file_name = tf.read_file(file_name)
    image = tf.image.decode_png(file_name, channels=channel)
    image =  tf.image.resize_images(image, [height, width])
    image_after = tf.clip_by_value(tf.reshape(image,[-1, channel]) * tf.reshape(label,[-1, channel]), 0, 255)
    image_after = tf.reshape(image_after, [height, width, channel])
    X, Y = tf.train.batch([image, image_after], batch_size=batch_size,
                      capacity=batch_size * 8,
                      num_threads=2)
    return X,Y


if __name__ == '__main__':
    dataset_dir = '/home/lyj/Downloads/Sony/preprocessed'
    dataset_file_name = 'tmp.txt'
    batch_size = 2
    height = width = 50
    image, image_after = load_batch(dataset_dir, dataset_file_name, batch_size, height, width)
    save_path = './check_dump'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for i in range(2):
            image_, image_after_ = sess.run([image, image_after])
            for j in range(batch_size):
                imsave(os.path.join(save_path, '%03d_%02d_input.png'%(i,j)), image_[j])
                imsave(os.path.join(save_path, '%03d_%02d_output.png'%(i,j)), image_after_[j])
        coord.request_stop()
        coord.join(threads)
