import os
import numpy as np
from scipy.misc import imsave
import tensorflow as tf
slim = tf.contrib.slim
app = tf.app
flags = tf.flags
gfile = tf.gfile

import net
from tf_utils import *
import utils
import data_provider

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_integer(
    'patch_size', 128, 'The height/width of images in each batch.')

flags.DEFINE_string('ckpt_path', './logs_tmp/',
                    'Directory where to write training.')
flags.DEFINE_string('save_dir', './save_dir', 'Directoru to save test results')
flags.DEFINE_string('dataset_dir', '/home/cpjp/lyj/Downloads/Sony/preprocessed', '')
flags.DEFINE_string('dataset_file_name', './tmp_test.txt','')
flags.DEFINE_integer('final_K', 5, 'size of filter')
flags.DEFINE_integer('final_W', 3, 'size of output channel')

flags.DEFINE_integer('total_test_num', 50, 'num of test file')
flags.DEFINE_boolean('write_sum', False, 'if write summay in test mode')
FLAGS = flags.FLAGS


def test(FLAGS):
    batch_size = FLAGS.batch_size
    height = width = FLAGS.patch_size
    final_W = FLAGS.final_W
    final_K = FLAGS.final_K
    dataset_dir = os.path.join(FLAGS.dataset_dir)
    dataset_file_name = FLAGS.dataset_file_name
    input_image, gt_image, label = data_provider.load_batch(dataset_dir, dataset_file_name,
                                                     batch_size, height, width, channel = final_W,
                                                     shuffle = False, use_ms = False, with_gain = True)

    with tf.variable_scope('generator'):
        filters = net.convolve_net(input_image, final_K, final_W, ch0=64, N=3, D=3,
                      scope='get_filted', separable=False, bonus=False)
    predict_image = net.convolve(input_image, filters, final_K, final_W)

    # summaies
    # filters_sum = tf.summary.image('filters', filters)
    # input_image_sum = tf.summary.image('input_image', input_image)
    # gt_image_sum = tf.summary.image('gt_image', gt_image)
    # predict_image_sum = tf.summary.image('predict_image', predict_image)

    sum_total = tf.summary.merge_all()

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:

        if FLAGS.write_sum:
            writer = tf.summary.FileWriter(FLAGS.save_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        restorer = tf.train.Saver(max_to_keep=None)

        print ('Initializers variables')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)
        if ckpt_path is not None:
            print ('Restoring from', ckpt_path)
            restorer.restore(sess, ckpt_path)

        max_steps = FLAGS.total_test_num // batch_size
        for i_step in range(max_steps):
            input_image_, gt_image_, predict_image_, filters_, label_, sum_total_ = \
                    sess.run([input_image, gt_image, predict_image, filters, label, sum_total])
            concat = utils.get_concat(input_image_, gt_image_, predict_image_)
            for batch_i in range(batch_size):
                imsave(os.path.join(FLAGS.save_dir, '%03d_%02d.png'%(i_step,batch_i)), concat[batch_i]*255.0 )
                print ('saved once')
            if FLAGS.write_sum and i_step % 20 == 0:
                writer.add_summary(sum_total_, i)
                print ('summary saved')

        coord.request_stop()
        coord.join(threads)


def main(_):
    if not gfile.Exists(FLAGS.save_dir):
        gfile.MakeDirs(FLAGS.save_dir)

    test(FLAGS)


if __name__ == '__main__':
    app.run()
