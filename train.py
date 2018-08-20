import os
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
app = tf.app
flags = tf.flags
gfile = tf.gfile

import net
from tf_utils import *
import data_provider

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_integer(
    'patch_size', 64, 'The height/width of images in each batch.')

flags.DEFINE_string('train_log_dir', './logs_tmp/',
                    'Directory where to write training.')
flags.DEFINE_string('dataset_dir', '/home/cpjp/lyj/Downloads/Sony/preprocessed', '')
flags.DEFINE_string('dataset_file_name', './tmp_test.txt','')
flags.DEFINE_float('learning_rate', .0001, 'The learning rate')

flags.DEFINE_integer('max_number_of_steps', 100000000,
                     'The maximum number of gradient steps.')
flags.DEFINE_integer('final_K', 1, 'size of filter')
flags.DEFINE_integer('final_W', 3, 'size of output channel')
flags.DEFINE_integer('save_iter', 500, 'save iter inter')
flags.DEFINE_integer('angular_loss_weight', 1, 'weight for angular_loss')
flags.DEFINE_integer('img_0_loss_weight', 1, 'weight for angular_loss')
flags.DEFINE_integer('img_1_loss_weight', 0, 'weight for angular_loss')

FLAGS = flags.FLAGS



def train(FLAGS):
    batch_size = FLAGS.batch_size
    height = width = FLAGS.patch_size
    final_W = FLAGS.final_W
    final_K = FLAGS.final_K
    dataset_dir = os.path.join(FLAGS.dataset_dir)
    dataset_file_name = FLAGS.dataset_file_name
    input_image, gt_image, gt_gain = data_provider.load_batch(dataset_dir, dataset_file_name,
                                                     batch_size, height, width, channel = final_W, with_gain = True)

    with tf.variable_scope('generator'):
        # filters = net.convolve_net(input_image, final_K, final_W, ch0=64, N=4, D=3,
        #               scope='get_filted', separable=False, bonus=False)
        filts, filts_gain = net.convolve_net_v1(input_image, final_K, final_W, ch0=64, N=2, D=3, scope='get_filted')
    gs = tf.Variable(0, name='global_step', trainable=False)

    predict_image_0 = convolve(input_image, filts, final_K, final_W)
    predict_image_1 = sep_convolve(input_image, filts_gain, final_K, final_W)

    # compute loss
    losses = []
    # pixel_wise loss
    # presh = tf.shape(predict_image)
    # gtsh = tf.shape(gt_image)
    # predict_image = tf.cond(tf.less(presh[1], gtsh[1]),
    #             lambda: tf.image.resize_images(predict_image, [gtsh[1], gtsh[2]], method = tf.image.ResizeMethod.BILINEAR),
    #             lambda: predict_image)
    # print ('predict_image shape: ', predict_image.shape)
    predict_image_0_srgb = sRGBforward(predict_image_0)
    predict_image_1_srgb = sRGBforward(predict_image_1)
    gt_image_srgb = sRGBforward(gt_image)
    img_loss_0 = FLAGS.img_0_loss_weight * basic_img_loss(gt_image_srgb, predict_image_0_srgb)
    img_loss_1 = FLAGS.img_1_loss_weight * basic_img_loss(gt_image_srgb, predict_image_1_srgb)
    losses.append(img_loss_0)
    losses.append(img_loss_1)
    # arcos loss
    est_gain = get_gain_from_filter(filts_gain, final_W)
    # angular_loss = FLAGS.angular_loss_weight * get_angular_loss(est_gain, gt_gain)
    angular_loss = FLAGS.angular_loss_weight * tf.reduce_mean(tf.square(est_gain - gt_gain))
    losses.append(angular_loss)
    slim.losses.add_loss(tf.reduce_sum(tf.stack(losses)))
    total_loss = slim.losses.get_total_loss()

    # summaies
    filter_r_sum = tf.summary.image('filter_r', tf.expand_dims(filts_gain[...,0], axis = -1))
    filter_g_sum = tf.summary.image('filter_g', tf.expand_dims(filts_gain[...,1], axis = -1))
    filter_b_sum = tf.summary.image('filter_b', tf.expand_dims(filts_gain[...,2], axis = -1))
    input_image_sum = tf.summary.image('input_image', input_image)
    gt_image_sum = tf.summary.image('gt_image', gt_image)
    predict_image_sum = tf.summary.image('predict_image', predict_image_0)
    total_loss_sum = tf.summary.scalar('total_loss', total_loss)
    angular_loss_sum = tf.summary.scalar('angular_loss', angular_loss)
    img_loss_0_sum = tf.summary.scalar('img_loss_0', img_loss_0)
    img_loss_1_sum = tf.summary.scalar('img_loss_1', img_loss_1)

    sum_total = tf.summary.merge_all()

    # optimizer
    g_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step_g = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(
            total_loss, global_step=gs, var_list=g_vars)

    max_steps = FLAGS.max_number_of_steps

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:

        writer = tf.summary.FileWriter(FLAGS.train_log_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(max_to_keep=None)

        print ('Initializers variables')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        ckpt_path = tf.train.latest_checkpoint(FLAGS.train_log_dir)
        if ckpt_path is not None:
            print ('Restoring from', ckpt_path)
            saver.restore(sess, ckpt_path)


        for i_step in range(max_steps):
            _, loss, i, sum_total_ = sess.run([train_step_g, total_loss, gs, sum_total])
            if i_step % 1 == 0:
                print ('Step', i, 'loss =', loss)

            if i % FLAGS.save_iter == 0:
                print ('Saving ckpt at step', i)
                saver.save(sess, FLAGS.train_log_dir + 'model.ckpt', global_step=i)

            if i % 20 == 0:
                writer.add_summary(sum_total_, i)
                print ('summary saved')

        coord.request_stop()
        coord.join(threads)


def main(_):
    if not gfile.Exists(FLAGS.train_log_dir):
        gfile.MakeDirs(FLAGS.train_log_dir)

    train(FLAGS)


if __name__ == '__main__':
    app.run()
