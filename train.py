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

flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_integer(
    'patch_size', 128, 'The height/width of images in each batch.')

flags.DEFINE_string('train_log_dir', './logs_sony_ex/',
                    'Directory where to write training.')
flags.DEFINE_string('dataset_dir', './data/sony/', '')
flags.DEFINE_string('dataset_file_name_train', './data_txt_file/file_train.txt','train_files')
flags.DEFINE_string('dataset_file_name_val', './data_txt_file/file_val.txt','val_files')
flags.DEFINE_float('learning_rate', .0001, 'The learning rate')

flags.DEFINE_integer('max_number_of_steps', 100000000,
                     'The maximum number of gradient steps.')
flags.DEFINE_integer('final_K', 5, 'size of filter')
# flags.DEFINE_integer('final_K', 1, 'size of filter')
flags.DEFINE_integer('final_W', 3, 'size of output channel')
flags.DEFINE_integer('input_ch', 3, 'size of input channel')
flags.DEFINE_integer('save_iter', 500, 'save iter inter')
flags.DEFINE_integer('sum_iter', 5, 'sum iter inter')
flags.DEFINE_float('img_loss_weight', 1.0, 'weight for img_loss')
flags.DEFINE_float('filts_reg_weight', 0.001, 'weight for filts regularation')
flags.DEFINE_boolean('use_ms', True, 'if use multi_source trianing')
flags.DEFINE_boolean('h_flip', True, 'horizontal fip')
flags.DEFINE_boolean('v_flip', True, 'vertical fip')
flags.DEFINE_float('rotate', 60.0, 'rotate angle')
flags.DEFINE_float('crop_prob', 0.5, 'crop_probability')
flags.DEFINE_float('crop_min_percent', 0.3, 'crop min percent' )
flags.DEFINE_float('crop_max_percent', 1.0, 'crop max percent' )
flags.DEFINE_float('aug_color', 0.0, 'color aug' )
flags.DEFINE_float('aug_color_offdiag', 0.0, 'aug color offdiag' )
flags.DEFINE_float('mixup', 0.0, 'mix up for data augmentation')


FLAGS = flags.FLAGS

def train(FLAGS):
    batch_size = FLAGS.batch_size
    height = width = FLAGS.patch_size
    final_W = FLAGS.final_W
    final_K = FLAGS.final_K
    input_ch = FLAGS.input_ch
    dataset_dir = os.path.join(FLAGS.dataset_dir)
    dataset_file_name_train = FLAGS.dataset_file_name_train
    dataset_file_name_val = FLAGS.dataset_file_name_val

    input_image, gt_image = data_provider.load_batch(dataset_dir, dataset_file_name_train, batch_size, height, width,
                                    channel = input_ch, use_ms = FLAGS.use_ms,
                                    aug_color = FLAGS.aug_color, aug_color_offdiag = FLAGS.aug_color_offdiag )
    input_image_val, gt_image_val = data_provider.load_batch(dataset_dir, dataset_file_name_val, batch_size, height, width,
                                    channel = input_ch, use_ms = FLAGS.use_ms )

    image_concat = tf.concat([input_image, gt_image], axis = -1)
    image_concat = data_augment(image_concat, horizontal_flip=FLAGS.h_flip, vertical_flip=FLAGS.v_flip, rotate=FLAGS.rotate,
                                crop_probability=FLAGS.crop_prob, crop_min_percent=FLAGS.crop_min_percent, crop_max_percent=FLAGS.crop_max_percent, mixup=0)
    input_image, gt_image = tf.split(image_concat, [3,3], axis = -1)

    with tf.variable_scope('generator'):
        if FLAGS.patch_size == 128:
            N_size = 3
        else:
            N_size = 2
        filts = net.convolve_net(input_image, final_K, final_W, ch0=64,
                                 N=N_size, D=3,
                      scope='get_filted', separable=False, bonus=False)
    gs = tf.Variable(0, name='global_step', trainable=False)

    predict_image = convolve(input_image, filts, final_K, final_W)

    # compute loss
    losses = []
    # pixel_wise loss
    # presh = tf.shape(predict_image)
    # gtsh = tf.shape(gt_image)
    # predict_image = tf.cond(tf.less(presh[1], gtsh[1]),
    #             lambda: tf.image.resize_images(predict_image, [gtsh[1], gtsh[2]], method = tf.image.ResizeMethod.BILINEAR),
    #             lambda: predict_image)
    # print ('predict_image shape: ', predict_image.shape)
    predict_image_srgb = sRGBforward(predict_image)
    gt_image_srgb = sRGBforward(gt_image)
    img_loss = FLAGS.img_loss_weight * basic_img_loss(gt_image_srgb, predict_image_srgb)
    losses.append(img_loss)
    filts_loss = FLAGS.filts_reg_weight * filts_reg_loss(filts, input_ch=input_ch, final_K=final_K, final_W=final_W)
    losses.append(filts_loss)
    slim.losses.add_loss(tf.reduce_sum(tf.stack(losses)))
    total_loss = slim.losses.get_total_loss()

    # check val loss
    predict_image_val = convolve(input_image_val, filts, final_K, final_W)
    predict_image_val_srgb = sRGBforward(predict_image_val)
    gt_image_val_srgb = sRGBforward(gt_image_val)
    val_loss = basic_img_loss(gt_image_val_srgb, predict_image_val_srgb)

    # summaies
    filts_sh = tf.shape(filts)
    # filts_for_sum = tf.reshape(filts, [filts_sh[0], filts[1], filts[2], -1, final_W])
    filts_for_sum = tf.reshape(filts, [batch_size, height, width, final_K * final_K, input_ch, final_W])
    filts_r_r_sum = tf.summary.image('filts_r_r', tf.expand_dims(filts_for_sum[...,0,0,0], axis = -1))
    filts_r_g_sum = tf.summary.image('filts_r_g', tf.expand_dims(filts_for_sum[...,0,1,0], axis = -1))
    filts_r_b_sum = tf.summary.image('filts_r_b', tf.expand_dims(filts_for_sum[...,0,2,0], axis = -1))
    filts_g_r_sum = tf.summary.image('filts_g_r', tf.expand_dims(filts_for_sum[...,0,0,1], axis = -1))
    filts_g_g_sum = tf.summary.image('filts_g_g', tf.expand_dims(filts_for_sum[...,0,1,1], axis = -1))
    filts_g_b_sum = tf.summary.image('filts_g_b', tf.expand_dims(filts_for_sum[...,0,2,1], axis = -1))
    filts_b_r_sum = tf.summary.image('filts_b_r', tf.expand_dims(filts_for_sum[...,0,0,2], axis = -1))
    filts_b_g_sum = tf.summary.image('filts_b_g', tf.expand_dims(filts_for_sum[...,0,1,2], axis = -1))
    filts_b_b_sum = tf.summary.image('filts_b_b', tf.expand_dims(filts_for_sum[...,0,2,2], axis = -1))
    input_image_sum = tf.summary.image('input_image', input_image)
    gt_image_sum = tf.summary.image('gt_image', gt_image)
    predict_image_sum = tf.summary.image('predict_image', predict_image)
    total_loss_sum = tf.summary.scalar('total_loss', total_loss)
    img_loss_sum = tf.summary.scalar('img_loss', img_loss)
    filts_loss_sum = tf.summary.scalar('filts_loss', filts_loss)

    sum_total = tf.summary.merge_all()
    sum_val = tf.summary.scalar('val_loss', val_loss)

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

        print ('Initializers variables')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        writer_train = tf.summary.FileWriter(os.path.join(FLAGS.train_log_dir,'train'), sess.graph)
        writer_val = tf.summary.FileWriter(os.path.join(FLAGS.train_log_dir,'val'), sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(max_to_keep=None)

        ckpt_path = tf.train.latest_checkpoint(FLAGS.train_log_dir)
        if ckpt_path is not None:
            print ('Restoring from', ckpt_path)
            saver.restore(sess, ckpt_path)


        for i_step in range(max_steps):
            _, loss, i, sum_total_ = sess.run([train_step_g, total_loss, gs, sum_total])
            if i_step % 5 == 0:
                print ('Step', i, 'loss =', loss)

            if i % FLAGS.save_iter == 0:
                print ('Saving ckpt at step', i)
                saver.save(sess, FLAGS.train_log_dir + 'model.ckpt', global_step=i)
                sum_val_ = sess.run(sum_val)
                writer_val.add_summary(sum_val_, i)

            if i % FLAGS.sum_iter == 0:
                writer_train.add_summary(sum_total_, i)
                print ('summary saved')

        coord.request_stop()
        coord.join(threads)


def main(_):
    if not gfile.Exists(FLAGS.train_log_dir):
        gfile.MakeDirs(FLAGS.train_log_dir)
    if not gfile.Exists(os.path.join(FLAGS.train_log_dir, 'train')):
        gfile.MakeDirs(os.path.join(FLAGS.train_log_dir, 'train'))
    if not gfile.Exists(os.path.join(FLAGS.train_log_dir, 'val')):
        gfile.MakeDirs(os.path.join(FLAGS.train_log_dir, 'val'))

    train(FLAGS)


if __name__ == '__main__':
    app.run()
