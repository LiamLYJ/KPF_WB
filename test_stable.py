import os
import numpy as np
from scipy.misc import imsave
import tensorflow as tf
slim = tf.contrib.slim
app = tf.app
flags = tf.flags
gfile = tf.gfile
import json

import net
from tf_utils import *
import utils
import data_provider

flags.DEFINE_integer('batch_size', 20, 'The number of images in each batch.')

flags.DEFINE_integer(
    'patch_size', 128, 'The height/width of images in each batch.')

flags.DEFINE_string('ckpt_path', './logs_nus/',
                    'Directory where to write training.')
flags.DEFINE_string('save_dir', None, 'Directoru to save test results')
flags.DEFINE_string('dataset_dir', './data/nus', '')
flags.DEFINE_string('dataset_file_name', './data_txt_file/NUS_val.txt','')
flags.DEFINE_integer('final_K', 5, 'size of filter')
flags.DEFINE_integer('final_W', 3, 'size of output channel')
flags.DEFINE_boolean('shuffle', False, 'if shuffle')
flags.DEFINE_integer('total_test_num', 200, 'num of test file')
flags.DEFINE_boolean('use_ms', False, 'if use multi_source trianing')
flags.DEFINE_boolean('use_crop', False, 'if check crop')
flags.DEFINE_boolean('use_clip', False, 'if check clip')
flags.DEFINE_boolean('use_flip', False, 'if check flip')
flags.DEFINE_boolean('use_rotate', False, 'if check rotate')
flags.DEFINE_boolean('use_noise', False, 'if check noise')
FLAGS = flags.FLAGS


def test(FLAGS):
    batch_size = FLAGS.batch_size
    height = width = FLAGS.patch_size
    final_W = FLAGS.final_W
    final_K = FLAGS.final_K
    dataset_dir = os.path.join(FLAGS.dataset_dir)
    dataset_file_name = FLAGS.dataset_file_name
    shuffle = FLAGAS.shuffle
    input_image = tf.placeholder(tf.float32, shape = (None,height, width,3))

    with tf.variable_scope('generator'):
        if FLAGS.patch_size == 128:
            N_size = 3
        else:
            N_size = 2
        filters = net.convolve_net(input_image, final_K, final_W, ch0=64,
                                   N=N_size, D=3,
                      scope='get_filted', separable=False, bonus=False)
    predict_image = net.convolve(input_image, filters, final_K, final_W)

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:

        print ('Initializers variables')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if FLAGS.write_sum:
            writer = tf.summary.FileWriter(FLAGS.save_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        restorer = tf.train.Saver(max_to_keep=None)

        ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)
        if ckpt_path is not None:
            print ('Restoring from', ckpt_path)
            restorer.restore(sess, ckpt_path)

        errors = []

        max_steps = FLAGS.total_test_num // batch_size
        for i_step in range(max_steps):
            if FLAGS.use_ms:
                imgs, imgs_gt, labels, file_names, configs = utils.data_loader_np(data_folder = dataset_dir,
                                        data_txt=dataset_file_name, start_index=i_step*batch_size, batch_size=batch_size,
                                        use_ms = True)
            else:
                imgs, imgs_gt, labels, labels, file_names = utils.data_loader_np(data_folder = dataset_dir,
                                        data_txt=dataset_file_name, start_index=i_step*batch_size, batch_size=batch_size,
                                        use_ms = False)
            input_image_ = utils.batch_stable_process(imgs, use_crop=FLAGS.use_crop,
                    use_clip=FLAGS.use_clip, use_flip=FLAGS.use_flip, use_rotate=FLAGS.use_rotate, use_noise=FLAGS.use_noise)
            gt_image_ = imgs_gt
            predict_image_, filters_ = sess.run([predict_image, filters], feed_dict={input_image: input_image_})
            batch_confidence_r = utils.compute_rate_confidence(filters_, input_image_, vinal_K, final_W, sel_ch = 0, ref_ch = [2])
            batch_confidence_b = utils.compute_rate_confidence(filters_, input_image_, final_K, final_W, sel_ch = 2, ref_ch = [0])

            concat = utils.get_concat(input_image_, gt_image_, predict_image_)
            for batch_i in range(batch_size):
                est = utils.solve_gain(input_image_[batch_i], predict_image_[batch_i])
                print ('confidence_r: ', batch_confidence_r[batch_i])
                print ('confidence_b: ', batch_confidence_b[batch_i])

                if FLAGS.use_ms:
                    save_file_name = '%s_%s.png'%(file_names[batch_i][0].decode('utf-8').split('/')[-1],
                                                file_names[batch_i][1].decode('utf-8').split('/')[-1])
                else:
                    current_file_name = file_name_[batch_i][0].decode('utf-8').split('/')[-1]
                    print (' {} saved once'.format(current_file_name))
                    gt = label_[batch_i]
                    error = utils.angular_error(est, gt)
                    print ('est is ; ', est)
                    print ('gt is ; ', gt)
                    print ('error is ; ', error)
                    errors.append(error)
                    save_file_name = current_file_name

                est_img_ = np.clip(input_image_[batch_i] * est, 0, 255.0) / 255.0
                all_concat = np.concatenate([concat[batch_i], est_img_], axis = 1)
                if FLAGS.save_dir is not None:
                    imsave(os.path.join(FLAGS.save_dir, save_file_name), all_concat*255.0 )
                    if FLAGS.use_ms:
                        file_name_json = save_file_name[:-3] + 'json'
                        save_dict = configs[batch_i]
                        save_dict['file1'] = file_names[batch_i][0].decode('utf-8').split('/')[-1]
                        save_dict['file2'] = file_names[batch_i][1].decode('utf-8').split('/')[-1]
                        with open(file_names_json, 'w') as fp:
                            json.dump(save_dict, fp, ensure_ascii=False)
                # np.save(os.path.join(FLAGS.save_dir,'%03d_%02d.npy'%(i_step,batch_i)), predict_image_[batch_i])

        coord.request_stop()
        coord.join(threads)
    if errors:
        utils.print_angular_errors(errors)

def main(_):
    if FLAGS.save_dir is not None:
        if not gfile.Exists(FLAGS.save_dir):
            gfile.MakeDirs(FLAGS.save_dir)

    test(FLAGS)


if __name__ == '__main__':
    app.run()