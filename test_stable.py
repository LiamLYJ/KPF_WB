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

flags.DEFINE_integer('batch_size', 10, 'The number of images in each batch.')

flags.DEFINE_integer(
    'patch_size', 128, 'The height/width of images in each batch.')
    # 'patch_size', 64, 'The height/width of images in each batch.')

# flags.DEFINE_string('ckpt_path', './logs_sony_ex/',
# flags.DEFINE_string('ckpt_path', './logs_sony_ex_64/',
flags.DEFINE_string('ckpt_path', './logs_nus_ex/',
                    'Directory where to write training.')
flags.DEFINE_string('save_dir', './dump_nus_test_ms', 'Directoru to save test results')
# flags.DEFINE_string('save_dir', './tmp_test', 'Directoru to save test results')
# flags.DEFINE_string('dataset_dir', './data/sony', '')
flags.DEFINE_string('dataset_dir', './data/nus', '')
flags.DEFINE_string('dataset_file_name', './data_txt_file/NUS_val.txt','')
flags.DEFINE_integer('final_K', 5, 'size of filter')
# flags.DEFINE_integer('final_K', 1, 'size of filter')
flags.DEFINE_integer('final_W', 3, 'size of output channel')
flags.DEFINE_boolean('shuffle', True, 'if shuffle')
flags.DEFINE_integer('total_test_num', 100, 'num of test file')
flags.DEFINE_boolean('use_ms', True, 'if use multi_source trianing')
flags.DEFINE_boolean('use_crop', False, 'if check crop')
flags.DEFINE_boolean('use_clip', False, 'if check clip')
flags.DEFINE_boolean('use_flip', False, 'if check flip')
flags.DEFINE_boolean('use_rotate', False, 'if check rotate')
flags.DEFINE_boolean('use_noise', False, 'if check noise')
flags.DEFINE_boolean('save_clus', False, 'if save clus')
flags.DEFINE_boolean('save_filt', False, 'if save filt')

FLAGS = flags.FLAGS


def test(FLAGS):
    batch_size = FLAGS.batch_size
    height = width = FLAGS.patch_size
    final_W = FLAGS.final_W
    final_K = FLAGS.final_K
    dataset_dir = os.path.join(FLAGS.dataset_dir)
    dataset_file_name = FLAGS.dataset_file_name
    shuffle = FLAGS.shuffle
    input_image = tf.placeholder(tf.float32, shape = (None,height, width,3))

    with tf.variable_scope('generator'):
        if FLAGS.patch_size == 128:
            N_size = 3
        else:
            N_size = 2
        filters = net.convolve_net(input_image, final_K, final_W, ch0=64,
                                   N=N_size, D=3,
                      scope='get_filted', separable=False, bonus=False)
    predict_image = convolve(input_image, filters, final_K, final_W)

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:

        print ('Initializers variables')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

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
                                        data_txt=dataset_file_name, patch_size=FLAGS.patch_size, start_index=i_step*batch_size, batch_size=batch_size,
                                        use_ms = True)
            else:
                imgs, imgs_gt, labels, file_names = utils.data_loader_np(data_folder = dataset_dir,
                                        data_txt=dataset_file_name, patch_size=FLAGS.patch_size, start_index=i_step*batch_size, batch_size=batch_size,
                                        use_ms = False)
            input_image_ = utils.batch_stable_process(imgs, use_crop=FLAGS.use_crop,
                    use_clip=FLAGS.use_clip, use_flip=FLAGS.use_flip, use_rotate=FLAGS.use_rotate, use_noise=FLAGS.use_noise)
            gt_image_ = imgs_gt
            predict_image_, filters_ = sess.run([predict_image, filters], feed_dict={input_image: input_image_})
            # [batch, h ,w]
            batch_confidence_r = utils.compute_rate_confidence(filters_, input_image_, final_K, final_W, sel_ch = 0, ref_ch = [2], is_spatial = True)
            batch_confidence_b = utils.compute_rate_confidence(filters_, input_image_, final_K, final_W, sel_ch = 2, ref_ch = [0], is_spatial = True)

            concat = utils.get_concat(input_image_, gt_image_, predict_image_)
            num_filt = (FLAGS.final_K **2) * (FLAGS.final_W **2)
            for batch_i in range(batch_size):
                est_global = utils.solve_gain(input_image_[batch_i], np.clip(predict_image_[batch_i], 0, 500))

                if FLAGS.use_ms:
                    save_file_name = '%s_%s.png'%(file_names[batch_i][0][:-4],
                                                file_names[batch_i][1][:-4])
                else:
                    print ('confidence_r: ', np.mean(batch_confidence_r[batch_i]))
                    print ('confidence_b: ', np.mean(batch_confidence_b[batch_i]))
                    current_file_name = file_names[batch_i]
                    print (' {} saved once'.format(current_file_name))
                    gt = labels[batch_i]
                    error = utils.angular_error(est, gt)
                    print ('est is ; ', est)
                    print ('gt is ; ', gt)
                    print ('error is ; ', error)
                    errors.append(error)
                    save_file_name = current_file_name

                est_global_img_ = np.clip(input_image_[batch_i] * est_global, 0, 255.0) / 255.0
                all_concat = np.concatenate([concat[batch_i], est_global_img_], axis = 1)
                if FLAGS.save_dir is not None:
                    imsave(os.path.join(FLAGS.save_dir, save_file_name), all_concat*255.0 )
                    np_concat = np.concatenate([input_image_[batch_i], predict_image_[batch_i]], axis = 0)
                    file_name_np = os.path.join(FLAGS.save_dir, save_file_name[:-3] + 'npy')
                    np.save(file_name_np, np_concat)
                    if FLAGS.use_ms:
                        if FLAGS.save_clus:
                            print ('local gain fitting', save_file_name)
                            gain_box, clus_img, clus_labels = utils.gain_fitting(input_image_[batch_i], predict_image_[batch_i], is_local = True, n_clusters =2, gamma = 4.0, with_clus = True)
                            num_multi = len(set(clus_labels))
                            for index_ill in range(num_multi):
                                confi_multi_r = utils.get_confi_multi(clus_labels, batch_confidence_r[batch_i], label = index_ill)
                                confi_multi_b = utils.get_confi_multi(clus_labels, batch_confidence_b[batch_i], label = index_ill)
                                print ('confidence_r for ill %d'%index_ill, confi_multi_r)
                                print ('confidence_b for ill %d'%index_ill, confi_multi_b)
                            imsave(os.path.join(FLAGS.save_dir, '%s_clus.png'%(save_file_name[:-4])), clus_img)
                        if FLAGS.save_filt:
                            cur_filt = filters_[batch_i]
                            for filt_index in range(num_filt):
                                cur_ = cur_filt[..., filt_index]
                                imsave(os.path.join(FLAGS.save_dir, '%s_filt_%d.png'%(save_file_name[:-4], filt_index)), cur_)
                        file_name_json = os.path.join(FLAGS.save_dir, save_file_name[:-3] + 'json')
                        save_dict = configs[batch_i]
                        with open(file_name_json, 'w') as fp:
                            json.dump(save_dict, fp, ensure_ascii=False)
                # np.save(os.path.join(FLAGS.save_dir,'%03d_%02d.npy'%(i_step,batch_i)), predict_image_[batch_i])
    if errors:
        utils.print_angular_errors(errors)

def main(_):
    if FLAGS.save_dir is not None:
        if not gfile.Exists(FLAGS.save_dir):
            gfile.MakeDirs(FLAGS.save_dir)

    test(FLAGS)


if __name__ == '__main__':
    app.run()
