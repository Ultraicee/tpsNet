# Code for
# Reconstructing Dynamic Soft Tissue with Stereo Endoscope Based on a Single-layer Network
# Bo Yang, Siyuan Xu
#
# parts of the code from https://github.com/SiyuanXuu/tpsNet

from __future__ import absolute_import, division, print_function
from decoder import *
from read_images import *
from linear_sample import *
import argparse
# only keep warnings and errors
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import copy
import time
import tensorflow as tf

parser = argparse.ArgumentParser(description='alternative train TPS and test')

parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--data_size', type=int, help='num of total train images', default=200)
parser.add_argument('--test_num', type=int, help='num of total test images', default=100)
parser.add_argument('--batch_size', type=int, help='num of minimum batch size', default=10)
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=3e-1)
parser.add_argument('--epoch_num', type=int, help='num of epochs', default=10)
parser.add_argument('--max_step', type=int, help='max training step per epoch ', default=100)
parser.add_argument('--continue_predict', type=bool, help='use last step feature or not', default=False)
parser.add_argument('--output_directory', type=str,
                    help='output directory for test disparities, if empty outputs to checkpoint folder',
                    default='output/')
parser.add_argument('--source_directory', type=str, help='directory to load source images',
                    default='dataset/invivo1_rect/')
parser.add_argument('--A_directory', type=str, help='directory to load A matrix',
                    default='output/A_val.txt')
parser.add_argument('--input_height', type=int, help='input height', default=288)
parser.add_argument('--input_width', type=int, help='input width', default=360)
parser.add_argument('--input_channel', type=int, help='input channel', default=3)
parser.add_argument('--crop_top', type=int, help='crop pos[top,bottom,left,right]', default=54)
parser.add_argument('--crop_bottom', type=int, help='crop pos[top,bottom,left,right]', default=34)
parser.add_argument('--crop_left', type=int, help='crop pos[top,bottom,left,right]', default=42)
parser.add_argument('--crop_right', type=int, help='crop pos[top,bottom,left,right]', default=118)
parser.add_argument('--cpts_row', type=int, help='row of control points', default=4)
parser.add_argument('--cpts_col', type=int, help='col of control points', default=4)


def train_feature(params, feature_in, left_ims, right_ims, tps_matrix, linear_interpolator):

    print_str = 'Step{:4} | recons loss={:4} | norm loss={:4} | total loss={:4}' \
                '  | feature_var_mean={:4}'

    learning_rate_init = np.float32(params.learning_rate)

    optimize_op1 = tf.train.AdamOptimizer(learning_rate_init)  # train z alternatively
    optimize_op2 = tf.train.AdamOptimizer(2e-3)  # train A alternatively

    tps_mat = tps_matrix
    feature_next = feature_in
    # build tf graph
    with tf.variable_scope(tf.get_variable_scope()):

        left = tf.constant(left_ims[:params.batch_size],tf.float32,name='left')
        right = tf.constant(right_ims[:params.batch_size],tf.float32,name='right')
        feature_f = tf.constant(feature_in, dtype=tf.float32, name='feature')
        tps_weight_TRUE = tf.constant(tps_matrix, dtype=tf.float32, name='tps_weight_tg')
        tps_weight_f = tf.constant(tps_matrix, dtype=tf.float32, name='tps_weight_f')
        compensateI = tf.Variable(4.3, dtype=tf.float32, name='contr_val')  # light compensation
        feature_in_base = tf.placeholder(tf.float32, shape=(params.batch_size, params.cpts_row * params.cpts_col, 1),
                                         name='feature_base')
        disp_base = tf.placeholder(tf.float32, shape=(params.batch_size, params.input_height, params.input_width, 1),
                                   name='disp_base')

        feature_input = tf.Variable(feature_f, dtype=tf.float32)
        tps_weight = tf.Variable(tps_weight_f, dtype=tf.float32)
        update1 = tf.assign(feature_input, feature_f)
        update2 = tf.assign(tps_weight, tps_weight_f)

        disp2 = decoder_forward2(feature_input, feature_in_base, tps_weight, disp_base,
                                 linear_interpolator.sz_params)  # calculate disparity of alternative training
        right_est2 = linear_interpolator.interpolate(left, disp2)  # generate right image of alternative training

        loss_rec2, _, compa_sum2, loss_A_smooth2 = compute_laplace_rec_loss(
            right_est2, right, compensateI,linear_interpolator.sz_params,tps_weight_TRUE,tps_weight, 180.0, 0.26, mode='0')
        loss2 = tf.add(loss_rec2, loss_A_smooth2, name='Total_loss2')
        train_op1 = optimize_op1.minimize(loss_rec2, var_list=feature_input) # for test mode
        if params.mode == 'train':
            loss_rec3, loss_rec3_img, compa_sum3, loss_A_smooth3 = compute_laplace_rec_loss(
                right_est2, right,compensateI,linear_interpolator.sz_params,tps_weight_TRUE,tps_weight, 180.0, 0.26,mode='1')

            # make sure whether points far from excepted region
            disp_size = get_tps_size(params)
            disp_True = tf.zeros([params.batch_size, params.input_height, params.input_width, 1], dtype=tf.float32)
            disp_vec1 = tf.map_fn(lambda x: tf.matmul(tps_weight, x), feature_input)  # AZ
            disp_vec2 = tf.slice(disp_True, [0, params.crop_top, params.crop_left, 0],
                                 [-1, disp_size[0], disp_size[1], -1])
            disp_vec2 = tf.reshape(disp_vec2, [params.batch_size, disp_size[0] * disp_size[1], 1])
            loss_wt_norm = 0.017 * tf.reduce_mean(tf.reduce_mean(tf.square(disp_vec1 - disp_vec2), axis=[1, 2]))
            loss3 = loss_rec2 + loss_A_smooth2 + loss_wt_norm
            loss4 = loss_rec3

            # setting of train optimizer
            train_op1 = optimize_op1.minimize(loss4, var_list=feature_input)
            train_op2 = optimize_op2.minimize(loss3, var_list=tps_weight)

        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            loss_rec_temp = 0.
            feature_before = feature_in
            feature_val = 0.
            disp_val = np.zeros([params.batch_size, params.input_height, params.input_width, 1])
            est_right_val = np.zeros([params.batch_size, params.input_height, params.input_width, params.input_channel])

            train_dd = np.load(
                os.path.join(params.output_directory, 'disp_batch_invivo_d.npy'))  # value of d of trained 200 images
            train_zz = np.load(
                os.path.join(params.output_directory, 'disp_batch_invivo_z.npy'))  # value of z of trained 200 images

            feature_base_ = np.load(os.path.join(params.output_directory, 'z_batch_invivo_mean.npy'))
            feature_base_ = np.tile(feature_base_, (params.batch_size, 1, 1))

            disp_base_val = np.load(os.path.join(params.output_directory, 'disp_batch_invivo_mean.npy'))
            disp_base_val = np.tile(disp_base_val, (params.batch_size, 1, 1, 1))
            loss_result = []  # record loss
            if params.mode == 'train':
                out_dd = train_dd.copy()
                tps_dd = train_dd.copy()
                out_zz = train_zz.copy()
                out_left = left_ims.copy()
                out_right = right_ims.copy()

                inner_step = int(params.data_size / params.batch_size)
                total_loss_result = []
                index = [i for i in range(params.data_size)]

                # train A
                tms_overfit = 0  # record time of over fit, more than 3 times to stop
                loss_rec_val_mean = 0
                loss_A_smooth_val_mean = 0
                loss_norm_val_mean = 0
                compa_sum = 0
                a_step = 0

                # shuffle training samples
                np.random.shuffle(index)
                train_dd = train_dd[index]
                train_zz = train_zz[index]
                left_ims = left_ims[index]
                right_ims = right_ims[index]

                for batch_idx in range(inner_step):
                    cur_zz = train_zz[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                    cur_dd = train_dd[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                    cur_left = left_ims[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                    cur_right = right_ims[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]

                    _, feature_val, tps_val, loss_rec_val, loss_A_smooth_val, loss_norm_val, compa_sum2_val, loss_val, disp_val, est_right_val = sess.run(
                    [train_op2, feature_input, tps_weight, loss_rec2, loss_A_smooth2, loss_wt_norm, compa_sum2,
                     loss3, disp2, right_est2],
                    feed_dict={feature_f: cur_zz, tps_weight_f: tps_mat, left: cur_left, right: cur_right,
                               feature_in_base: feature_base_, disp_base: disp_base_val, disp_True: cur_dd})
                    tps_mat = tps_val

                for batch_idx in range(inner_step):
                    cur_zz = train_zz[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                    cur_dd = train_dd[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                    cur_left = left_ims[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]
                    cur_right = right_ims[batch_idx * params.batch_size:(batch_idx + 1) * params.batch_size]

                    feature_val, tps_val, loss_rec_val, loss_A_smooth_val, loss_norm_val, compa_sum3_val, disp_val, est_right_val = sess.run(
                            [feature_input, tps_weight, loss_rec3_img, loss_A_smooth3, loss_wt_norm,compa_sum3, disp2,right_est2],
                        feed_dict={feature_f: cur_zz,
                                   tps_weight_f: tps_mat,
                                   left: cur_left,
                                   right: cur_right,
                                   feature_in_base: feature_base_,
                                   disp_base: disp_base_val,
                                   disp_True: cur_dd}
                    )

                    loss_rec_val_mean += loss_rec_val * params.cpts_row * params.cpts_col
                    loss_A_smooth_val_mean += loss_A_smooth_val / inner_step
                    loss_norm_val_mean += loss_norm_val / inner_step
                    compa_sum += compa_sum3_val

                    loss_rec_val_mean = loss_rec_val_mean / compa_sum
                    loss_val_mean = loss_rec_val_mean + loss_A_smooth_val_mean + loss_norm_val_mean

                    loss_result.append([loss_rec_val_mean, 0])  # record loss of A
                    total_loss_result.append([loss_val_mean, 0])
                    if loss_val_mean - loss_rec_temp > 0:
                        tms_overfit += 1
                    else:
                        tms_overfit = 0
                    if tms_overfit > 4:
                        break
                    loss_rec_temp = loss_val_mean
                    a_step += 1
                    if a_step > 0 and (0 == a_step % 10 or a_step + 1 == params.max_step):
                        feature_var_mean = np.mean(feature_val - feature_before)

                        print(
                            print_str.format(a_step, loss_rec_val_mean, loss_A_smooth_val_mean, loss_norm_val_mean,
                                             loss_val_mean, feature_var_mean))  # loss message of current patch
                        feature_before = feature_val
                        if a_step >= params.max_step:
                            break

                feature_val, tps_val = sess.run(
                    [feature_input, tps_weight],
                    feed_dict={feature_f: feature_next, tps_weight_f: tps_mat, left: cur_left, right: cur_right,
                               feature_in_base: feature_base_, disp_base: disp_base_val, disp_True: cur_dd})
                tps_mat = tps_val
                distance = np.mean(np.square(feature_next - feature_base_))
                print('feature distance after training:{}'.format(distance))
                # an iterative file to check trained A
                np.save(os.path.join(params.output_directory, 'temp_A.npy'), tps_mat)
                loss_result.append([1, 1])
                total_loss_result.append([1, 1])
                for j in range(inner_step):
                    loss_result.append([])
                    total_loss_result.append([])
                    cur_zz = out_zz[j * params.batch_size:(j + 1) * params.batch_size]
                    cur_dd = tps_dd[j * params.batch_size:(j + 1) * params.batch_size]
                    cur_left = out_left[j * params.batch_size:(j + 1) * params.batch_size]
                    cur_right = out_right[j * params.batch_size:(j + 1) * params.batch_size]

                    sess.run(update1, feed_dict={feature_f: cur_zz})  # update feature
                    sess.run(update2, feed_dict={tps_weight_f: tps_mat})
                print(" finished training of A.")

            # train z
            print("start train z")
            for step in range(0, params.max_step):
                if params.mode == 'train':
                    _, feature_val, tps_val, loss_rec_val, loss_A_smooth_val, loss_norm_val, compa_sum3_val, loss_val, disp_val, est_right_val = sess.run(
                        [train_op1, feature_input, tps_weight, loss_rec3_img, loss_A_smooth3, loss_wt_norm, compa_sum3,
                         loss4, disp2, right_est2],
                        feed_dict={tps_weight_f: tps_mat, left: cur_left, right: cur_right,
                                   feature_in_base: feature_base_, disp_base: disp_base_val, disp_True: cur_dd})

                    feature_next = feature_val
                    loss_result[-1].append(
                        [copy.deepcopy(loss_rec_val) * params.cpts_row * params.cpts_col, copy.deepcopy(compa_sum3_val)])
                    total_loss_result[-1].append(
                        [copy.deepcopy(loss_rec_val) * params.cpts_row * params.cpts_col, copy.deepcopy(compa_sum3_val),
                         loss_norm_val + loss_A_smooth_val])
                    if 0 == step % 5 or step + 1 == params.max_step:
                        feature_var_mean = np.mean(feature_val - feature_before)

                        print(
                            print_str.format(step, 20 * loss_rec_val / compa_sum3_val, loss_A_smooth_val, loss_norm_val,
                                             loss_val, feature_var_mean))
                        tps_before = tps_val
                        feature_before = feature_val
                        loss_rec_temp = loss_rec_val
                        if step >= 0.1*params.max_step:
                            feature_val, tps_val, loss_rec_val, loss_A_smooth_val, loss_norm_val, loss_val, disp_val, est_right_val = sess.run(
                                [feature_input, tps_weight, loss_rec3_img, loss_A_smooth3, loss_wt_norm, loss4, disp2,
                                 right_est2],
                                feed_dict={feature_f: feature_next, tps_weight_f: tps_mat, left: cur_left,
                                           right: cur_right,
                                           feature_in_base: feature_base_, disp_base: disp_base_val, disp_True: cur_dd})
                            feature_next = feature_val
                            loss_rec_temp = loss_rec_val
                            out_zz[j * params.batch_size:(j + 1) * params.batch_size] = feature_val
                            out_dd[j * params.batch_size:(j + 1) * params.batch_size] = disp_val
                            break

                    train_dd = tps_dd.copy()
                    train_zz = out_zz.copy()
                    left_ims = out_left.copy()
                    right_ims = out_right.copy()
                else:
                    _, feature_val, tps_val, loss_rec_val, loss_A_smooth_val, compa_sum2_val, loss_val, disp_val, est_right_val = sess.run(
                        [train_op1, feature_input, tps_weight, loss_rec2, loss_A_smooth2, compa_sum2, loss2, disp2,
                         right_est2],
                        feed_dict={feature_f: feature_next, tps_weight_f: tps_mat, feature_in_base: feature_base_,
                                   disp_base: disp_base_val})

                    feature_next = feature_val
                    if 0 == step % 10 or step + 1 == params.max_step:
                        loss_result.append(
                            [copy.deepcopy(loss_rec_val) * params.data_size, copy.deepcopy(compa_sum2_val)])

                        feature_var_mean = np.mean(feature_val - feature_before)
                        feature_before = feature_val

                        if step >= 0.1*params.max_step:
                            feature_val, tps_val, loss_rec_val, loss_A_smooth_val, loss_val, disp_val, est_right_val = sess.run(
                                [feature_input, tps_weight, loss_rec2, loss_A_smooth2, loss2, disp2, right_est2],
                                feed_dict={feature_f: feature_next, tps_weight_f: tps_mat,
                                           feature_in_base: feature_base_,
                                           disp_base: disp_base_val})
                            break
            feature = feature_val
            tps_mat = tps_val
            print("finished training of z")
            distance = np.mean(np.square(feature - feature_in))
            print('feature distance after training:{}'.format(distance))

    if params.mode == 'train':
        np.save(os.path.join(params.output_directory, 'alternative_loss.npy'), loss_result)
        np.save(os.path.join(params.output_directory, 'alternative_total_loss.npy'), total_loss_result)
        np.save(os.path.join(params.output_directory, 'out_dd.npy'), out_dd)

    return feature, tps_mat, disp_val, est_right_val, distance, loss_rec_val


def train_z(params, feature_in, tps_matrix, left_ims, right_ims):
    """
    train feature control points and A matrix of tps
    @param params: arguments
    @param feature_in: input feature control points
    @param left_ims: 
    @param right_ims: 
    @param tps_matrix: input A matrix 
    @return: updated feature ,disp and loss of reconstruction by disp
    """
    with tf.Graph().as_default(), tf.device('/gpu: 0'):
        linear_interpolator = LinearInterpolator(params)  # initialize linear interpolator
        loss_seq = np.empty(shape=[0, 3], dtype=np.float32)
        feature_in, tps_matrix, disp, est_right, distance, loss_rec_val = train_feature(
            params, feature_in, left_ims, right_ims, tps_matrix, linear_interpolator)
        # save trained tps matrix for predict
        np.save(os.path.join(params.output_directory, 'tps_trained.npy'), tps_matrix)
        # return loss value to check training details.
        return feature_in, disp, loss_rec_val


def train(params):
    tps_base0 = np.loadtxt(params.A_directory).astype(np.float32)
    # an appropriate initial value of disparities
    feature_in = 83.2 * np.ones([params.batch_size, params.cpts_row * params.cpts_col, 1], dtype=np.float32)
    ids = [x for x in range(params.data_size)]
    left_ims, right_ims = read_stereo_images(params.source_directory, ids)
    left_ims = np.array(left_ims, dtype=np.float32)
    right_ims = np.array(right_ims, dtype=np.float32)
    print("load train images num: {}".format(len(ids)))
    for epoch_num in range(params.epoch_num):
        T1 = time.time()
        print("----------Epoch {}---------".format(epoch_num))
        train_z(params, feature_in, tps_base0, left_ims, right_ims)
        T2 = time.time()
        print('Epoch %d cost time:%s s' % (epoch_num, (T2 - T1)))


def test(params):
    ids = [x for x in range(params.data_size, params.data_size + params.test_num)]
    print("load test images num: {}".format(len(ids)))
    # same original feature to predict
    if params.continue_predict:
        # to fit shape of tf variable feature_in_base
        params.batch_size = params.test_num
        left_ims, right_ims = read_stereo_images(params.source_directory, ids)
        left_ims = np.array(left_ims, dtype=np.float32)
        right_ims = np.array(right_ims, dtype=np.float32)
        tps_base0 = np.load(os.path.join(params.output_directory, 'tps_trained_.npy'))  # use trained tps matrix
        feature_in = 83.2 * np.ones([params.test_num, params.cpts_row * params.cpts_col, 1], dtype=np.float32)
        new_feature, disp, res_loss = train_z(params, feature_in, tps_base0, left_ims, right_ims)
        np.save(params.output_directory + 'TPSdisp_per_img.npy', disp)
        np.save(params.output_directory + 'TPSloss_per_img.npy', res_loss)
        print("save TPSdisp_per_img.npy and TPSloss_per_img.npy")
    # predict using updated feature
    else:
        disps = []
        loss = []
        params.batch_size = 1
        for i in ids:
            left_im, right_im = read_stereo_images(params.source_directory, [i])
            left_im = np.array(left_im, dtype=np.float32)
            right_im = np.array(right_im, dtype=np.float32)
            feature_in = 83.2 * np.ones([1, params.cpts_row * params.cpts_col, 1], dtype=np.float32)
            new_feature, disp, res_loss = train_z(params, feature_in, tps_base0, left_ims, right_ims)
            feature_in = new_feature  # update feature input for next image
            disps.append(disp)
            loss.append(res_loss)
        np.save(params.output_directory + 'TPSdisp_per_img.npy', disps)
        np.save(params.output_directory + 'TPSloss_per_img.npy', loss)
        print("save TPSdisp_per_img.npy and TPSloss_per_img.npy")


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        test(args)
