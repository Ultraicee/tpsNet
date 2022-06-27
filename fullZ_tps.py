# Code for
# Reconstructing Dynamic Soft Tissue with Stereo Endoscope Based on a Single-layer Network
# Bo Yang, Siyuan Xu
#
# parts of the code from https://github.com/SiyuanXuu/tpsNet

from __future__ import absolute_import, division, print_function
from linear_sample import *
from decoder import *
from read_images import *

import argparse
# only keep warnings and errors
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import time
import tensorflow as tf

# set parameters of TPS
parser = argparse.ArgumentParser(description='train standard TPS')

parser.add_argument('--mode', type=str, help='TPS or FEATURE', default='FEATURE')
parser.add_argument('--continue_training', type=bool, help='use last step feature or not', default=False)
parser.add_argument('--data_size', type=int, help='num of total images', default=200)
parser.add_argument('--batch_size', type=int, help='num of minimum batch size', default=10)
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=3e-1)
parser.add_argument('--epoch_num', type=int, help='num of epochs', default=100)
parser.add_argument('--max_step', type=int, help='', default=1000)
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


def train_feature(params, feature_in, left_ims, right_ims, tps_base, tps_matrix, linear_interpolator,
                  loss_norm_mode=True):
    """
    a brief description to this function
    @param params: arguments
    @param feature_in: feature control points
    @param left_ims: 
    @param right_ims: 
    @param tps_base: original A matrix
    @param tps_matrix: updated A matrix 
    @param linear_interpolator: 
    @param loss_norm_mode:
    @return: 
    """
    print_str = 'Step{:4} | recons loss={:4} | norm loss={:4} | total loss={:4}' \
                '  | feature_var_mean={:4}'

    learning_rate_init = np.float32(params.learning_rate)
    optimize_op = tf.train.AdamOptimizer(learning_rate_init)

    with tf.variable_scope(tf.get_variable_scope()):
        tps_base = tf.constant(tps_base, dtype=tf.float32)
        left = tf.constant(left_ims[:params.batch_size], dtype=tf.float32)
        right = tf.constant(right_ims[:params.batch_size], dtype=tf.float32)
        if params.mode == 'TPS':
            feature_input = tf.Variable(feature_in, dtype=tf.float32, name='contr_val', trainable=False)
            tps_weight = tf.Variable(tps_matrix, dtype=tf.float32, name='tps_val')
        elif params.mode == 'FEATURE':
            feature_input = tf.Variable(feature_in, dtype=tf.float32, name='contr_val')
            tps_weight = tf.Variable(tps_matrix, dtype=tf.float32, name='tps_val', trainable=False)
        else:
            feature_input = tf.Variable(feature_in, dtype=tf.float32)
            tps_weight = tf.Variable(tps_matrix, dtype=tf.float32)

        compensateI = tf.Variable(4.3, dtype=tf.float32, name='contr_val')  # light compensation
        disp = decoder_forward(feature_input, tps_weight, linear_interpolator.sz_params)
        right_est = linear_interpolator.interpolate(left, disp)
        loss_rec, compa_sum, loss_rec_sum = compute_rec_loss(right_est, right, compensateI,
                                                             linear_interpolator.sz_params, 180.0)

        loss_wt_norm = tf.reduce_sum(tf.square(tps_weight - tps_base))
        loss = tf.add(loss_rec, loss_wt_norm, name='Total_loss')

        if params.mode == 'FEATURE' or not loss_norm_mode:
            train_op = optimize_op.minimize(loss_rec, var_list=feature_input)
        else:
            train_op = optimize_op.minimize(loss_rec, var_list=feature_input)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        loss_seq = np.zeros([params.max_step, 3])

        feature_before = feature_in
        feature_val, tps_val = 0., 0.
        disp_val = np.zeros([params.batch_size, params.input_height, params.input_width, 1])
        est_right_val = np.zeros([params.batch_size, params.input_height, params.input_width, params.input_channel])
        res_loss = [[], []]
        for step in range(0, params.max_step):
            _, feature_val, tps_val, loss_rec_val, loss_wt_norm_val, compa_sum_val, loss_rec_sum_val, loss_val, disp_val, est_right_val = sess.run(
                [train_op, feature_input, tps_weight, loss_rec, loss_wt_norm, compa_sum, loss_rec_sum, loss, disp,
                 right_est]
            )

            loss_seq[step, :] = loss_rec_val, loss_wt_norm_val, loss_val
            res_loss[0].append(loss_rec_sum_val * params.batch_size)
            res_loss[1].append(compa_sum_val)

            if 0 == step % 10 or step + 1 == params.max_step:
                feature_var_mean = np.mean(feature_val - feature_before)
                print(print_str.format(step, loss_rec_val, loss_wt_norm_val, loss_val, feature_var_mean))
                feature_before = feature_val
                if step >= 400:
                    break
        feature = feature_val
        tps_mat = tps_val
        distance = np.mean(np.square(feature - feature_in))
        print('feature distance after training:{}'.format(distance))

        return feature, tps_mat, disp_val, est_right_val, distance, loss_rec_val, res_loss


def train_z(params, feature_in, tps_base0, tps_matrix, left_ims, right_ims):
    with tf.Graph().as_default(), tf.device('/gpu: 0'):
        linear_interpolator = LinearInterpolator(params)  # initialize linear interpolator
        feature_out, tps_matrix, disp, est_right, distance, loss_rec_val, res_loss = train_feature(
            params, feature_in, left_ims, right_ims, tps_base0, tps_matrix, linear_interpolator)
    return feature_out, res_loss, disp, tps_matrix


def train(params):
    tps_base0 = np.loadtxt(params.A_directory).astype(np.float32)
    # inintial tps_matrix
    # tps_matrix= np.random.randn(tps_base0.shape[0],tps_base0.shape[1])
    tps_matrix = tps_base0
    # an appropriate initial value of disparities
    feature_in = 83.2 * np.ones([params.batch_size, params.cpts_row * params.cpts_col, 1], dtype=np.float32)
    for i in range(params.epoch_num):
        print("---------Epoch {}---------".format(i))
        T1 = time.time()
        ids = [x for x in range(params.data_size)]
        left_ims, right_ims = read_stereo_images(params.source_directory, ids)
        left_ims = np.array(left_ims, dtype=np.float32)
        right_ims = np.array(right_ims, dtype=np.float32)
        # update tps_matrix
        cur_feature, res_loss, disp, tps_matrix = train_z(params, feature_in, tps_base0, tps_matrix, left_ims, right_ims)
        if params.continue_training:
            feature_in = cur_feature
        T2 = time.time()
        print('Epoch %d cost time:%s s' % (i, (T2 - T1)))
    np.save(os.path.join(params.output_directory, 'tps_trained_.npy'), tps_matrix)
    np.save(os.path.join(params.output_directory, 'disp_batch_invivo.npy'), disp)
    np.save(os.path.join(params.output_directory, 'feature_trained.npy'), cur_feature)
    print("tps_matrix, disparity and current feature have saved")


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
    
