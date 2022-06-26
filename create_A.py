# Code for
# Reconstructing Dynamic Soft Tissue with Stereo Endoscope Based on a Single-layer Network
# Bo Yang, Siyuan Xu
#
# parts of the code from https://github.com/SiyuanXuu/tpsNet

from __future__ import absolute_import, division, print_function
import argparse
# only keep warnings and errors
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import tensorflow as tf
import numpy as np

# set interp_size and feature_size of TPS
parser = argparse.ArgumentParser(description='create A matrix')

parser.add_argument('--interp_height' , type=int, help='height of interpolate image', default= 200)
parser.add_argument('--interp_width',  type=int, help='width of interpolate image', default= 200)
parser.add_argument('--interp_top' ,   type=int, help='top left coordinate of interpolate image',default= 0)
parser.add_argument('--interp_left' ,  type=int, help='top left coordinate of interpolate image',default= 0)
parser.add_argument('--feature_row',   type=int, help='num of control points, sum=h*w',default=5)
parser.add_argument('--feature_col',   type=int, help='num of control points, sum=h*w',default=6)
parser.add_argument('--save_path',    type=str, help='path to save generated A',default='output/')


def matrix_Q(cp_cor_val, ob_cor_val):
    cp_row, cp_col = tf.split(cp_cor_val, 2, axis=1)
    ob_row, ob_col = tf.split(ob_cor_val, 2, axis=1)
    cp_ex_u, ob_ex_u = tf.meshgrid(cp_row, ob_row)
    cp_ex_v, ob_ex_v = tf.meshgrid(cp_col, ob_col)
    minus_u = ob_ex_u - cp_ex_u
    minus_v = ob_ex_v - cp_ex_v
    dist_mat = tf.add(tf.square(minus_u), tf.square(minus_v))
    H_mat = tf.multiply(dist_mat, tf.log(tf.clip_by_value(dist_mat, 1e-16, dist_mat)))
    m = tf.concat([ob_cor_val, tf.ones(shape=[tf.shape(ob_cor_val)[0], 1])], axis=1)
    Q = tf.concat([H_mat, m], axis=1)
    return Q


def get_A(params):
    tps = TPS_param([params.interp_height,params.interp_width],[params.interp_top,params.interp_left], [params.feature_row, params.feature_col])
    A = tps.A
    with tf.device('/gpu: 0'):
        with tf.Session() as sess:
            A_val = sess.run(A)
            print("A_val.shape={}".format(A_val.shape))
            if params.save_path:
                np.savetxt(params.save_path+'A_val.txt', A_val)
                print('save file :',params.save_path+'A_val.txt')
            else:
                print("Warning: You haven't save it, set [--save_path] to save A_val.txt")

class TPS_param:
    def __init__(self, interp_size, interp_start, feature_size):
        """
        :param interp_size: [h, w]
        :param interp_start: [u, v]
        :param feature_size: the shape of feature:[h, w]
        """
        with tf.variable_scope('tps_param_initial'):
            self.h = interp_size[0]
            self.w = interp_size[1]
            self.control_num_v = feature_size[0]
            self.control_num_u = feature_size[1]
            cp_cor_val = self.generate_cord()  # pixel coordinate of control points
            cp_cor_val = tf.reshape(cp_cor_val, [-1, 2])
            self.cp_cor = tf.cast(cp_cor_val, tf.float32)
            self.D = self.matrix_D(self.cp_cor)
            self.img_top_left = tf.constant(interp_start, dtype=tf.float32)
            self.img_top_left_u = self.img_top_left[0]
            self.img_top_left_v = self.img_top_left[1]
            self.A = self.matrix_A()  # A=Q*D

    def generate_cord(self):
        edge_u_left = tf.cast(tf.floor(self.w % (self.control_num_u - 1) / 2), tf.int32)
        edge_v_up = tf.cast(tf.floor(self.h % (self.control_num_v - 1) / 2), tf.int32)
        edge_u_right = self.w % (self.control_num_u - 1) - edge_u_left
        edge_v_low = self.h % (self.control_num_v - 1) - edge_v_up
        step_u = tf.cast(tf.floor(self.w / (self.control_num_u - 1)), tf.int32)
        step_v = tf.cast(tf.floor(self.h / (self.control_num_v - 1)), tf.int32)
        # The control points are uniform distribution,
        # and calculate the coordinate boundaries of the control points
        range_end_u = self.w - edge_u_right
        range_end_v = self.h - edge_v_low

        control_u = tf.range(edge_u_left, range_end_u + 1, delta=step_u)
        control_v = tf.range(edge_v_up, range_end_v + 1, delta=step_v)
        c_u, c_v = tf.meshgrid(control_u, control_v)
        c_u = tf.expand_dims(c_u, axis=2)
        c_v = tf.expand_dims(c_v, axis=2)
        control_cor = tf.concat([c_u, c_v], axis=2)
        return control_cor

    def matrix_D(self, cor_val):
        """
        calculate the D matrix and model parameters A in the paper
        @param cor_val: all control points' pixel coordinate，shape=[K, 2]，dtype = tf.float32
        @return: matrix D
        """
        cp_num = tf.shape(cor_val)[0]
        eye = tf.ones(shape=[1, cp_num], dtype=tf.float32)
        G = tf.concat([tf.matrix_transpose(cor_val), eye], axis=0)
        G = tf.cast(G, tf.float32)
        G_t = tf.matrix_transpose(G)
        cor_row, cor_col = tf.split(cor_val, 2, axis=1)
        u = tf.reshape(cor_row, shape=[-1])
        v = tf.reshape(cor_col, shape=[-1])
        H = self.matrix_H(u, v)
        F_left = tf.concat([H, G], axis=0)
        F_right = tf.concat([G_t, tf.zeros(shape=[3, 3])], axis=0)
        F = tf.concat([F_left, F_right], axis=1)
        F_inv = tf.matrix_inverse(F)
        D = tf.slice(F_inv, [0, 0], [cp_num + 3, cp_num])
        return D

    def matrix_H(self, cp_cor_row, cp_cor_col):
        """
        calculate the H matrix in the paper
        """
        minus_u = self.matrix_cor(cp_cor_row)
        minus_v = self.matrix_cor(cp_cor_col)
        dist_mat = tf.cast(tf.add(tf.square(minus_u), tf.square(minus_v)), dtype=tf.float32)
        H = tf.multiply(dist_mat, tf.log(tf.clip_by_value(dist_mat, 1e-16, dist_mat)))
        return H

    def matrix_cor(self, cor_val):
        """
        construct a distance matrix between control points for computing H
        """
        _, cor_ob = tf.meshgrid(cor_val, cor_val)
        cor_cp = tf.tile(tf.expand_dims(cor_val, axis=0), [self.control_num_u * self.control_num_v, 1])
        minus = cor_ob - cor_cp  # difference of colum coordinate
        return minus

    def matrix_A(self):
        w = tf.range(self.img_top_left_u, self.w)
        h = tf.range(self.img_top_left_v, self.h)
        mesh_h, mesh_w = tf.meshgrid(w, h)
        img_cor = tf.concat([tf.expand_dims(mesh_w, axis=-1), tf.expand_dims(mesh_h, axis=-1)], axis=2)
        img_cor_resh = tf.cast(tf.reshape(img_cor, shape=[self.h * self.w, 2]), dtype=tf.float32)
        Q = matrix_Q(self.cp_cor, img_cor_resh)
        A = tf.matmul(Q, self.D)
        print('------------tps calculate------------')
        return A
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    get_A(args)