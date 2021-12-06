from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from PixelateMetalion import *
from SampleSeg import *
import csv
import h5py
import os
import pandas as pd
import math
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import cnn_utils
from argparse import ArgumentParser

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


ION_TYPE = 'Mg'
TIME_FORMAT = "-%y-%m-%d-%H-%M-%S"
GRID_SIZE = 20
GRID_VOXELS = GRID_SIZE * GRID_SIZE * GRID_SIZE
NB_TYPE = 4


def PReLU(_x, name=None):
    with tf.variable_scope(name_or_scope=name, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.001))
    return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def create_placeholders(m, n_n0, n_a0, n_b0, n_c0, n_y):
    X = tf.placeholder(tf.float32, [None, n_n0, n_a0, n_b0, n_c0])
    # X = tf.Variable(X_init)
    Y = tf.placeholder(tf.float32, [None, n_y])

    return X, Y


def _weight_variable(name, shape):
    return tf.get_variable(name, shape, tf.float32, tf.truncated_normal_initializer(stddev=0.01))


def _bias_variable(name, shape):
    return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0.1, dtype=tf.float32))


def scoringModel(maps, isTraining, batch_norm=True, validation='softplus', final_activation='sigmoid'):
    print("Create model start")
    prev_layer = tf.reshape(maps, [-1, NB_TYPE, GRID_SIZE, GRID_SIZE, GRID_SIZE])
    prev_layer = tf.transpose(prev_layer, perm=[0, 2, 3, 4, 1])

    CONV1_OUT = 32

    kernelConv1 = _weight_variable("weights_C1" + "_" + str(NB_TYPE), [3, 3, 3, NB_TYPE, CONV1_OUT])
    prev_layer = tf.nn.conv3d(prev_layer, kernelConv1, [1, 1, 1, 1, 1], padding='VALID')
    biasConv1 = _bias_variable("biases_C1" + "_" + str(NB_TYPE), [CONV1_OUT])

    prev_layer = prev_layer + biasConv1;

    if batch_norm:
        prev_layer = tf.layers.batch_normalization(prev_layer, training=isTraining, name="batchn1")
    prev_layer = tf.nn.dropout(prev_layer, 1 - tf.cast(isTraining, dtype=tf.float32) * 0.5, name="dropout1")

    if validation == 'softplus':
        conv1 = tf.nn.softplus(prev_layer, name="softplus1")
    elif validation == 'elu':
        conv1 = tf.nn.elu(prev_layer, name="elu1")
    else:
        conv1 = tf.nn.relu(prev_layer, name="relu1")

    CONV2_OUT = 64

    kernelConv2 = _weight_variable("weights_C2" + "_" + str(NB_TYPE), [3, 3, 3, CONV1_OUT, CONV2_OUT])
    prev_layer = tf.nn.conv3d(conv1, kernelConv2, [1, 1, 1, 1, 1], padding='VALID')
    biasConv2 = _bias_variable("biases_C2" + "_" + str(NB_TYPE), [CONV2_OUT])

    prev_layer = prev_layer + biasConv2;

    if batch_norm:
        prev_layer = tf.layers.batch_normalization(prev_layer, training=isTraining, name="batchn2")
    prev_layer = tf.nn.dropout(prev_layer, 1 - tf.cast(isTraining, dtype=tf.float32) * 0.5, name="dropout2")

    if validation == 'softplus':
        conv2 = tf.nn.softplus(prev_layer, name="softplus2")
    elif validation == 'elu':
        conv2 = tf.nn.elu(prev_layer, name="elu2")
    else:
        conv2 = tf.nn.relu(prev_layer, name="relu2")
    POOL_SIZE = 2

    prev_layer = tf.nn.max_pool3d(conv2,
                                  [1, POOL_SIZE, POOL_SIZE, POOL_SIZE, 1],
                                  [1, POOL_SIZE, POOL_SIZE, POOL_SIZE, 1],
                                  padding='VALID')

    CONV3_OUT = 128

    kernelConv3 = _weight_variable("weights_C3" + "_" + str(NB_TYPE), [3, 3, 3, CONV2_OUT, CONV3_OUT])
    prev_layer = tf.nn.conv3d(prev_layer, kernelConv3, [1, 1, 1, 1, 1], padding='VALID')

    biasConv3 = _bias_variable("biases_C3" + "_" + str(NB_TYPE), [CONV3_OUT])

    with tf.name_scope('Conv3'):
        tf.summary.histogram("weights_C3", kernelConv3)
        tf.summary.histogram("bias_C3", biasConv3)

    prev_layer = prev_layer + biasConv3;

    if batch_norm:
        prev_layer = tf.layers.batch_normalization(prev_layer, training=isTraining, name="batchn3")
    prev_layer = tf.nn.dropout(prev_layer, 1 - tf.cast(isTraining, dtype=tf.float32) * 0.5, name="dropout3")

    if validation == 'softplus':
        conv3 = tf.nn.softplus(prev_layer, name="softplus3")
    elif validation == 'elu':
        conv3 = tf.nn.elu(prev_layer, name="elu3")
    else:
        conv3 = tf.nn.relu(prev_layer, name="relu3")

    CONV4_OUT = 128

    kernelConv4 = _weight_variable("weights_C4" + "_" + str(NB_TYPE), [3, 3, 3, CONV3_OUT, CONV4_OUT])
    prev_layer = tf.nn.conv3d(conv3, kernelConv4, [1, 1, 1, 1, 1], padding='VALID')

    biasConv4 = _bias_variable("biases_C4" + "_" + str(NB_TYPE), [CONV4_OUT])

    with tf.name_scope('Conv4'):
        tf.summary.histogram("weights_C4", kernelConv4)
        tf.summary.histogram("bias_C4", biasConv4)

    prev_layer = prev_layer + biasConv4;

    if batch_norm:
        prev_layer = tf.layers.batch_normalization(prev_layer, training=isTraining, name="batchn4")
    prev_layer = tf.nn.dropout(prev_layer, 1 - tf.cast(isTraining, dtype=tf.float32) * 0.5, name="dropout4")

    if validation == 'softplus':
        conv4 = tf.nn.softplus(prev_layer, name="softplus4")
    elif validation == 'elu':
        conv4 = tf.nn.elu(prev_layer, name="elu4")
    else:
        conv4 = tf.nn.relu(prev_layer, name="relu4")

    prev_layer = tf.nn.max_pool3d(conv4,
                                  [1, POOL_SIZE, POOL_SIZE, POOL_SIZE, 1],
                                  [1, POOL_SIZE, POOL_SIZE, POOL_SIZE, 1],
                                  padding='VALID')

    NB_DIMOUT = 2 * 2 * 2 * CONV4_OUT
    flat0 = tf.reshape(prev_layer, [-1, NB_DIMOUT])

    LINEAR1_OUT = 512

    weightsLinear = _weight_variable("weights_L1" + "_" + str(NB_TYPE), [NB_DIMOUT, LINEAR1_OUT])

    prev_layer = tf.matmul(flat0, weightsLinear)
    biasLinear1 = _bias_variable("biases_L1" + "_" + str(NB_TYPE), [LINEAR1_OUT])

    with tf.name_scope('Linear1'):
        tf.summary.histogram("weights_L1", weightsLinear)
        tf.summary.histogram("biases_L1", biasLinear1)

    prev_layer = prev_layer + biasLinear1

    # prev_layer = tf.nn.l2_normalize(flat0,dim=1)
    if batch_norm:
        prev_layer = tf.layers.batch_normalization(prev_layer, training=isTraining, name="batchn5")

    if validation == 'softplus':
        flat1 = tf.nn.softplus(prev_layer, name="softplus5")
    elif validation == 'elu':
        flat1 = tf.nn.elu(prev_layer, name="elu5")
    else:
        flat1 = PReLU(prev_layer, name="relu5")

    weightsLinear2 = _weight_variable("weights_L2" + "_" + str(NB_TYPE), [LINEAR1_OUT, 2])

    with tf.name_scope('Linear2'):
        tf.summary.histogram("weights_L2", weightsLinear2)

    last = tf.matmul(flat1, weightsLinear2)
    print("Create model end")
    scores = tf.squeeze(last)

    return scores


def loss(scores, Y):
    cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
        logits=scores, labels=Y, pos_weight=1))
    return cost


def training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay = 0.999)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def Sequence_mini_batches(X, mini_batch_size=64):
    # 用于模型评估部分
    m = X.shape[0]  # number of training examples
    mini_batches = []
    num_complete_minibatches = math.floor(m / mini_batch_size)
    # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :, :]
        mini_batch = (mini_batch_X)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch &lt; mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch = (mini_batch_X)
        mini_batches.append(mini_batch)
    return mini_batches


def model(X_independ):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    X_independ = np.squeeze(X_independ)
    X_independ = np.squeeze(X_independ)
    (m, n_n0, n_a0, n_b0, n_c0) = X_independ.shape
    m_test, _, _, _, _ = X_independ.shape
    m_independ, _, _, _, _ = X_independ.shape

    keep_prob = tf.placeholder(tf.float32)
    X = tf.placeholder(tf.float32, [None, n_n0, n_a0, n_b0, n_c0])
    scores = scoringModel(X, isTraining=True, batch_norm=True,
                          validation='Relu', final_activation='softmax')
    tf.add_to_collection("predict", scores)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    if ION_TYPE in ['Mg']:
        saver.restore(sess, "./model/Mg/cnn.ckpt")
    elif ION_TYPE in ['Na']:
        saver.restore(sess, "./model/Na/cnn.ckpt")
    elif ION_TYPE in ['K']:
        saver.restore(sess, "./model/K/cnn.ckpt")

    P_in_pred = []
    P = tf.nn.softmax(scores)
    in_minibatches = Sequence_mini_batches(X_independ, 32)

    for in_minibatch in in_minibatches:
        mini_in_X = in_minibatch
        P_in_pred.extend(sess.run(P, feed_dict={X: mini_in_X, keep_prob: 1.0}))

    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)

    P_in_pred = np.array(P_in_pred)
    return P_in_pred


def get_test_maps(vector):
    length = len(vector)
    pixels = np.zeros((length, 5, NBINS, NBINS, NBINS))
    for index in range(length):
        RNA_name = vector[index][1]
        root = 'pdb/'
        RNA_path = os.path.join(root, RNA_name)
        with open(RNA_path) as RNA:
            p = PDBParser(QUIET=True)  # 构建PDB解释器 #
            s = p.get_structure(RNA, RNA)
            model = s[0]
            residue_list = list(model.get_residues())
            modify_residue_atom_name1(residue_list)
            pixels = pixelate_atoms_in_box(vector[index][0], model, pixels, index)

    return pixels


script_dir, script_name = os.path.split(os.path.abspath(sys.argv[0]))
parent_dir = os.path.dirname(script_dir)
DATA_BASE_PATH = script_dir + '/pdb/'
RESULT_BASE_PATH = script_dir + '/result/'
parser = ArgumentParser()
parser.add_argument('-s', '--structure', type=str, help='The structure you want to predict.')
parser.add_argument('-m', '--metal', type=str, help='The ion type you want to predict.')
args = parser.parse_args()

if args.metal != None:
    ION_TYPE = args.metal
print("ion type: %s" % ION_TYPE)
result_save_path = RESULT_BASE_PATH + ION_TYPE + time.strftime(TIME_FORMAT, time.localtime()) + "/"
if not os.path.exists(result_save_path):
    os.mkdir(result_save_path)
args.structure = "1hc8d.pdb"
if args.structure != None:
    for root, dirs, pdb_list in os.walk(DATA_BASE_PATH):
        for RNA_list in pdb_list:
            if RNA_list == args.structure:
                RNA_path = os.path.join(root, RNA_list)
                break

    sample_list, index_list = generate_grid_sample(RNA_path)
    random_pixels = get_test_maps(sample_list)
    X_independ = random_pixels
    X_independ = cnn_utils.slice_array(X_independ)
    P_in_pred = model_metrics = model(X_independ)
    result = np.hstack((sample_list, P_in_pred))
    result = pd.DataFrame(data=result)
    result.to_excel(result_save_path + RNA_list + ".xlsx")
    print("predict finished")
else:
    print("please input the structure that you want to predict")


