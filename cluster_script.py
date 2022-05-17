import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import argparse
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import utils.tf_nndistance as tf_nndistance
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--batch_size', type=int, default=5, help='Batch Size for attack [default: 5]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--data_dir', default='data', help='data folder path [data]')
parser.add_argument('--dump_dir', default='cluster', help='dump folder path [cluster]')

parser.add_argument('--add_num', type=int, default=32, help='number of added points [default: 512]')
parser.add_argument('--target', type=int, default=5, help='target class index')
parser.add_argument('--lr_attack', type=float, default=0.01, help='learning rate for optimization based attack')

parser.add_argument('--initial_weight', type=float, default=5, help='initial value for the parameter lambda')
parser.add_argument('--upper_bound_weight', type=float, default=30, help='upper_bound value for the parameter lambda')
parser.add_argument('--step', type=int, default=5, help='binary search step')
parser.add_argument('--num_iter', type=int, default=500, help='number of iterations for each binary search step')
parser.add_argument('--mu', type=float, default=0.05, help='preset value for parameter mu')
parser.add_argument('--init_dir', default='critical', help='the dir which contains the initial point')

FLAGS = parser.parse_args()

#batchsize = 5
BATCH_SIZE = FLAGS.batch_size
#num_point  = 1024
NUM_POINT = FLAGS.num_point
MODEL_PATH = os.path.join(FLAGS.log_dir, "model.ckpt")
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
#dump_dir 'cluster'
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
DATA_DIR = FLAGS.data_dir
#target class index
TARGET=FLAGS.target
#add_num = 1120- 1024 = 96?, 但是是32
NUM_ADD=FLAGS.add_num
#lr_attack learning rate for attack
LR_ATTACK=FLAGS.lr_attack
#WEIGHT=FLAGS.weight

attacked_data_all=joblib.load(os.path.join(DATA_DIR,'attacked_data.z'))
#initial value for lambda
INITIAL_WEIGHT=FLAGS.initial_weight
UPPER_BOUND_WEIGHT=FLAGS.upper_bound_weight
#ABORT_EARLY=False
BINARY_SEARCH_STEP=FLAGS.step
#number iteration for each step: 500
NUM_ITERATIONS=FLAGS.num_iter
# mu is mu
MU=FLAGS.mu
INIT_PATH=FLAGS.init_dir

#put the following code in the main code script
assert os.path.exists(os.path.join(BASE_DIR,INIT_PATH,'init_points_list_{}.z' .format(TARGET))), 'No init point found! run dbscan_clustering.py to generate init points'

init_points_list=joblib.load(os.path.join(BASE_DIR,INIT_PATH,'init_points_list_{}.z' .format(TARGET)))

#NUM_CLUSTER = 2
NUM_CLUSTER=len(init_points_list)#sometimes, there is only a limited number of cluster formed
                                 #so that DBSCAN may only get a NUM_CLUSTER smaller than the specified parameter
                                 #considering that, NUM_CLUSTER in this script is not a given parameter but obtained from the init point data

#make sure each element in init_point_list is in shape of BATCH_SIZE*NUM_ADD*3
#resize list中的元素：batch_size*num_add*3(xyz)
for i in range(NUM_CLUSTER):
    #tmp = 1024*3
    tmp=init_points_list[i]
    if len(tmp) >= NUM_ADD:
        tmp=tmp[-NUM_ADD:] 
    else:
        tmp=np.tile(tmp,[NUM_ADD // len(tmp),1])
        if NUM_ADD % len(tmp) != 0:
            tmp=np.concatenate([tmp,tmp[- (NUM_ADD % len(tmp)) : ]],axis=0)
    tmp=np.expand_dims(tmp,axis=0)
    init_points_list[i]=np.tile(tmp,[BATCH_SIZE,1,1])
    print(init_points_list[i].shape)
