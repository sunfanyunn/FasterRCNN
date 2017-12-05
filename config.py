#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: config.py

import numpy as np

# mode flags ---------------------
MODE_MASK = True

# dataset -----------------------
BASEDIR = '/tmp2/oar/ct_segmentation_data_fine/'
TRAIN_DATASET = ['train']
VAL_DATASET = 'val'   # only support evaluation on one dataset
TEST_DATASET = 'test'
NUM_CLASS = 10
 # NUM_CLASS strings
CLASS_NAMES = ['BG', 'Brain_Stem', 'Chiasm', 'Cochlea', 'Eye',
          'Inner_Ears', 'Larynx', 'Lens',
          'Optic_Nerve', 'Spinal_Cord']

# basemodel ----------------------
RESNET_NUM_BLOCK = [3, 4, 6, 3]     # resnet50

# preprocessing --------------------
SHORT_EDGE_SIZE = 800
MAX_SIZE = 1333
#SHORT_EDGE_SIZE = 200
#MAX_SIZE = 200
# alternative (better) setting: 800, 1333

# anchors -------------------------
ANCHOR_STRIDE = 16
#ANCHOR_STRIDE = 4
# sqrtarea of the anchor box
ANCHOR_SIZES = (32, 64, 128, 256, 512)
#ANCHOR_SIZES = (8, 16, 32, 64, 128)
ANCHOR_RATIOS = (0.5, 1., 2.)
NUM_ANCHOR = len(ANCHOR_SIZES) * len(ANCHOR_RATIOS)
POSITIVE_ANCHOR_THRES = 0.7
NEGATIVE_ANCHOR_THRES = 0.3
# just to avoid too large numbers.
BBOX_DECODE_CLIP = np.log(MAX_SIZE / 16.0)

# rpn training -------------------------
# keep fg ratio in a batch in this range
RPN_FG_RATIO = 0.5
RPN_BATCH_PER_IM = 256
RPN_MIN_SIZE = 0
RPN_PROPOSAL_NMS_THRESH = 0.7
TRAIN_PRE_NMS_TOPK = 12000
TRAIN_POST_NMS_TOPK = 2000
# boxes overlapping crowd will be ignored.
CROWD_OVERLAP_THRES = 0.7

# fastrcnn training ---------------------
FASTRCNN_BATCH_PER_IM = 256
FASTRCNN_BBOX_REG_WEIGHTS = np.array([10, 10, 5, 5], dtype='float32')
FASTRCNN_FG_THRESH = 0.5
# keep fg ratio in a batch in this range
FASTRCNN_FG_RATIO = 0.25

# testing -----------------------
TEST_PRE_NMS_TOPK = 6000
TEST_POST_NMS_TOPK = 1000
FASTRCNN_NMS_THRESH = 0.5
RESULT_SCORE_THRESH = 0.05
RESULTS_PER_IM = 100
