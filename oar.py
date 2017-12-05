#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: coco.py

import numpy as np
import os
from termcolor import colored
from tabulate import tabulate

from tensorpack.utils import logger
from tensorpack.utils.rect import FloatBox
from tensorpack.utils.timer import timed_operation
from tensorpack.utils.argtools import log_once

import pickle
import glob
from tqdm import tqdm
import config

ORGANS = ['Brain Stem', 'Chiasm', 'Cochlea', 'Eye',
          'Inner Ears', 'Larynx', 'Lens',
          'Optic Nerve', 'Spinal Cord']

ORGANS_MAPPING = dict([ (ind, organ) for ind, organ in zip(range(1,10), ORGANS) ])

class OARDetection(object):
    def __init__(self, basedir, dataset):
        self.basedir = os.path.join(basedir, dataset)

    def load(self, add_gt=True, add_mask=False):
        """
        Args:
            add_gt: whether to add ground truth bounding box annotations to the dicts
            add_mask: whether to also add ground truth mask
        Returns:
            a list of dict, each has keys including:
                height, width, id, file_name,
                and (if add_gt is True) boxes, class, is_crowd
        """
        filenames = glob.glob(self.basedir + '/**/*.pkl')
        ret = []
        for index, filename in tqdm(enumerate(filenames)):
            basename = os.path.basename(filename)
            basename, _ = os.path.splitext(basename)
            patid, idx = basename.split('_')
            patid, idx = int(patid), int(idx)

            with open(filename, 'rb') as f:
                _file = pickle.load(f, encoding='bytes')

            image_data = _file[b'image']

            xoffset = (image_data.shape[0]-200)//2
            yoffset = (image_data.shape[1]-200)//2
            assert xoffset == yoffset
            offset = xoffset

            assert image_data.shape == (200,200,3)
            image_data = image_data.astype(np.uint8)

            data = {}
            ret.append(data)
            data['height'] = 200
            data['width'] = 200
            data['id'] = index
            data['file_name'] = os.path.join(config.IMAGEDIR,basename) + '.png'

            num_instances = _file[b'num_instance']
#            if (num_instances == 0):
#               continue

            gt_masks = _file[b'masks']
            gt_masks = [ mask[offset:offset+200, offset:offset+200] for mask in gt_masks ]

            # list of numpy arrays
            data['masks'] = gt_masks
            assert len(data['masks'])== num_instances

            labels = _file[b'label']
            bboxes = _file[b'bboxes']
            boxes = []
            for gt in bboxes:
                xmin = float(gt[0])-offset
                ymin = float(gt[1])-offset
                xmax = float(gt[2])-offset
                ymax = float(gt[3])-offset
#                box = FloatBox(xmin, ymin, xmax, ymax)
                box = FloatBox(ymin, xmin, ymax, xmax)
                boxes.append([box.x1, box.y1, box.x2, box.y2])

            #classes = [ORGANS_MAPPING[la] for la in labels]
            classes = labels
            data['class'] = np.asarray(classes)
            data['boxes'] = np.asarray(boxes).astype(np.float32)
            #print(np.asarray(boxes).dtype)
            data['is_crowd'] = np.asarray([0]*data['class'].shape[0])
        return ret

    @staticmethod
    def load_many(basedir,names, add_gt=True, add_mask=False):
        """
        Load and merges several instance files together.
        """
        if not isinstance(names, (list, tuple)):
            names = [names]
        ret = []
        for n in names:
            oar = OARDetection(basedir, n)
            ret.extend(oar.load(add_gt, add_mask=add_mask))
        return ret

if __name__ == '__main__':
    c = OARDetection('/tmp2/oar/', 'ct_segmentation_data_val')
    gt_boxes = c.load(add_gt=True, add_mask=True)
    print(gt_boxes)
