#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py

import numpy as np
import cv2

from tensorpack.dataflow import RNGDataFlow
from tensorpack.dataflow.imgaug import transform
from tensorpack.utils import logger

#import pycocotools.mask as cocomask

import config

def evaluate_mask_mean_iou(results_pack):
        gt_masks = {}
        pred_masks = {}
        case_hist = {}
        
        nine_organs = {}
        for result_rec in results_pack['results_list']:
            # result_rec contain prediction info
            # result_rec['image'] : source image store path
            # result_rec['boxes'], result_rec['masks'] are aligned
            image_path = result_rec['image']
            result_image_id = image_path.split("/")[-1].split(".")[0]
            im_info = result_rec['im_info']
            detections = result_rec['boxes']
            seg_masks = result_rec['masks']
            filename = image_path.split("/")[-1]
            filename = filename.replace('.png', '')
            # pred_semantic for debug (store slice information to debug)
            # you can ignore
            result_path = 'data/oar/results/pred_semantic/'
            if not (os.path.exists(result_path)):
                os.makedirs(result_path)
            print ('writing results for: ', filename)
            result_txt = os.path.join(result_path, filename)
            result_txt = result_txt + '.txt'
            count = 0
            # Whole image mask to store debug slice, can ignore
            # mask_image store instance masks need for final results
            whole_mask_image = np.zeros((int(im_info[0, 0]), int(im_info[0, 1])))
            for j, labelID in enumerate(self.class_id):
                if labelID == 0:
                    continue
                dets = detections[j]
                masks = seg_masks[j]
                mask_image = np.zeros((int(im_info[0, 0]), int(im_info[0, 1])))
                for i in range(len(dets)):
                    bbox = dets[i, :4]
                    # bbox = dets[i, :4]
                    score = dets[i, -1]
                    bbox = map(int, bbox)
                    # mask_image = np.zeros((int(im_info[0, 0]), int(im_info[0, 1])))
                    mask = masks[i, :, :]
                    mask = cv2.resize(mask, (bbox[2] - bbox[0], (bbox[3] - bbox[1])), interpolation=cv2.INTER_LINEAR)
                    mask[mask > 0.5] = 1
                    mask[mask <= 0.5] = 0
                    mask_image[bbox[1]: bbox[3], bbox[0]: bbox[2]] = mask
                    whole_mask_image[bbox[1]: bbox[3], bbox[0]: bbox[2]] = mask
                    # cv2.imwrite(os.path.join(result_path, filename) + '_' + str(count) + '.png', mask_image)
                    # f.write('{:s} {:s} {:.8f}\n'.format(filename + '_' + str(count) + '.png', str(labelID), score))
                    count += 1
                # According to patient_id and slice_num (depth, order of the ct slice)
                # I store mask_image into dict for later dump to nii
                patient_id = result_image_id.split("_")[0]
                slice_num = result_image_id.split("_")[1]
                if patient_id in nine_organs:
                    if slice_num not in nine_organs[patient_id]:
                        nine_organs[patient_id][slice_num] = {}
                    nine_organs[patient_id][slice_num][self.classes[labelID]] = mask_image
                else:
                    nine_organs[patient_id] = {}
                    nine_organs[patient_id][slice_num] = {}
                    nine_organs[patient_id][slice_num][self.classes[labelID]] = mask_image
                    # [122434_0]['Eye'] = mask
            #(for debug) cv2.imwrite(os.path.join(result_path, filename) + '.png', mask_image)
            pred_masks[result_image_id] = whole_mask_image
            #(for debug) writeLabelImage(mask_image, os.path.join(result_path, filename) + '.png')

        # write nine label binary nii
        for patient_id in nine_organs.keys():
            print("now processing {} ...".format(patient_id))
            for cls in self.classes:
                _s = []
                print("procssing {} organ".format(cls))
                for slice_id in range(len(nine_organs[patient_id])):
                    _slice = np.zeros((200, 200)).astype(np.uint8)
                    #offset = int((512 - 200)/2)
                    if cls in nine_organs[patient_id][str(slice_id)]:
                        #_slice[offset:offset + 200,offset:offset + 200] = cv2.resize(nine_organs[patient_id][str(slice_id)][cls], None, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                        current_path = result_path + str(patient_id) + '/' + cls
                        if not (os.path.exists(current_path)):
                            os.makedirs(current_path)
                        # just resize to 200*200
                        _slice = cv2.resize(nine_organs[patient_id][str(slice_id)][cls], None, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                        cv2.imwrite(current_path + '/' + '{0:03d}'.format(slice_id) + '.png', (_slice*1.0 / np.max(_slice)*255))
                    _s.append(_slice)
                _s = np.array(_s)
                # n*200*200
                print(_s.shape)
                dir_path = "/tmp2/oar/eval_files/"+patient_id
                filename = "/tmp2/oar/eval_files/"+patient_id+"/"+cls+".nii"
                if not (os.path.exists(dir_path)):
                    os.makedirs(dir_path)
                # sitk can write nii
                sitk.WriteImage(sitk.GetImageFromArray(_s), filename)


class DataFromListOfDict(RNGDataFlow):
    def __init__(self, lst, keys, shuffle=False):
        self._lst = lst
        self._keys = keys
        self._shuffle = shuffle
        self._size = len(lst)

    def size(self):
        return self._size

    def get_data(self):
        if self._shuffle:
            self.rng.shuffle(self._lst)
        for dic in self._lst:
            dp = [dic[k] for k in self._keys]
            yield dp


class CustomResize(transform.TransformAugmentorBase):
    """
    Try resizing the shortest edge to a certain number
    while avoiding the longest edge to exceed max_size.
    """

    def __init__(self, size, max_size, interp=cv2.INTER_LINEAR):
        """
        Args:
            size (int): the size to resize the shortest edge to.
            max_size (int): maximum allowed longest edge.
        """
        self._init(locals())

    def _get_augment_params(self, img):
        h, w = img.shape[:2]
        scale = self.size * 1.0 / min(h, w)
        if h < w:
            newh, neww = self.size, scale * w
        else:
            newh, neww = scale * h, self.size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return transform.ResizeTransform(h, w, newh, neww, self.interp)


def box_to_point8(boxes):
    """
    Args:
        boxes: nx4

    Returns:
        (nx4)x2
    """
    b = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]]
    b = b.reshape((-1, 2))
    return b


def point8_to_box(points):
    """
    Args:
        points: (nx4)x2
    Returns:
        nx4 boxes (x1y1x2y2)
    """
    p = points.reshape((-1, 4, 2))
    minxy = p.min(axis=1)   # nx2
    maxxy = p.max(axis=1)   # nx2
    return np.concatenate((minxy, maxxy), axis=1)


def segmentation_to_mask(polys, height, width):
    """
    Convert polygons to binary masks.

    Args:
        polys: a list of nx2 float array

    Returns:
        a binary matrix of (height, width)
    """
    polys = [p.flatten().tolist() for p in polys]
    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle)


def clip_boxes(boxes, shape):
    """
    Args:
        boxes: (...)x4, float
        shape: h, w
    """
    orig_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    h, w = shape
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], w)
    boxes[:, 3] = np.minimum(boxes[:, 3], h)
    return boxes.reshape(orig_shape)


def print_config():
    logger.info("Config: ------------------------------------------")
    for k in dir(config):
        if k == k.upper():
            logger.info("{} = {}".format(k, getattr(config, k)))
    logger.info("--------------------------------------------------")
