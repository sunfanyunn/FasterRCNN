#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

import os
import argparse
import cv2
import shutil
import itertools
import tqdm
import math
import numpy as np
import json
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import optimizer
import tensorpack.utils.viz as tpviz
from tensorpack.utils.gpu import get_nr_gpu

from basemodel import (
    image_preprocess, pretrained_resnet_conv4, resnet_conv5)
from model import (
    clip_boxes, decode_bbox_target, encode_bbox_target, crop_and_resize,
    rpn_head, rpn_losses,
    generate_rpn_proposals, sample_fast_rcnn_targets, roi_align,
    fastrcnn_head, fastrcnn_losses, fastrcnn_predictions,
    maskrcnn_head, maskrcnn_loss)
from data import (
    get_train_dataflow, get_eval_dataflow,
    get_all_anchors)
from viz import (
    draw_annotation, draw_proposal_recall,
    draw_predictions, draw_final_outputs)
from common import print_config
from eval import (
    eval_on_dataflow, detect_one_image, DetectionResult)
import config

from oar import OARDetection
import matplotlib.pyplot as plt
import collections
import SimpleITK as sitk
from tqdm import tqdm

'''
 input: a list of dic
{
 'image': image_path
 'boxes': dic of list
 'masks': dic of list of numpy arrys
 'im_info':
'''
def evaluate_mask_mean_iou(results_pack):
        gt_masks = {}
        pred_masks = {}
        case_hist = {}
        
        nine_organs = collections.defaultdict(lambda :collections.defaultdict(lambda :collections.defaultdict(lambda :np.zeros((200,200)))))

        for result_rec in results_pack:
            # result_rec contain prediction info
            # result_rec['image'] : source image store path
            # result_rec['boxes'], result_rec['masks'] are aligned
            image_path = result_rec['image']
            result_image_id = image_path.split("/")[-1].split(".")[0]
            w,h = result_rec['im_info']
            detections = result_rec['boxes']
            seg_masks = result_rec['masks']
            filename = image_path.split("/")[-1]
            filename = filename.replace('.png', '')

            count = 0
            # Whole image mask to store debug slice, can ignore
            # mask_image store instance masks need for final results
            whole_mask_image = np.zeros((w, h))
            for idx, cl in enumerate(config.CLASS_NAMES):
                if idx == 0:
                    continue
                dets = detections[cl]
                masks = seg_masks[cl]
                mask_image = np.zeros((w, h))
                for i in range(len(dets)):
                    assert len(dets[i]) == 5
                    bbox = dets[i][:4]
                    score = dets[i][-1]
                    if score < 0.5:
                        continue
                    bbox = list(map(int, bbox))
                    mask = masks[i]
#                    mask = cv2.resize(mask, (bbox[2] - bbox[0], (bbox[3] - bbox[1])), interpolation=cv2.INTER_LINEAR)
                    mask[mask > 0.5] = 1
                    mask[mask <= 0.5] = 0
                    assert mask_image.shape == mask.shape
                    mask_image[bbox[1]: bbox[3], bbox[0]: bbox[2]] = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    assert whole_mask_image.shape == mask.shape
                    whole_mask_image[bbox[1]: bbox[3], bbox[0]: bbox[2]] = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    # cv2.imwrite(os.path.join(result_path, filename) + '_' + str(count) + '.png', mask_image)
                    # f.write('{:s} {:s} {:.8f}\n'.format(filename + '_' + str(count) + '.png', str(idx), score))
                    count += 1
                # According to patient_id and slice_num (depth, order of the ct slice)
                # I store mask_image into dict for later dump to nii
                patient_id = result_image_id.split("_")[0]
                slice_num = result_image_id.split("_")[1]
                nine_organs[patient_id][slice_num][cl] = mask_image
            #(for debug) cv2.imwrite(os.path.join(result_path, filename) + '.png', mask_image)
            pred_masks[result_image_id] = whole_mask_image
            #(for debug) writeLabelImage(mask_image, os.path.join(result_path, filename) + '.png')

        # write nine label binary nii
        for patient_id in nine_organs.keys():
            print("now processing {} ...".format(patient_id))
            for cls in config.CLASS_NAMES[1:]:
                _s = []
                print("procssing {} organ".format(cls))
                for slice_id in range(len(nine_organs[patient_id])):
                    _slice = nine_organs[patient_id][str(slice_id)][cls]
#                    print(_slice.shape)
#                    _slice = cv2.resize(nine_organs[patient_id][str(slice_id)][cls], None, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
#                    print(_slice.shape)
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

def get_batch_factor():
    nr_gpu = get_nr_gpu()
    assert nr_gpu in [1, 2, 4, 8], nr_gpu
    return 8 // nr_gpu

def get_model_output_names():
    ret = ['final_boxes', 'final_probs', 'final_labels']
    if config.MODE_MASK:
        ret.append('final_masks')
    return ret

class Model(ModelDesc):
    def _get_inputs(self):
        ret = [
            InputDesc(tf.float32, (None, None, 3), 'image'),
            InputDesc(tf.int32, (None, None, config.NUM_ANCHOR), 'anchor_labels'),
            InputDesc(tf.float32, (None, None, config.NUM_ANCHOR, 4), 'anchor_boxes'),
            InputDesc(tf.float32, (None, 4), 'gt_boxes'),
            InputDesc(tf.int64, (None,), 'gt_labels')]  # all > 0
        if config.MODE_MASK:
            ret.append(
                InputDesc(tf.uint8, (None, None, None), 'gt_masks')
            )   # NR_GT x height x width
        return ret

    def _preprocess(self, image):
        image = tf.expand_dims(image, 0)
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

    def _get_anchors(self, image):
        """
        Returns:
            FSxFSxNAx4 anchors,
        """
        # FSxFSxNAx4 (FS=MAX_SIZE//ANCHOR_STRIDE)
        with tf.name_scope('anchors'):
            all_anchors = tf.constant(get_all_anchors(), name='all_anchors', dtype=tf.float32)
            fm_anchors = tf.slice(
                all_anchors, [0, 0, 0, 0], tf.stack([
                    tf.shape(image)[0] // config.ANCHOR_STRIDE,
                    tf.shape(image)[1] // config.ANCHOR_STRIDE,
                    -1, -1]), name='fm_anchors')

            return fm_anchors

    def _build_graph(self, inputs):
        is_training = get_current_tower_context().is_training
        if config.MODE_MASK:
            image, anchor_labels, anchor_boxes, gt_boxes, gt_labels, gt_masks = inputs
        else:
            image, anchor_labels, anchor_boxes, gt_boxes, gt_labels = inputs
        fm_anchors = self._get_anchors(image)
        image = self._preprocess(image)     # 1CHW
        image_shape2d = tf.shape(image)[2:]

        anchor_boxes_encoded = encode_bbox_target(anchor_boxes, fm_anchors)
        featuremap = pretrained_resnet_conv4(image, config.RESNET_NUM_BLOCK[:3])
        rpn_label_logits, rpn_box_logits = rpn_head('rpn', featuremap, 1024, config.NUM_ANCHOR)

        decoded_boxes = decode_bbox_target(rpn_box_logits, fm_anchors)  # fHxfWxNAx4, floatbox
        proposal_boxes, proposal_scores = generate_rpn_proposals(
            tf.reshape(decoded_boxes, [-1, 4]),
            tf.reshape(rpn_label_logits, [-1]),
            image_shape2d)

        if is_training:
            # sample proposal boxes in training
            rcnn_sampled_boxes, rcnn_labels, fg_inds_wrt_gt = sample_fast_rcnn_targets(
                proposal_boxes, gt_boxes, gt_labels)
            boxes_on_featuremap = rcnn_sampled_boxes * (1.0 / config.ANCHOR_STRIDE)
        else:
            # use all proposal boxes in inference
            boxes_on_featuremap = proposal_boxes * (1.0 / config.ANCHOR_STRIDE)

        roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)

        # HACK to work around https://github.com/tensorflow/tensorflow/issues/14657
        def ff_true():
            feature_fastrcnn = resnet_conv5(roi_resized, config.RESNET_NUM_BLOCK[-1])    # nxcx7x7
            fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_head('fastrcnn', feature_fastrcnn, config.NUM_CLASS)
            return feature_fastrcnn, fastrcnn_label_logits, fastrcnn_box_logits

        def ff_false():
            ncls = config.NUM_CLASS
            return tf.zeros([0, 2048, 7, 7]), tf.zeros([0, ncls]), tf.zeros([0, ncls - 1, 4])

        feature_fastrcnn, fastrcnn_label_logits, fastrcnn_box_logits = tf.cond(
            tf.size(boxes_on_featuremap) > 0, ff_true, ff_false)

        if is_training:
            # rpn loss
            rpn_label_loss, rpn_box_loss = rpn_losses(
                anchor_labels, anchor_boxes_encoded, rpn_label_logits, rpn_box_logits)

            # fastrcnn loss
            fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])   # fg inds w.r.t all samples
            fg_sampled_boxes = tf.gather(rcnn_sampled_boxes, fg_inds_wrt_sample)

            with tf.name_scope('fg_sample_patch_viz'):
                fg_sampled_patches = crop_and_resize(
                    image, fg_sampled_boxes,
                    tf.zeros_like(fg_inds_wrt_sample, dtype=tf.int32), 300)
                fg_sampled_patches = tf.transpose(fg_sampled_patches, [0, 2, 3, 1])
                fg_sampled_patches = tf.reverse(fg_sampled_patches, axis=[-1])  # BGR->RGB
                tf.summary.image('viz', fg_sampled_patches, max_outputs=30)

            matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)
            encoded_boxes = encode_bbox_target(
                matched_gt_boxes,
                fg_sampled_boxes) * tf.constant(config.FASTRCNN_BBOX_REG_WEIGHTS)
            fastrcnn_label_loss, fastrcnn_box_loss = fastrcnn_losses(
                rcnn_labels, fastrcnn_label_logits,
                encoded_boxes,
                tf.gather(fastrcnn_box_logits, fg_inds_wrt_sample))

            if config.MODE_MASK:
                # maskrcnn loss
                fg_labels = tf.gather(rcnn_labels, fg_inds_wrt_sample)
                fg_feature = tf.gather(feature_fastrcnn, fg_inds_wrt_sample)
                mask_logits = maskrcnn_head('maskrcnn', fg_feature, config.NUM_CLASS)   # #fg x #cat x 14x14

                gt_masks_for_fg = tf.gather(gt_masks, fg_inds_wrt_gt)  # nfg x H x W
                target_masks_for_fg = crop_and_resize(
                    tf.expand_dims(gt_masks_for_fg, 1),
                    fg_sampled_boxes,
                    tf.range(tf.size(fg_inds_wrt_gt)), 14)  # nfg x 1x14x14
                target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, 'sampled_fg_mask_targets')
                mrcnn_loss = maskrcnn_loss(mask_logits, fg_labels, target_masks_for_fg)
            else:
                mrcnn_loss = 0.0

            wd_cost = regularize_cost(
                '(?:group1|group2|group3|rpn|fastrcnn|maskrcnn)/.*W',
                l2_regularizer(1e-4), name='wd_cost')

            self.cost = tf.add_n([
                rpn_label_loss, rpn_box_loss,
                fastrcnn_label_loss, fastrcnn_box_loss,
                mrcnn_loss,
                wd_cost], 'total_cost')

            add_moving_summary(self.cost, wd_cost)
        else:
            label_probs = tf.nn.softmax(fastrcnn_label_logits, name='fastrcnn_all_probs')  # #proposal x #Class
            anchors = tf.tile(tf.expand_dims(proposal_boxes, 1), [1, config.NUM_CLASS - 1, 1])   # #proposal x #Cat x 4
            decoded_boxes = decode_bbox_target(
                fastrcnn_box_logits /
                tf.constant(config.FASTRCNN_BBOX_REG_WEIGHTS), anchors)
            decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')

            # indices: Nx2. Each index into (#proposal, #category)
            pred_indices, final_probs = fastrcnn_predictions(decoded_boxes, label_probs)
            final_probs = tf.identity(final_probs, 'final_probs')
            final_boxes = tf.gather_nd(decoded_boxes, pred_indices, name='final_boxes')
            final_labels = tf.add(pred_indices[:, 1], 1, name='final_labels')

            if config.MODE_MASK:
                # HACK to work around https://github.com/tensorflow/tensorflow/issues/14657
                def f1():
                    roi_resized = roi_align(featuremap, final_boxes * (1.0 / config.ANCHOR_STRIDE), 14)
                    feature_maskrcnn = resnet_conv5(roi_resized, config.RESNET_NUM_BLOCK[-1])
                    mask_logits = maskrcnn_head(
                        'maskrcnn', feature_maskrcnn, config.NUM_CLASS)   # #result x #cat x 14x14
                    indices = tf.stack([tf.range(tf.size(final_labels)), tf.to_int32(final_labels) - 1], axis=1)
                    final_mask_logits = tf.gather_nd(mask_logits, indices)   # #resultx14x14
                    return tf.sigmoid(final_mask_logits)

                final_masks = tf.cond(tf.size(final_probs) > 0, f1, lambda: tf.zeros([0, 14, 14]))
                tf.identity(final_masks, name='final_masks')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.003, trainable=False)
        tf.summary.scalar('learning_rate', lr)

        factor = get_batch_factor()
        if factor != 1:
            lr = lr / float(factor)
            opt = tf.train.MomentumOptimizer(lr, 0.9)
            opt = optimizer.AccumGradOptimizer(opt, factor)
        else:
            opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt

def visualize(model_path, nr_visualize=50, output_dir='visualize'):
    df = get_train_dataflow()   # we don't visualize mask stuff
    df.reset_state()

    pred = OfflinePredictor(PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_path),
        input_names=['image', 'gt_boxes', 'gt_labels'],
        output_names=[
            'generate_rpn_proposals/boxes',
            'generate_rpn_proposals/probs',
            'fastrcnn_all_probs',
            'final_boxes',
            'final_probs',
            'final_labels',
            'final_masks',
        ]))

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    utils.fs.mkdir_p(output_dir)
    with tqdm.tqdm(total=nr_visualize) as pbar:
        for idx, dp in itertools.islice(enumerate(df.get_data()), nr_visualize):
            img, _, _, gt_boxes, gt_labels = dp

            rpn_boxes, rpn_scores, all_probs, \
                final_boxes, final_probs, final_labels, final_masks = pred(img, gt_boxes, gt_labels)

            # draw groundtruth boxes
            gt_viz = draw_annotation(img, gt_boxes, gt_labels)
            # draw best proposals for each groundtruth, to show recall
            proposal_viz, good_proposals_ind = draw_proposal_recall(img, rpn_boxes, rpn_scores, gt_boxes)
            # draw the scores for the above proposals
            score_viz = draw_predictions(img, rpn_boxes[good_proposals_ind], all_probs[good_proposals_ind])

            #results = [DetectionResult(*args) for args in zip(final_boxes, final_probs, final_labels, final_masks)]
            results = [DetectionResult(*args) for args in
                       zip(final_boxes, final_probs, final_labels,
                           [None] * len(final_labels))]
            final_viz = draw_final_outputs(img, results)

            viz = tpviz.stack_patches([
                gt_viz, proposal_viz,
                score_viz, final_viz], 2, 2)

            if os.environ.get('DISPLAY', None):
                tpviz.interactive_imshow(viz)
            cv2.imwrite("{}/{:03d}.png".format(output_dir, idx), viz)
            pbar.update()

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def offline_evaluate(pred_func, output_file):
    df = get_eval_dataflow()
    all_results = eval_on_dataflow(
        df, lambda img: detect_one_image(img, pred_func))
    print(all_results)
#    input()
#    with open(output_file, 'w') as f:
#        json.dump(all_results, f, cls=MyEncoder)
    #print_evaluation_scores(output_file)

'''
 input: a list of dic
{
 'image': image_path
 'boxes': dic of list
 'masks': dic of list of numpy arrys
 'im_info':
'''
def generate_nii(pred_func, input_files):
    arg = []
    for idx,input_file in tqdm(enumerate(input_files)):
        boxes = collections.defaultdict(list)
        masks = collections.defaultdict(list)

        img = cv2.imread(input_file, cv2.IMREAD_COLOR)
        results = detect_one_image(img, pred_func)
        for res in results:
            cl = config.CLASS_NAMES[res.class_id]
            boxes[cl].append(list(res.box) + [res.score])
            masks[cl].append(res.mask)

        arg.append({
            'image': input_file,
            'boxes': boxes,
            'masks': masks,
            'im_info': (200,200)
        })

    evaluate_mask_mean_iou(arg)

def predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = detect_one_image(img, pred_func)
    final = draw_final_outputs(img, results)
    viz = np.concatenate((img, final), axis=1)
    tpviz.interactive_imshow(viz)

def predict_many(pred_func, input_files):
    if not os.path.exists('output'):
        os.mkdir('output')
    for idx,input_file in enumerate(input_files):
        img = cv2.imread(input_file, cv2.IMREAD_COLOR)
        results = detect_one_image(img, pred_func)
        final = draw_final_outputs(img, results)
        plt.imshow(final)
        plt.savefig(os.path.join('output', str(idx)+'.png'))

class EvalCallback(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image'],
            get_model_output_names())
        self.df = get_eval_dataflow()

    def _before_train(self):
        EVAL_TIMES = 5  # eval 5 times during training
        interval = self.trainer.max_epoch // (EVAL_TIMES + 1)
        self.epochs_to_eval = set([interval * k for k in range(1, EVAL_TIMES)])
        self.epochs_to_eval.add(self.trainer.max_epoch)

    def _eval(self):
        all_results = eval_on_dataflow(self.df, lambda img: detect_one_image(img, self.pred))
        output_file = os.path.join(
            logger.get_logger_dir(), 'outputs{}.json'.format(self.global_step))
        with open(output_file, 'w') as f:
            json.dump(all_results, f)

    def _trigger_epoch(self):
        if self.epoch_num in self.epochs_to_eval:
            self._eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--logdir', help='logdir', default='train_log/fastrcnn')
    parser.add_argument('--datadir', help='override config.BASEDIR')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--predict', help='path to the input image file')
    args = parser.parse_args()
    if args.datadir:
        config.BASEDIR = args.datadir

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.visualize or args.evaluate or args.predict:
        # autotune is too slow for inference
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

        assert args.load
        print_config()

        pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['image'],
            output_names=get_model_output_names()))

        imgs = OARDetection.load_many(config.BASEDIR, config.TEST_DATASET, add_gt=False)

        if args.visualize:
            imgs = [img['file_name'] for img in imgs]
            predict_many(pred, imgs)
        else:
            if args.evaluate:
                imgs = [img['file_name'] for img in imgs]
                generate_nii(pred, imgs)
    #                assert args.evaluate.endswith('.json')
    #                offline_evaluate(pred, args.evaluate)
            
            elif args.predict:
                #COCODetection(config.BASEDIR, 'train2014')   # to load the class names into caches
                predict(pred, args.predict)
    else:
        logger.set_logger_dir(args.logdir)
        print_config()
        stepnum = 300
        warmup_epoch = max(math.ceil(500.0 / stepnum), 5)
        factor = get_batch_factor()

        cfg = TrainConfig(
            model=Model(),
            data=QueueInput(get_train_dataflow(add_mask=config.MODE_MASK)),
            callbacks=[
                ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
                # linear warmup
                ScheduledHyperParamSetter(
                    'learning_rate',
                    [(0, 3e-3), (warmup_epoch * factor, 1e-2)], interp='linear'),
                # step decay
                ScheduledHyperParamSetter(
                    'learning_rate',
                    [(warmup_epoch * factor, 1e-2),
                     (150000 * factor // stepnum, 1e-3),
                     (230000 * factor // stepnum, 1e-4)]),
                EvalCallback(),
                GPUUtilizationTracker(),
            ],
            steps_per_epoch=stepnum,
            max_epoch=280000 * factor // stepnum,
            session_init=get_model_loader(args.load) if args.load else None,
        )
        trainer = SyncMultiGPUTrainerReplicated(get_nr_gpu())
        launch_train_with_config(cfg, trainer)
