#!/usr/bin/env python
# Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from multiprocessing import Pool
import pickle
import time

import numpy as np
import torch
from scipy.stats import norm
from collections import OrderedDict

import plotting as plg
import utils.model_utils as mutils


def get_mirrored_patch_crops(patch_crops, org_img_shape):
    mirrored_patch_crops = []
    mirrored_patch_crops.append([[org_img_shape[2] - ii[1], org_img_shape[2] - ii[0], ii[2], ii[3]]
                                 if len(ii) == 4 else [org_img_shape[2] - ii[1], org_img_shape[2] - ii[0], ii[2],
                                                       ii[3], ii[4], ii[5]]
                                 for ii in patch_crops])

    mirrored_patch_crops.append([[ii[0], ii[1], org_img_shape[3] - ii[3], org_img_shape[3] - ii[2]]
                                 if len(ii) == 4 else [ii[0], ii[1], org_img_shape[3] - ii[3],
                                                       org_img_shape[3] - ii[2], ii[4], ii[5]]
                                 for ii in patch_crops])

    mirrored_patch_crops.append([[org_img_shape[2] - ii[1],
                                  org_img_shape[2] - ii[0],
                                  org_img_shape[3] - ii[3],
                                  org_img_shape[3] - ii[2]]
                                 if len(ii) == 4 else
                                 [org_img_shape[2] - ii[1],
                                  org_img_shape[2] - ii[0],
                                  org_img_shape[3] - ii[3],
                                  org_img_shape[3] - ii[2], ii[4], ii[5]]
                                 for ii in patch_crops])

    return mirrored_patch_crops

def get_mirrored_patch_crops_ax_dep(patch_crops, org_img_shape, mirror_axes):
    mirrored_patch_crops = []
    for ax_ix, axes in enumerate(mirror_axes):
        if isinstance(axes, (int, float)) and int(axes) == 0:
            mirrored_patch_crops.append([[org_img_shape[2] - ii[1], org_img_shape[2] - ii[0], ii[2], ii[3]]
                                         if len(ii) == 4 else [org_img_shape[2] - ii[1], org_img_shape[2] - ii[0],
                                                               ii[2], ii[3], ii[4], ii[5]]
                                         for ii in patch_crops])
        elif isinstance(axes, (int, float)) and int(axes) == 1:
            mirrored_patch_crops.append([[ii[0], ii[1], org_img_shape[3] - ii[3], org_img_shape[3] - ii[2]]
                                         if len(ii) == 4 else [ii[0], ii[1], org_img_shape[3] - ii[3],
                                                               org_img_shape[3] - ii[2], ii[4], ii[5]]
                                         for ii in patch_crops])
        elif hasattr(axes, "__iter__") and (tuple(axes) == (0, 1) or tuple(axes) == (1, 0)):
            mirrored_patch_crops.append([[org_img_shape[2] - ii[1],
                                          org_img_shape[2] - ii[0],
                                          org_img_shape[3] - ii[3],
                                          org_img_shape[3] - ii[2]]
                                         if len(ii) == 4 else
                                         [org_img_shape[2] - ii[1],
                                          org_img_shape[2] - ii[0],
                                          org_img_shape[3] - ii[3],
                                          org_img_shape[3] - ii[2], ii[4], ii[5]]
                                         for ii in patch_crops])
        else:
            raise Exception("invalid mirror axes {} in get mirrored patch crops".format(axes))

    return mirrored_patch_crops

def apply_wbc_to_patient(inputs):
    """
    wrapper around prediction box consolidation: weighted box clustering (wbc). processes a single patient.
    loops over batch elements in patient results (1 in 3D, slices in 2D) and foreground classes,
    aggregates and stores results in new list.
    :return. patient_results_list: list over batch elements. each element is a list over boxes, where each box is
                                 one dictionary: [[box_0, ...], [box_n,...]]. batch elements are slices for 2D
                                 predictions, and a dummy batch dimension of 1 for 3D predictions.
    :return. pid: string. patient id.
    """
    regress_flag, in_patient_results_list, pid, class_dict, clustering_iou, n_ens = inputs
    out_patient_results_list = [[] for _ in range(len(in_patient_results_list))]

    for bix, b in enumerate(in_patient_results_list):

        for cl in list(class_dict.keys()):

            boxes = [(ix, box) for ix, box in enumerate(b) if
                     (box['box_type'] == 'det' and box['box_pred_class_id'] == cl)]
            box_coords = np.array([b[1]['box_coords'] for b in boxes])
            box_scores = np.array([b[1]['box_score'] for b in boxes])
            box_center_factor = np.array([b[1]['box_patch_center_factor'] for b in boxes])
            box_n_overlaps = np.array([b[1]['box_n_overlaps'] for b in boxes])
            try:
                box_patch_id = np.array([b[1]['patch_id'] for b in boxes])
            except KeyError: #backward compatibility for already saved pred results ... omg
                box_patch_id = np.array([b[1]['ens_ix'] for b in boxes])
            box_regressions = np.array([b[1]['regression'] for b in boxes]) if regress_flag else None
            box_rg_bins = np.array([b[1]['rg_bin'] if 'rg_bin' in b[1].keys() else float('NaN') for b in boxes])
            box_rg_uncs = np.array([b[1]['rg_uncertainty'] if 'rg_uncertainty' in b[1].keys() else float('NaN') for b in boxes])

            if 0 not in box_scores.shape:
                keep_scores, keep_coords, keep_n_missing, keep_regressions, keep_rg_bins, keep_rg_uncs = \
                    weighted_box_clustering(box_coords, box_scores, box_center_factor, box_n_overlaps, box_rg_bins, box_rg_uncs,
                                             box_regressions, box_patch_id, clustering_iou, n_ens)


                for boxix in range(len(keep_scores)):
                    clustered_box = {'box_type': 'det', 'box_coords': keep_coords[boxix],
                                     'box_score': keep_scores[boxix], 'cluster_n_missing': keep_n_missing[boxix],
                                     'box_pred_class_id': cl}
                    if regress_flag:
                        clustered_box.update({'regression': keep_regressions[boxix],
                                              'rg_uncertainty': keep_rg_uncs[boxix],
                                              'rg_bin': keep_rg_bins[boxix]})

                    out_patient_results_list[bix].append(clustered_box)

        # add gt boxes back to new output list.
        out_patient_results_list[bix].extend([box for box in b if box['box_type'] == 'gt'])

    return [out_patient_results_list, pid]


def weighted_box_clustering(box_coords, scores, box_pc_facts, box_n_ovs, box_rg_bins, box_rg_uncs,
                             box_regress, box_patch_id, thresh, n_ens):
    """Consolidates overlapping predictions resulting from patch overlaps, test data augmentations and temporal ensembling.
    clusters predictions together with iou > thresh (like in NMS). Output score and coordinate for one cluster are the
    average weighted by individual patch center factors (how trustworthy is this candidate measured by how centered
    its position within the patch is) and the size of the corresponding box.
    The number of expected predictions at a position is n_data_aug * n_temp_ens * n_overlaps_at_position
    (1 prediction per unique patch). Missing predictions at a cluster position are defined as the number of unique
    patches in the cluster, which did not contribute any predict any boxes.
    :param dets: (n_dets, (y1, x1, y2, x2, (z1), (z2), scores, box_pc_facts, box_n_ovs).
    :param box_coords: y1, x1, y2, x2, (z1), (z2).
    :param scores: confidence scores.
    :param box_pc_facts: patch-center factors from position on patch tiles.
    :param box_n_ovs: number of patch overlaps at box position.
    :param box_rg_bins: regression bin predictions.
    :param box_rg_uncs: (n_dets,) regression uncertainties (from model mrcnn_aleatoric).
    :param box_regress: (n_dets, n_regression_features).
    :param box_patch_id: ensemble index.
    :param thresh: threshold for iou_matching.
    :param n_ens: number of models, that are ensembled. (-> number of expected predictions per position).
    :return: keep_scores: (n_keep)  new scores of boxes to be kept.
    :return: keep_coords: (n_keep, (y1, x1, y2, x2, (z1), (z2)) new coordinates of boxes to be kept.
    """

    dim = 2 if box_coords.shape[1] == 4 else 3
    y1 = box_coords[:,0]
    x1 = box_coords[:,1]
    y2 = box_coords[:,2]
    x2 = box_coords[:,3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    if dim == 3:
        z1 = box_coords[:, 4]
        z2 = box_coords[:, 5]
        areas *= (z2 - z1 + 1)

    # order is the sorted index.  maps order to index o[1] = 24 (rank1, ix 24)
    order = scores.argsort()[::-1]

    keep_scores = []
    keep_coords = []
    keep_n_missing = []
    keep_regress = []
    keep_rg_bins = []
    keep_rg_uncs = []

    while order.size > 0:
        i = order[0]  # highest scoring element
        yy1 = np.maximum(y1[i], y1[order])
        xx1 = np.maximum(x1[i], x1[order])
        yy2 = np.minimum(y2[i], y2[order])
        xx2 = np.minimum(x2[i], x2[order])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h

        if dim == 3:
            zz1 = np.maximum(z1[i], z1[order])
            zz2 = np.minimum(z2[i], z2[order])
            d = np.maximum(0, zz2 - zz1 + 1)
            inter *= d

        # overlap between currently highest scoring box and all boxes.
        ovr = inter / (areas[i] + areas[order] - inter)
        ovr_fl = inter.astype('float64') / (areas[i] + areas[order] - inter.astype('float64'))
        assert np.all(ovr==ovr_fl), "ovr {}\n ovr_float {}".format(ovr, ovr_fl)
        # get all the predictions that match the current box to build one cluster.
        matches = np.nonzero(ovr > thresh)[0]

        match_n_ovs = box_n_ovs[order[matches]]
        match_pc_facts = box_pc_facts[order[matches]]
        match_patch_id = box_patch_id[order[matches]]
        match_ov_facts = ovr[matches]
        match_areas = areas[order[matches]]
        match_scores = scores[order[matches]]

        # weight all scores in cluster by patch factors, and size.
        match_score_weights = match_ov_facts * match_areas * match_pc_facts
        match_scores *= match_score_weights

        # for the weighted average, scores have to be divided by the number of total expected preds at the position
        # of the current cluster. 1 Prediction per patch is expected. therefore, the number of ensembled models is
        # multiplied by the mean overlaps of  patches at this position (boxes of the cluster might partly be
        # in areas of different overlaps).
        n_expected_preds = n_ens * np.mean(match_n_ovs)
        # the number of missing predictions is obtained as the number of patches,
        # which did not contribute any prediction to the current cluster.
        n_missing_preds = np.max((0, n_expected_preds - np.unique(match_patch_id).shape[0]))

        # missing preds are given the mean weighting
        # (expected prediction is the mean over all predictions in cluster).
        denom = np.sum(match_score_weights) + n_missing_preds * np.mean(match_score_weights)

        # compute weighted average score for the cluster
        avg_score = np.sum(match_scores) / denom

        # compute weighted average of coordinates for the cluster. now only take existing
        # predictions into account.
        avg_coords = [np.sum(y1[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(x1[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(y2[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(x2[order[matches]] * match_scores) / np.sum(match_scores)]

        if dim == 3:
            avg_coords.append(np.sum(z1[order[matches]] * match_scores) / np.sum(match_scores))
            avg_coords.append(np.sum(z2[order[matches]] * match_scores) / np.sum(match_scores))

        if box_regress is not None:
            # compute wt. avg. of regression vectors (component-wise average)
            avg_regress = np.sum(box_regress[order[matches]] * match_scores[:, np.newaxis], axis=0) / np.sum(
                match_scores)
            avg_rg_bins = np.round(np.sum(box_rg_bins[order[matches]] * match_scores) / np.sum(match_scores))
            avg_rg_uncs = np.sum(box_rg_uncs[order[matches]] * match_scores) / np.sum(match_scores)
        else:
            avg_regress = np.array(float('NaN'))
            avg_rg_bins = np.array(float('NaN'))
            avg_rg_uncs = np.array(float('NaN'))

        # some clusters might have very low scores due to high amounts of missing predictions.
        # filter out the with a conservative threshold, to speed up evaluation.
        if avg_score > 0.01:
            keep_scores.append(avg_score)
            keep_coords.append(avg_coords)
            keep_n_missing.append((n_missing_preds / n_expected_preds * 100))  # relative
            keep_regress.append(avg_regress)
            keep_rg_uncs.append(avg_rg_uncs)
            keep_rg_bins.append(avg_rg_bins)

        # get index of all elements that were not matched and discard all others.
        inds = np.nonzero(ovr <= thresh)[0]
        inds_where = np.where(ovr<=thresh)[0]
        assert np.all(inds == inds_where), "inds_nonzero {} \ninds_where {}".format(inds, inds_where)
        order = order[inds]

    return keep_scores, keep_coords, keep_n_missing, keep_regress, keep_rg_bins, keep_rg_uncs


def apply_nms_to_patient(inputs):

    in_patient_results_list, pid, class_dict, iou_thresh = inputs
    out_patient_results_list = []


    # collect box predictions over batch dimension (slices) and store slice info as slice_ids.
    for batch in in_patient_results_list:
        batch_el_boxes = []
        for cl in list(class_dict.keys()):
            det_boxes = [box for box in batch if (box['box_type'] == 'det' and box['box_pred_class_id'] == cl)]

            box_coords = np.array([box['box_coords'] for box in det_boxes])
            box_scores = np.array([box['box_score'] for box in det_boxes])
            if 0 not in box_scores.shape:
                keep_ix = mutils.nms_numpy(box_coords, box_scores, iou_thresh)
            else:
                keep_ix = []

            batch_el_boxes += [det_boxes[ix] for ix in keep_ix]

        batch_el_boxes += [box for box in batch if box['box_type'] == 'gt']
        out_patient_results_list.append(batch_el_boxes)

    assert len(in_patient_results_list) == len(out_patient_results_list), "batch dim needs to be maintained, in: {}, out {}".format(len(in_patient_results_list), len(out_patient_results_list))

    return [out_patient_results_list, pid]

def nms_2to3D(dets, thresh):
    """
    Merges 2D boxes to 3D cubes. For this purpose, boxes of all slices are regarded as lying in one slice.
    An adaptation of Non-maximum suppression is applied where clusters are found (like in NMS) with the extra constraint
    that suppressed boxes have to have 'connected' z coordinates w.r.t the core slice (cluster center, highest
    scoring box, the prevailing box). 'connected' z-coordinates are determined
    as the z-coordinates with predictions until the first coordinate for which no prediction is found.

    example: a cluster of predictions was found overlap > iou thresh in xy (like NMS). The z-coordinate of the highest
    scoring box is 50. Other predictions have 23, 46, 48, 49, 51, 52, 53, 56, 57.
    Only the coordinates connected with 50 are clustered to one cube: 48, 49, 51, 52, 53. (46 not because nothing was
    found in 47, so 47 is a 'hole', which interrupts the connection). Only the boxes corresponding to these coordinates
    are suppressed. All others are kept for building of further clusters.

    This algorithm works better with a certain min_confidence of predictions, because low confidence (e.g. noisy/cluttery)
    predictions can break the relatively strong assumption of defining cubes' z-boundaries at the first 'hole' in the cluster.

    :param dets: (n_detections, (y1, x1, y2, x2, scores, slice_id)
    :param thresh: iou matchin threshold (like in NMS).
    :return: keep: (n_keep,) 1D tensor of indices to be kept.
    :return: keep_z: (n_keep, [z1, z2]) z-coordinates to be added to boxes, which are kept in order to form cubes.
    """

    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    assert np.all(y1 <= y2) and np.all(x1 <= x2), """"the definition of the coordinates is crucially important here: 
        where maximum is taken needs to be the lower coordinate"""
    scores = dets[:, -2]
    slice_id = dets[:, -1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    keep_z = []

    while order.size > 0:  # order is the sorted index.  maps order to index: order[1] = 24 means (rank1, ix 24)
        i = order[0]  # highest scoring element
        yy1 = np.maximum(y1[i], y1[order])  # highest scoring element still in >order<, is compared to itself: okay?
        xx1 = np.maximum(x1[i], x1[order])
        yy2 = np.minimum(y2[i], y2[order])
        xx2 = np.minimum(x2[i], x2[order])

        h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1 + 1)
        inter = h * w

        iou = inter / (areas[i] + areas[order] - inter)
        matches = np.argwhere(
            iou > thresh)  # get all the elements that match the current box and have a lower score

        slice_ids = slice_id[order[matches]]
        core_slice = slice_id[int(i)]
        upper_holes = [ii for ii in np.arange(core_slice, np.max(slice_ids)) if ii not in slice_ids]
        lower_holes = [ii for ii in np.arange(np.min(slice_ids), core_slice) if ii not in slice_ids]
        max_valid_slice_id = np.min(upper_holes) if len(upper_holes) > 0 else np.max(slice_ids)
        min_valid_slice_id = np.max(lower_holes) if len(lower_holes) > 0 else np.min(slice_ids)
        z_matches = matches[(slice_ids <= max_valid_slice_id) & (slice_ids >= min_valid_slice_id)]

        # expand by one z voxel since box content is surrounded w/o overlap, i.e., z-content computed as z2-z1
        z1 = np.min(slice_id[order[z_matches]]) - 1
        z2 = np.max(slice_id[order[z_matches]]) + 1

        keep.append(i)
        keep_z.append([z1, z2])
        order = np.delete(order, z_matches, axis=0)

    return keep, keep_z

def apply_2d_3d_merging_to_patient(inputs):
    """
    wrapper around 2Dto3D merging operation. Processes a single patient. Takes 2D patient results (slices in batch dimension)
    and returns 3D patient results (dummy batch dimension of 1). Applies an adaption of Non-Maximum Surpression
    (Detailed methodology is described in nms_2to3D).
    :return. results_dict_boxes: list over batch elements (1 in 3D). each element is a list over boxes, where each box is
                                 one dictionary: [[box_0, ...], [box_n,...]].
    :return. pid: string. patient id.
    """

    in_patient_results_list, pid, class_dict, merge_3D_iou = inputs
    out_patient_results_list = []

    for cl in list(class_dict.keys()):
        det_boxes, slice_ids = [], []
        # collect box predictions over batch dimension (slices) and store slice info as slice_ids.
        for batch_ix, batch in enumerate(in_patient_results_list):
            batch_element_det_boxes = [(ix, box) for ix, box in enumerate(batch) if
                                       (box['box_type'] == 'det' and box['box_pred_class_id'] == cl)]
            det_boxes += batch_element_det_boxes
            slice_ids += [batch_ix] * len(batch_element_det_boxes)

        box_coords = np.array([batch[1]['box_coords'] for batch in det_boxes])
        box_scores = np.array([batch[1]['box_score'] for batch in det_boxes])
        slice_ids = np.array(slice_ids)

        if 0 not in box_scores.shape:
            keep_ix, keep_z = nms_2to3D(
                np.concatenate((box_coords, box_scores[:, None], slice_ids[:, None]), axis=1), merge_3D_iou)
        else:
            keep_ix, keep_z = [], []

        # store kept predictions in new results list and add corresponding z-dimension info to coordinates.
        for kix, kz in zip(keep_ix, keep_z):
            keep_box = det_boxes[kix][1]
            keep_box['box_coords'] = list(keep_box['box_coords']) + kz
            out_patient_results_list.append(keep_box)

    gt_boxes = [box for b in in_patient_results_list for box in b if box['box_type'] == 'gt']
    if len(gt_boxes) > 0:
        assert np.all([len(box["box_coords"]) == 6 for box in gt_boxes]), "expanded preds to 3D but GT is 2D."
    out_patient_results_list += gt_boxes

    return [[out_patient_results_list], pid]  # additional list wrapping is extra batch dim.


class Predictor:
    """
	    Prediction pipeline:
	    - receives a patched patient image (n_patches, c, y, x, (z)) from patient data loader.
	    - forwards patches through model in chunks of batch_size. (method: batch_tiling_forward)
	    - unmolds predictions (boxes and segmentations) to original patient coordinates. (method: spatial_tiling_forward)

	    Ensembling (mode == 'test'):
	    - for inference, forwards 4 mirrored versions of image to through model and unmolds predictions afterwards
	      accordingly (method: data_aug_forward)
	    - for inference, loads multiple parameter-sets of the trained model corresponding to different epochs. for each
	      parameter-set loops over entire test set, runs prediction pipeline for each patient. (method: predict_test_set)

	    Consolidation of predictions:
	    - consolidates a patient's predictions (boxes, segmentations) collected over patches, data_aug- and temporal ensembling,
	      performs clustering and weighted averaging (external function: apply_wbc_to_patient) to obtain consistent outptus.
	    - for 2D networks, consolidates box predictions to 3D cubes via clustering (adaption of non-maximum surpression).
	      (external function: apply_2d_3d_merging_to_patient)

	    Ground truth handling:
	    - dissmisses any ground truth boxes returned by the model (happens in validation mode, patch-based groundtruth)
	    - if provided by data loader, adds patient-wise ground truth to the final predictions to be passed to the evaluator.
    """
    def __init__(self, cf, net, logger, mode):

        self.cf = cf
        self.batch_size = cf.batch_size
        self.logger = logger
        self.mode = mode
        self.net = net
        self.n_ens = 1
        self.rank_ix = '0'
        self.regress_flag = any(['regression' in task for task in self.cf.prediction_tasks])

        if self.cf.merge_2D_to_3D_preds:
            assert self.cf.dim == 2, "Merge 2Dto3D only valid for 2D preds, but current dim is {}.".format(self.cf.dim)

        if self.mode == 'test':
            try:
                self.epoch_ranking = np.load(os.path.join(self.cf.fold_dir, 'epoch_ranking.npy'))[:cf.test_n_epochs]
            except:
                raise RuntimeError('no epoch ranking file in fold directory. '
                                   'seems like you are trying to run testing without prior training...')
            self.n_ens = cf.test_n_epochs
            if self.cf.test_aug_axes is not None:
                self.n_ens *= (len(self.cf.test_aug_axes)+1)
            self.example_plot_dir = os.path.join(cf.test_dir, "example_plots")
            os.makedirs(self.example_plot_dir, exist_ok=True)

    def batch_tiling_forward(self, batch):
        """
        calls the actual network forward method. in patch-based prediction, the batch dimension might be overladed
        with n_patches >> batch_size, which would exceed gpu memory. In this case, batches are processed in chunks of
        batch_size. validation mode calls the train method to monitor losses (returned ground truth objects are discarded).
        test mode calls the test forward method, no ground truth required / involved.
        :return. results_dict: stores the results for one patient. dictionary with keys:
                 - 'boxes': list over batch elements. each element is a list over boxes, where each box is
                            one dictionary: [[box_0, ...], [box_n,...]]. batch elements are slices for 2D predictions,
                            and a dummy batch dimension of 1 for 3D predictions.
                 - 'seg_preds': pixel-wise predictions. (b, 1, y, x, (z))
                 - loss / class_loss (only in validation mode)
        """

        img = batch['data']

        if img.shape[0] <= self.batch_size:

            if self.mode == 'val':
                # call training method to monitor losses
                results_dict = self.net.train_forward(batch, is_validation=True)
                # discard returned ground-truth boxes (also training info boxes).
                results_dict['boxes'] = [[box for box in b if box['box_type'] == 'det'] for b in results_dict['boxes']]
            elif self.mode == 'test':
                results_dict = self.net.test_forward(batch, return_masks=self.cf.return_masks_in_test)

        else: # needs batch tiling
            split_ixs = np.split(np.arange(img.shape[0]), np.arange(img.shape[0])[::self.batch_size])
            chunk_dicts = []
            for chunk_ixs in split_ixs[1:]:  # first split is elements before 0, so empty
                b = {k: batch[k][chunk_ixs] for k in batch.keys()
                     if (isinstance(batch[k], np.ndarray) and batch[k].shape[0] == img.shape[0])}
                if self.mode == 'val':
                    chunk_dicts += [self.net.train_forward(b, is_validation=True)]
                else:
                    chunk_dicts += [self.net.test_forward(b, return_masks=self.cf.return_masks_in_test)]

            results_dict = {}
            # flatten out batch elements from chunks ([chunk, chunk] -> [b, b, b, b, ...])
            results_dict['boxes'] = [item for d in chunk_dicts for item in d['boxes']]
            results_dict['seg_preds'] = np.array([item for d in chunk_dicts for item in d['seg_preds']])

            if self.mode == 'val':
                # if hasattr(self.cf, "losses_to_monitor"):
                #     loss_names = self.cf.losses_to_monitor
                # else:
                #     loss_names = {name for dic in chunk_dicts for name in dic if 'loss' in name}
                # estimate patient loss by mean over batch_chunks. Most similar to training loss.
                results_dict['torch_loss'] = torch.mean(torch.cat([d['torch_loss'] for d in chunk_dicts]))
                results_dict['class_loss'] = np.mean([d['class_loss'] for d in chunk_dicts])
                # discard returned ground-truth boxes (also training info boxes).
                results_dict['boxes'] = [[box for box in b if box['box_type'] == 'det'] for b in results_dict['boxes']]

        return results_dict

    def spatial_tiling_forward(self, batch, patch_crops = None, n_aug='0'):
        """
        forwards batch to batch_tiling_forward method and receives and returns a dictionary with results.
        if patch-based prediction, the results received from batch_tiling_forward will be on a per-patch-basis.
        this method uses the provided patch_crops to re-transform all predictions to whole-image coordinates.
        Patch-origin information of all box-predictions will be needed for consolidation, hence it is stored as
        'patch_id', which is a unique string for each patch (also takes current data aug and temporal epoch instances
        into account). all box predictions get additional information about the amount overlapping patches at the
        respective position (used for consolidation).
        :return. results_dict: stores the results for one patient. dictionary with keys:
                 - 'boxes': list over batch elements. each element is a list over boxes, where each box is
                            one dictionary: [[box_0, ...], [box_n,...]]. batch elements are slices for 2D predictions,
                            and a dummy batch dimension of 1 for 3D predictions.
                 - 'seg_preds': pixel-wise predictions. (b, 1, y, x, (z))
                 - monitor_values (only in validation mode)
        returned dict is a flattened version with 1 batch instance (3D) or slices (2D)
        """

        if patch_crops is not None:
            #print("patch_crops not None, applying patch center factor")

            patches_dict = self.batch_tiling_forward(batch)
            results_dict = {'boxes': [[] for _ in range(batch['original_img_shape'][0])]}
            #bc of ohe--> channel dim of seg has size num_classes
            out_seg_shape = list(batch['original_img_shape'])
            out_seg_shape[1] = patches_dict["seg_preds"].shape[1]
            out_seg_preds = np.zeros(out_seg_shape, dtype=np.float16)
            patch_overlap_map = np.zeros_like(out_seg_preds, dtype='uint8')
            for pix, pc in enumerate(patch_crops):
                if self.cf.dim == 3:
                    out_seg_preds[:, :, pc[0]:pc[1], pc[2]:pc[3], pc[4]:pc[5]] += patches_dict['seg_preds'][pix]
                    patch_overlap_map[:, :, pc[0]:pc[1], pc[2]:pc[3], pc[4]:pc[5]] += 1
                elif self.cf.dim == 2:
                    out_seg_preds[pc[4]:pc[5], :, pc[0]:pc[1], pc[2]:pc[3], ] += patches_dict['seg_preds'][pix]
                    patch_overlap_map[pc[4]:pc[5], :, pc[0]:pc[1], pc[2]:pc[3], ] += 1

            out_seg_preds[patch_overlap_map > 0] /= patch_overlap_map[patch_overlap_map > 0]
            results_dict['seg_preds'] = out_seg_preds

            for pix, pc in enumerate(patch_crops):
                patch_boxes = patches_dict['boxes'][pix]
                for box in patch_boxes:

                    # add unique patch id for consolidation of predictions.
                    box['patch_id'] = self.rank_ix + '_' + n_aug + '_' + str(pix)
                    # boxes from the edges of a patch have a lower prediction quality, than the ones at patch-centers.
                    # hence they will be down-weighted for consolidation, using the 'box_patch_center_factor', which is
                    # obtained by a gaussian distribution over positions in the patch and average over spatial dimensions.
                    # Also the info 'box_n_overlaps' is stored for consolidation, which represents the amount of
                    # overlapping patches at the box's position.

                    c = box['box_coords']
                    #box_centers = np.array([(c[ii] + c[ii+2])/2 for ii in range(len(c)//2)])
                    box_centers = [(c[ii] + c[ii + 2]) / 2 for ii in range(2)]
                    if self.cf.dim == 3:
                        box_centers.append((c[4] + c[5]) / 2)
                    box['box_patch_center_factor'] = np.mean(
                        [norm.pdf(bc, loc=pc, scale=pc * 0.8) * np.sqrt(2 * np.pi) * pc * 0.8 for bc, pc in
                         zip(box_centers, np.array(self.cf.patch_size) / 2)])
                    if self.cf.dim == 3:
                        c += np.array([pc[0], pc[2], pc[0], pc[2], pc[4], pc[4]])
                        int_c = [int(np.floor(ii)) if ix%2 == 0 else int(np.ceil(ii))  for ix, ii in enumerate(c)]
                        box['box_n_overlaps'] = np.mean(patch_overlap_map[:, :, int_c[1]:int_c[3], int_c[0]:int_c[2], int_c[4]:int_c[5]])
                        results_dict['boxes'][0].append(box)
                    else:
                        c += np.array([pc[0], pc[2], pc[0], pc[2]])
                        int_c = [int(np.floor(ii)) if ix % 2 == 0 else int(np.ceil(ii)) for ix, ii in enumerate(c)]
                        box['box_n_overlaps'] = np.mean(
                            patch_overlap_map[pc[4], :, int_c[1]:int_c[3], int_c[0]:int_c[2]])
                        results_dict['boxes'][pc[4]].append(box)

            if self.mode == 'val':
                results_dict['torch_loss'] = patches_dict['torch_loss']
                results_dict['class_loss'] = patches_dict['class_loss']

        else:
            results_dict = self.batch_tiling_forward(batch)
            for b in results_dict['boxes']:
                for box in b:
                    box['box_patch_center_factor'] = 1
                    box['box_n_overlaps'] = 1
                    box['patch_id'] = self.rank_ix + '_' + n_aug

        return results_dict

    def data_aug_forward(self, batch):
        """
        in val_mode: passes batch through to spatial_tiling method without data_aug.
        in test_mode: if cf.test_aug is set in configs, createst 4 mirrored versions of the input image,
        passes all of them to the next processing step (spatial_tiling method) and re-transforms returned predictions
        to original image version.
        :return. results_dict: stores the results for one patient. dictionary with keys:
                 - 'boxes': list over batch elements. each element is a list over boxes, where each box is
                            one dictionary: [[box_0, ...], [box_n,...]]. batch elements are slices for 2D predictions,
                            and a dummy batch dimension of 1 for 3D predictions.
                 - 'seg_preds': pixel-wise predictions. (b, 1, y, x, (z))
                 - loss / class_loss (only in validation mode)
        """
        patch_crops = batch['patch_crop_coords'] if self.patched_patient else None
        results_list = [self.spatial_tiling_forward(batch, patch_crops)]
        org_img_shape = batch['original_img_shape']

        if self.mode == 'test' and self.cf.test_aug_axes is not None:
            if isinstance(self.cf.test_aug_axes, (int, float)):
                self.cf.test_aug_axes = (self.cf.test_aug_axes,)
            #assert np.all(np.array(self.cf.test_aug_axes)<self.cf.dim), "test axes {} need to be spatial axes".format(self.cf.test_aug_axes)

            if self.patched_patient:
                # apply mirror transformations to patch-crop coordinates, for correct tiling in spatial_tiling method.
                mirrored_patch_crops = get_mirrored_patch_crops_ax_dep(patch_crops, batch['original_img_shape'],
                                                                       self.cf.test_aug_axes)
                self.logger.info("mirrored patch crop coords for patched patient in test augs!")
            else:
                mirrored_patch_crops = [None] * 3

            img = np.copy(batch['data'])

            for n_aug, sp_axis in enumerate(self.cf.test_aug_axes):
                #sp_axis = np.array(axis) #-2 #spatial axis index
                axis = np.array(sp_axis)+2
                if isinstance(sp_axis, (int, float)):
                    # mirroring along one axis at a time
                    batch['data'] = np.flip(img, axis=axis).copy()
                    chunk_dict = self.spatial_tiling_forward(batch, mirrored_patch_crops[n_aug], n_aug=str(n_aug))
                    # re-transform coordinates.
                    for ix in range(len(chunk_dict['boxes'])):
                        for boxix in range(len(chunk_dict['boxes'][ix])):
                            coords = chunk_dict['boxes'][ix][boxix]['box_coords'].copy()
                            coords[sp_axis] = org_img_shape[axis] - chunk_dict['boxes'][ix][boxix]['box_coords'][sp_axis+2]
                            coords[sp_axis+2] = org_img_shape[axis] - chunk_dict['boxes'][ix][boxix]['box_coords'][sp_axis]
                            assert coords[2] >= coords[0], [coords, chunk_dict['boxes'][ix][boxix]['box_coords']]
                            assert coords[3] >= coords[1], [coords, chunk_dict['boxes'][ix][boxix]['box_coords']]
                            chunk_dict['boxes'][ix][boxix]['box_coords'] = coords
                    # re-transform segmentation predictions.
                    chunk_dict['seg_preds'] = np.flip(chunk_dict['seg_preds'], axis=axis)

                elif hasattr(sp_axis, "__iter__") and tuple(sp_axis)==(0,1) or tuple(sp_axis)==(1,0):
                    #NEED: mirrored patch crops are given as [(y-axis), (x-axis), (y-,x-axis)], obey this order!
                    # mirroring along two axes at same time
                    batch['data'] = np.flip(np.flip(img, axis=axis[0]), axis=axis[1]).copy()
                    chunk_dict = self.spatial_tiling_forward(batch, mirrored_patch_crops[n_aug], n_aug=str(n_aug))
                    # re-transform coordinates.
                    for ix in range(len(chunk_dict['boxes'])):
                        for boxix in range(len(chunk_dict['boxes'][ix])):
                            coords = chunk_dict['boxes'][ix][boxix]['box_coords'].copy()
                            coords[sp_axis[0]] = org_img_shape[axis[0]] - chunk_dict['boxes'][ix][boxix]['box_coords'][sp_axis[0]+2]
                            coords[sp_axis[0]+2] = org_img_shape[axis[0]] - chunk_dict['boxes'][ix][boxix]['box_coords'][sp_axis[0]]
                            coords[sp_axis[1]] = org_img_shape[axis[1]] - chunk_dict['boxes'][ix][boxix]['box_coords'][sp_axis[1]+2]
                            coords[sp_axis[1]+2] = org_img_shape[axis[1]] - chunk_dict['boxes'][ix][boxix]['box_coords'][sp_axis[1]]
                            assert coords[2] >= coords[0], [coords, chunk_dict['boxes'][ix][boxix]['box_coords']]
                            assert coords[3] >= coords[1], [coords, chunk_dict['boxes'][ix][boxix]['box_coords']]
                            chunk_dict['boxes'][ix][boxix]['box_coords'] = coords
                    # re-transform segmentation predictions.
                    chunk_dict['seg_preds'] = np.flip(np.flip(chunk_dict['seg_preds'], axis=axis[0]), axis=axis[1]).copy()

                else:
                    raise Exception("Invalid axis type {} in test augs".format(type(axis)))
                results_list.append(chunk_dict)

            batch['data'] = img

        # aggregate all boxes/seg_preds per batch element from data_aug predictions.
        results_dict = {}
        results_dict['boxes'] = [[item for d in results_list for item in d['boxes'][batch_instance]]
                                 for batch_instance in range(org_img_shape[0])]
        # results_dict['seg_preds'] = np.array([[item for d in results_list for item in d['seg_preds'][batch_instance]]
        #                                       for batch_instance in range(org_img_shape[0])])
        results_dict['seg_preds'] = np.stack([dic['seg_preds'] for dic in results_list], axis=1)
        # needs segs probs in seg_preds entry:
        results_dict['seg_preds'] = np.sum(results_dict['seg_preds'], axis=1) #add up seg probs from different augs per class

        if self.mode == 'val':
            results_dict['torch_loss'] = results_list[0]['torch_loss']
            results_dict['class_loss'] = results_list[0]['class_loss']

        return results_dict

    def load_saved_predictions(self):
        """loads raw predictions saved by self.predict_test_set. aggregates and/or merges 2D boxes to 3D cubes for
            evaluation (if model predicts 2D but evaluation is run in 3D), according to settings config.
        :return: list_of_results_per_patient: list over patient results. each entry is a dict with keys:
            - 'boxes': list over batch elements. each element is a list over boxes, where each box is
                       one dictionary: [[box_0, ...], [box_n,...]]. batch elements are slices for 2D predictions
                       (if not merged to 3D), and a dummy batch dimension of 1 for 3D predictions.
            - 'batch_dices': dice scores as recorded in raw prediction results.
            - 'seg_preds': not implemented yet. could replace dices by seg preds to have raw seg info available, however
                would consume critically large memory amount. todo evaluation of instance/semantic segmentation.
        """

        results_file = 'pred_results.pkl' if not self.cf.held_out_test_set else 'pred_results_held_out.pkl'
        if not self.cf.held_out_test_set or self.cf.eval_test_fold_wise:
            self.logger.info("loading saved predictions of fold {}".format(self.cf.fold))
            with open(os.path.join(self.cf.fold_dir, results_file), 'rb') as handle:
                results_list = pickle.load(handle)
            box_results_list = [(res_dict["boxes"], pid) for res_dict, pid in results_list]

            da_factor = len(self.cf.test_aug_axes)+1 if self.cf.test_aug_axes is not None else 1
            self.n_ens = self.cf.test_n_epochs * da_factor
            self.logger.info('loaded raw test set predictions with n_patients = {} and n_ens = {}'.format(
                len(results_list), self.n_ens))
        else:
            self.logger.info("loading saved predictions of hold-out test set")
            fold_dirs = sorted([os.path.join(self.cf.exp_dir, f) for f in os.listdir(self.cf.exp_dir) if
                                os.path.isdir(os.path.join(self.cf.exp_dir, f)) and f.startswith("fold")])

            results_list = []
            folds_loaded = 0
            for fold in range(self.cf.n_cv_splits):
                fold_dir = os.path.join(self.cf.exp_dir, 'fold_{}'.format(fold))
                if fold_dir in fold_dirs:
                    with open(os.path.join(fold_dir, results_file), 'rb') as handle:
                        fold_list = pickle.load(handle)
                        results_list += fold_list
                        folds_loaded += 1
                else:
                    self.logger.info("Skipping fold {} since no saved predictions found.".format(fold))
            box_results_list = []
            for res_dict, pid in results_list: #without filtering gt out:
                box_results_list.append((res_dict['boxes'], pid))
                #it's usually not right to filter out gts here, is it?

            da_factor = len(self.cf.test_aug_axes)+1 if self.cf.test_aug_axes is not None else 1
            self.n_ens = self.cf.test_n_epochs * da_factor * folds_loaded

        # -------------- aggregation of boxes via clustering -----------------

        if self.cf.clustering == "wbc":
            self.logger.info('applying WBC to test-set predictions with iou {} and n_ens {} over {} patients'.format(
                self.cf.clustering_iou, self.n_ens, len(box_results_list)))

            mp_inputs = [[self.regress_flag, ii[0], ii[1], self.cf.class_dict, self.cf.clustering_iou, self.n_ens] for ii
                         in box_results_list]
            del box_results_list
            pool = Pool(processes=self.cf.n_workers)
            box_results_list = pool.map(apply_wbc_to_patient, mp_inputs, chunksize=1)
            pool.close()
            pool.join()
            del mp_inputs
        elif self.cf.clustering == "nms":
            self.logger.info('applying standard NMS to test-set predictions with iou {} over {} patients.'.format(
                self.cf.clustering_iou, len(box_results_list)))
            pool = Pool(processes=self.cf.n_workers)
            mp_inputs = [[ii[0], ii[1], self.cf.class_dict, self.cf.clustering_iou] for ii in box_results_list]
            del box_results_list
            box_results_list = pool.map(apply_nms_to_patient, mp_inputs, chunksize=1)
            pool.close()
            pool.join()
            del mp_inputs

        if self.cf.merge_2D_to_3D_preds:
            self.logger.info('applying 2Dto3D merging to test-set predictions with iou = {}.'.format(self.cf.merge_3D_iou))
            pool = Pool(processes=self.cf.n_workers)
            mp_inputs = [[ii[0], ii[1], self.cf.class_dict, self.cf.merge_3D_iou] for ii in box_results_list]
            box_results_list = pool.map(apply_2d_3d_merging_to_patient, mp_inputs, chunksize=1)
            pool.close()
            pool.join()
            del mp_inputs

        for ix in range(len(results_list)):
            assert np.all(results_list[ix][1] == box_results_list[ix][1]), "pid mismatch between loaded and aggregated results"
            results_list[ix][0]["boxes"] = box_results_list[ix][0]

        return results_list # holds (results_dict, pid)

    def predict_patient(self, batch):
        """
        predicts one patient.
        called either directly via loop over validation set in exec.py (mode=='val')
        or from self.predict_test_set (mode=='test).
        in val mode:  adds 3D ground truth info to predictions and runs consolidation and 2Dto3D merging of predictions.
        in test mode: returns raw predictions (ground truth addition, consolidation, 2D to 3D merging are
                      done in self.predict_test_set, because patient predictions across several epochs might be needed
                      to be collected first, in case of temporal ensembling).
        :return. results_dict: stores the results for one patient. dictionary with keys:
                 - 'boxes': list over batch elements. each element is a list over boxes, where each box is
                            one dictionary: [[box_0, ...], [box_n,...]]. batch elements are slices for 2D predictions
                            (if not merged to 3D), and a dummy batch dimension of 1 for 3D predictions.
                 - 'seg_preds': pixel-wise predictions. (b, 1, y, x, (z))
                 - loss / class_loss (only in validation mode)
        """
        if self.mode=="test":
            self.logger.info('predicting patient {} for fold {} '.format(np.unique(batch['pid']), self.cf.fold))

        # True if patient is provided in patches and predictions need to be tiled.
        self.patched_patient = 'patch_crop_coords' in list(batch.keys())

        # forward batch through prediction pipeline.
        results_dict = self.data_aug_forward(batch)
        #has seg probs in entry 'seg_preds'

        if self.mode == 'val':
            for b in range(batch['patient_bb_target'].shape[0]):
                for t in range(len(batch['patient_bb_target'][b])):
                    gt_box = {'box_type': 'gt', 'box_coords': batch['patient_bb_target'][b][t],
                              'class_targets': batch['patient_class_targets'][b][t]}
                    for name in self.cf.roi_items:
                        gt_box.update({name : batch['patient_'+name][b][t]})
                    results_dict['boxes'][b].append(gt_box)

            if 'dice' in self.cf.metrics:
                if self.patched_patient:
                    assert 'patient_seg' in batch.keys(), "Results_dict preds are in original patient shape."
                results_dict['batch_dices'] = mutils.dice_per_batch_and_class(
                    results_dict['seg_preds'], batch["patient_seg"] if self.patched_patient else batch['seg'],
                    self.cf.num_seg_classes, convert_to_ohe=True)
            if self.patched_patient and self.cf.clustering == "wbc":
                wbc_input = [self.regress_flag, results_dict['boxes'], 'dummy_pid', self.cf.class_dict, self.cf.clustering_iou, self.n_ens]
                results_dict['boxes'] = apply_wbc_to_patient(wbc_input)[0]
            elif self.patched_patient:
                nms_inputs = [results_dict['boxes'], 'dummy_pid', self.cf.class_dict, self.cf.clustering_iou]
                results_dict['boxes'] = apply_nms_to_patient(nms_inputs)[0]

            if self.cf.merge_2D_to_3D_preds:
                results_dict['2D_boxes'] = results_dict['boxes']
                merge_dims_inputs = [results_dict['boxes'], 'dummy_pid', self.cf.class_dict, self.cf.merge_3D_iou]
                results_dict['boxes'] = apply_2d_3d_merging_to_patient(merge_dims_inputs)[0]

        return results_dict

    def predict_test_set(self, batch_gen, return_results=True):
        """
        wrapper around test method, which loads multiple (or one) epoch parameters (temporal ensembling), loops through
        the test set and collects predictions per patient. Also flattens the results per patient and epoch
        and adds optional ground truth boxes for evaluation. Saves out the raw result list for later analysis and
        optionally consolidates and returns predictions immediately.
        :return: (optionally) list_of_results_per_patient: list over patient results. each entry is a dict with keys:
                 - 'boxes': list over batch elements. each element is a list over boxes, where each box is
                            one dictionary: [[box_0, ...], [box_n,...]]. batch elements are slices for 2D predictions
                            (if not merged to 3D), and a dummy batch dimension of 1 for 3D predictions.
                 - 'seg_preds': not implemented yet. todo evaluation of instance/semantic segmentation.
        """

        # -------------- raw predicting -----------------
        dict_of_patients_results = OrderedDict()
        set_of_result_types = set()
        # get paths of all parameter sets to be loaded for temporal ensembling. (or just one for no temp. ensembling).
        weight_paths = [os.path.join(self.cf.fold_dir, '{}_best_params.pth'.format(epoch)) for epoch in self.epoch_ranking]


        for rank_ix, weight_path in enumerate(weight_paths):
            self.logger.info(('tmp ensembling over rank_ix:{} epoch:{}'.format(rank_ix, weight_path)))
            self.net.load_state_dict(torch.load(weight_path))
            self.net.eval()
            self.rank_ix = str(rank_ix)
            with torch.no_grad():
                plot_batches = np.random.choice(np.arange(batch_gen['n_test']), size=self.cf.n_test_plots, replace=False)
                for i in range(batch_gen['n_test']):
                    batch = next(batch_gen['test'])
                    pid = np.unique(batch['pid'])
                    assert len(pid)==1
                    pid = pid[0]

                    if not pid in dict_of_patients_results.keys():  # store batch info in patient entry of results dict.
                        dict_of_patients_results[pid] = {}
                        dict_of_patients_results[pid]['results_dicts'] = []
                        dict_of_patients_results[pid]['patient_bb_target'] = batch['patient_bb_target']

                        for name in self.cf.roi_items:
                            dict_of_patients_results[pid]["patient_"+name] = batch["patient_"+name]
                    stime = time.time()
                    results_dict = self.predict_patient(batch) #only holds "boxes", "seg_preds"
                    # needs ohe seg probs in seg_preds entry:
                    results_dict['seg_preds'] = np.argmax(results_dict['seg_preds'], axis=1)[:,np.newaxis]
                    self.logger.info("predicting patient {} with weight rank {} (progress: {}/{}) took {:.2f}s".format(
                        str(pid), rank_ix, (rank_ix)*batch_gen['n_test']+(i+1), len(weight_paths)*batch_gen['n_test'], time.time()-stime))

                    if i in plot_batches and (not self.patched_patient or 'patient_data' in batch.keys()):
                        try:
                            # view qualitative results of random test case
                            self.logger.time("test_plot")
                            out_file = os.path.join(self.example_plot_dir,
                                                    'batch_example_test_{}_rank_{}.png'.format(self.cf.fold, rank_ix))
                            plg.view_batch(self.cf, batch, res_dict=results_dict, out_file=out_file,
                                           show_seg_ids='dice' in self.cf.metrics,
                                           has_colorchannels=self.cf.has_colorchannels, show_gt_labels=True)
                            self.logger.info("generated example test plot {} in {:.2f}s".format(os.path.basename(out_file), self.logger.time("test_plot")))
                        except Exception as e:
                            self.logger.info("WARNING: error in view_batch: {}".format(e))

                    if 'dice' in self.cf.metrics:
                        if self.patched_patient:
                            assert 'patient_seg' in batch.keys(), "Results_dict preds are in original patient shape."
                        results_dict['batch_dices'] = mutils.dice_per_batch_and_class( results_dict['seg_preds'],
                                batch["patient_seg"] if self.patched_patient else batch['seg'],
                                self.cf.num_seg_classes, convert_to_ohe=True)

                    dict_of_patients_results[pid]['results_dicts'].append({k:v for k,v in results_dict.items()
                                                                           if k in ["boxes", "batch_dices"]})
                    # collect result types to know which ones to look for when saving
                    set_of_result_types.update(dict_of_patients_results[pid]['results_dicts'][-1].keys())



        # -------------- re-order, save raw results -----------------
        self.logger.info('finished predicting test set. starting aggregation of predictions.')
        results_per_patient = []
        for pid, p_dict in dict_of_patients_results.items():
        # dict_of_patients_results[pid]['results_list'] has length batch['n_test']

            results_dict = {}
            # collect all boxes/seg_preds of same batch_instance over temporal instances.
            b_size = len(p_dict['results_dicts'][0]["boxes"])
            for res_type in [rtype for rtype in set_of_result_types if rtype in ["boxes", "batch_dices"]]:#, "seg_preds"]]:
                if not 'batch' in res_type: #assume it's results on batch-element basis
                    results_dict[res_type] = [[item for rank_dict in p_dict['results_dicts'] for item in rank_dict[res_type][batch_instance]]
                                             for batch_instance in range(b_size)]
                else:
                    results_dict[res_type] = []
                    for dict in p_dict['results_dicts']:
                        if 'dice' in res_type:
                            item = dict[res_type] #dict['batch_dices'] has shape (num_seg_classes,)
                            assert len(item) == self.cf.num_seg_classes, \
                                "{}, {}".format(len(item), self.cf.num_seg_classes)
                        else:
                            raise NotImplementedError
                        results_dict[res_type].append(item)
                    # rdict[dice] shape (n_rank_epochs (n_saved_ranks), nsegclasses)
                    # calc mean over test epochs so inline with shape from sampling
                    results_dict[res_type] = np.mean(results_dict[res_type], axis=0) #maybe error type with other than dice

            if not hasattr(self.cf, "eval_test_separately") or not self.cf.eval_test_separately:
                # add unpatched 2D or 3D (if dim==3 or merge_2D_to_3D) ground truth boxes for evaluation.
                for b in range(p_dict['patient_bb_target'].shape[0]):
                    for targ in range(len(p_dict['patient_bb_target'][b])):
                        gt_box = {'box_type': 'gt', 'box_coords':p_dict['patient_bb_target'][b][targ],
                                  'class_targets': p_dict['patient_class_targets'][b][targ]}
                        for name in self.cf.roi_items:
                            gt_box.update({name: p_dict["patient_"+name][b][targ]})
                        results_dict['boxes'][b].append(gt_box)

            results_per_patient.append([results_dict, pid])

        out_string = 'pred_results_held_out' if self.cf.held_out_test_set else 'pred_results'
        with open(os.path.join(self.cf.fold_dir, '{}.pkl'.format(out_string)), 'wb') as handle:
            pickle.dump(results_per_patient, handle)

        if return_results:
            # -------------- results processing, clustering, etc. -----------------
            final_patient_box_results = [ (res_dict["boxes"], pid) for res_dict,pid in results_per_patient ]
            if self.cf.clustering == "wbc":
                self.logger.info('applying WBC to test-set predictions with iou = {} and n_ens = {}.'.format(
                    self.cf.clustering_iou, self.n_ens))
                mp_inputs = [[self.regress_flag, ii[0], ii[1], self.cf.class_dict, self.cf.clustering_iou, self.n_ens] for ii in final_patient_box_results]
                del final_patient_box_results
                pool = Pool(processes=self.cf.n_workers)
                final_patient_box_results = pool.map(apply_wbc_to_patient, mp_inputs, chunksize=1)
                pool.close()
                pool.join()
                del mp_inputs
            elif self.cf.clustering == "nms":
                self.logger.info('applying standard NMS to test-set predictions with iou = {}.'.format(self.cf.clustering_iou))
                pool = Pool(processes=self.cf.n_workers)
                mp_inputs = [[ii[0], ii[1], self.cf.class_dict, self.cf.clustering_iou] for ii in final_patient_box_results]
                del final_patient_box_results
                final_patient_box_results = pool.map(apply_nms_to_patient, mp_inputs, chunksize=1)
                pool.close()
                pool.join()
                del mp_inputs

            if self.cf.merge_2D_to_3D_preds:
                self.logger.info('applying 2D-to-3D merging to test-set predictions with iou = {}.'.format(self.cf.merge_3D_iou))
                mp_inputs = [[ii[0], ii[1], self.cf.class_dict, self.cf.merge_3D_iou] for ii in final_patient_box_results]
                del final_patient_box_results
                pool = Pool(processes=self.cf.n_workers)
                final_patient_box_results = pool.map(apply_2d_3d_merging_to_patient, mp_inputs, chunksize=1)
                pool.close()
                pool.join()
                del mp_inputs
            # final_patient_box_results holds [avg_boxes, pid] if wbc
            for ix in range(len(results_per_patient)):
                assert results_per_patient[ix][1] == final_patient_box_results[ix][1], "should be same pid"
                results_per_patient[ix][0]["boxes"] = final_patient_box_results[ix][0]
            # results_per_patient = [(res_dict["boxes"] = boxes, pid) for (boxes,pid) in final_patient_box_results]

            return results_per_patient # holds list of (results_dict, pid)
