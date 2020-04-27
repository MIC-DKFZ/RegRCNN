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
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import torch

import utils.model_utils as mutils
import plotting as plg

import warnings


def get_roi_ap_from_df(inputs):
    '''
    :param df: data frame.
    :param det_thresh: min_threshold for filtering out low confidence predictions.
    :param per_patient_ap: boolean flag. evaluate average precision per patient id and average over per-pid results,
    instead of computing one ap over whole data set.
    :return: average_precision (float)
    '''

    df, det_thresh, per_patient_ap = inputs

    if per_patient_ap:
        pids_list = df.pid.unique()
        aps = []
        for match_iou in df.match_iou.unique():
            iou_df = df[df.match_iou == match_iou]
            for pid in pids_list:
                pid_df = iou_df[iou_df.pid == pid]
                all_p = len(pid_df[pid_df.class_label == 1])
                pid_df = pid_df[(pid_df.det_type == 'det_fp') | (pid_df.det_type == 'det_tp')].sort_values('pred_score', ascending=False)
                pid_df = pid_df[pid_df.pred_score > det_thresh]
                if (len(pid_df) ==0 and all_p == 0):
                   pass
                elif (len(pid_df) > 0 and all_p == 0):
                    aps.append(0)
                else:
                    aps.append(compute_roi_ap(pid_df, all_p))
        return np.mean(aps)

    else:
        aps = []
        for match_iou in df.match_iou.unique():
            iou_df = df[df.match_iou == match_iou]
            # it's important to not apply the threshold before counting all_p in order to not lose the fn!
            all_p = len(iou_df[(iou_df.det_type == 'det_tp') | (iou_df.det_type == 'det_fn')])
            # sorting out all entries that are not fp or tp or have confidence(=pred_score) <= detection_threshold
            iou_df = iou_df[(iou_df.det_type == 'det_fp') | (iou_df.det_type == 'det_tp')].sort_values('pred_score', ascending=False)
            iou_df = iou_df[iou_df.pred_score > det_thresh]
            if all_p>0:
                aps.append(compute_roi_ap(iou_df, all_p))
        return np.mean(aps)

def compute_roi_ap(df, all_p):
    """
    adapted from: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    :param df: dataframe containing class labels of predictions sorted in descending manner by their prediction score.
    :param all_p: number of all ground truth objects. (for denominator of recall.)
    :return:
    """
    tp = df.class_label.values
    fp = (tp == 0) * 1
    #recall thresholds, where precision will be measured
    R = np.linspace(0., 1., np.round((1. - 0.) / .01).astype(int) + 1, endpoint=True)
    tp_sum = np.cumsum(tp)
    fp_sum = np.cumsum(fp)
    n_dets = len(tp)
    rc = tp_sum / all_p
    pr = tp_sum / (fp_sum + tp_sum)

    # initialize precision array over recall steps (q=queries).
    q = [0. for _ in range(len(R))]
    # numpy is slow without cython optimization for accessing elements
    # python array gets significant speed improvement
    pr = pr.tolist()

    for i in range(n_dets - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]
        #--> pr[i]<=pr[i-1] for all i since we want to consider the maximum
        #precision value for a queried interval

    # discretize empiric recall steps with given bins.
    assert np.all(rc[:-1]<=rc[1:]), "recall not sorted ascendingly"
    inds = np.searchsorted(rc, R, side='left')
    try:
        for rc_ix, pr_ix in enumerate(inds):
            q[rc_ix] = pr[pr_ix]
    except IndexError: #now q is filled with pr values up to first non-available index
        pass

    return np.mean(q)

def roi_avp(inputs):
    '''
    :param df: data frame.
    :param det_thresh: min_threshold for filtering out low confidence predictions.
    :param per_patient_ap: boolean flag. evaluate average precision per patient id and average over per-pid results,
    instead of computing one ap over whole data set.
    :return: average_precision (float)
    '''

    df, det_thresh, per_patient_ap = inputs

    if per_patient_ap:
        pids_list = df.pid.unique()
        aps = []
        for match_iou in df.match_iou.unique():
            iou_df = df[df.match_iou == match_iou]
            for pid in pids_list:
                pid_df = iou_df[iou_df.pid == pid]
                all_p = len(pid_df[pid_df.class_label == 1])
                mask = ((pid_df.rg_bins == pid_df.rg_bin_target) & (pid_df.det_type == 'det_tp')) | (pid_df.det_type == 'det_fp')
                pid_df = pid_df[mask].sort_values('pred_score', ascending=False)
                pid_df = pid_df[pid_df.pred_score > det_thresh]
                if (len(pid_df) ==0 and all_p == 0):
                   pass
                elif (len(pid_df) > 0 and all_p == 0):
                    aps.append(0)
                else:
                    aps.append(compute_roi_ap(pid_df, all_p))
        return np.mean(aps)

    else:
        aps = []
        for match_iou in df.match_iou.unique():
            iou_df = df[df.match_iou == match_iou]
            #it's important to not apply the threshold before counting all_positives!
            all_p = len(iou_df[(iou_df.det_type == 'det_tp') | (iou_df.det_type == 'det_fn')])
            # filtering out tps which don't match rg_bin target at this point is same as reclassifying them as fn.
            # also sorting out all entries that are not fp or have confidence(=pred_score) <= detection_threshold
            mask = ((iou_df.rg_bins == iou_df.rg_bin_target) & (iou_df.det_type == 'det_tp')) | (iou_df.det_type == 'det_fp')
            iou_df = iou_df[mask].sort_values('pred_score', ascending=False)
            iou_df = iou_df[iou_df.pred_score > det_thresh]
            if all_p>0:
                aps.append(compute_roi_ap(iou_df, all_p))

        return np.mean(aps)

def compute_prc(df):
    """compute precision-recall curve with maximum precision per recall interval.
    :param df:
    :param all_p: # of all positive samples in data.
    :return: array: [precisions, recall query values]
    """
    assert (df.class_label==1).any(), "cannot compute prc when no positives in data."
    all_p = len(df[(df.det_type == 'det_tp') | (df.det_type == 'det_fn')])
    df = df[(df.det_type=="det_tp") | (df.det_type=="det_fp")]
    df = df.sort_values("pred_score", ascending=False)
    # recall thresholds, where precision will be measured
    scores = df.pred_score.values
    labels = df.class_label.values
    n_dets = len(scores)

    pr = np.zeros((n_dets,))
    rc = pr.copy()
    for rank in range(n_dets):
        tp = np.count_nonzero(labels[:rank+1]==1)
        fp = np.count_nonzero(labels[:rank+1]==0)

        pr[rank] = tp/(tp+fp)
        rc[rank] = tp/all_p

    #after obj detection convention/ coco-dataset template: take maximum pr within intervals:
    # --> pr[i]<=pr[i-1] for all i since we want to consider the maximum
    # precision value for a queried interval
    for i in range(n_dets - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]

    R = np.linspace(0., 1., np.round((1. - 0.) / .01).astype(int) + 1, endpoint=True)#precision queried at R points
    inds = np.searchsorted(rc, R, side='left')
    queries = np.zeros((len(R),))
    try:
        for q_ix, rank in enumerate(inds):
            queries[q_ix] = pr[rank]
    except IndexError:
        pass
    return np.array((queries, R))

def RMSE(y_true, y_pred, weights=None):
    if len(y_true)>0:
        return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=weights))
    else:
        return np.nan

def MAE_w_std(y_true, y_pred, weights=None):
    if len(y_true)>0:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        deltas = np.abs(y_true-y_pred)
        mae = np.average(deltas, weights=weights, axis=0).item()
        skmae = mean_absolute_error(y_true, y_pred, sample_weight=weights)
        assert np.allclose(mae, skmae, atol=1e-6), "mae {}, sklearn mae {}".format(mae, skmae)
        std = np.std(weights*deltas)
        return mae, std

    else:
        return np.nan, np.nan

def MAE(y_true, y_pred, weights=None):
    if len(y_true)>0:
        return mean_absolute_error(y_true, y_pred, sample_weight=weights)
    else:
        return np.nan

def accuracy(y_true, y_pred, weights=None):
    if len(y_true)>0:
        return accuracy_score(y_true, y_pred, sample_weight=weights)
    else:
        return np.nan


# noinspection PyCallingNonCallable
class Evaluator():
    """ Evaluates given results dicts. Can return results as updated monitor_metrics. Can save test data frames to
        file.
    """

    def __init__(self, cf, logger, mode='test'):
        """
        :param mode: either 'train', 'val_sampling', 'val_patient' or 'test'. handles prediction lists of different forms.
        """
        self.cf = cf
        self.logger = logger
        self.mode = mode

        self.regress_flag = any(['regression' in task for task in self.cf.prediction_tasks])

        self.plot_dir = self.cf.test_dir if self.mode == "test" else self.cf.plot_dir
        if self.cf.plot_prediction_histograms:
            self.hist_dir = os.path.join(self.plot_dir, 'histograms')
            os.makedirs(self.hist_dir, exist_ok=True)
        if self.cf.plot_stat_curves:
            self.curves_dir = os.path.join(self.plot_dir, 'stat_curves')
            os.makedirs(self.curves_dir, exist_ok=True)


    def eval_losses(self, batch_res_dicts):
        if hasattr(self.cf, "losses_to_monitor"):
            loss_names = self.cf.losses_to_monitor
        else:
            loss_names = {name for b_res_dict in batch_res_dicts for name in b_res_dict if 'loss' in name}
        self.epoch_losses = {l_name: torch.tensor([b_res_dict[l_name] for b_res_dict in batch_res_dicts if l_name
                                                   in b_res_dict.keys()]).mean().item() for l_name in loss_names}

    def eval_segmentations(self, batch_res_dicts, pid_list):

        batch_dices = [b_res_dict['batch_dices'] for b_res_dict in batch_res_dicts if
                       'batch_dices' in b_res_dict.keys()]  # shape (n_batches, n_seg_classes)
        if len(batch_dices) > 0:
            batch_dices = np.array(batch_dices)  # dims n_batches x 1 in sampling / n_test_epochs x n_classes
            assert batch_dices.shape[1] == self.cf.num_seg_classes, "bdices shp {}, n seg cl {}, pid lst len {}".format(
                batch_dices.shape, self.cf.num_seg_classes, len(pid_list))
            self.seg_df = pd.DataFrame()
            for seg_id in range(batch_dices.shape[1]):
                self.seg_df[self.cf.seg_id2label[seg_id].name + "_dice"] = batch_dices[:,
                                                                           seg_id]  # one row== one batch, one column== one class
                # self.seg_df[self.cf.seg_id2label[seg_id].name+"_dice"] = np.concatenate(batch_dices[:,:,seg_id])
            self.seg_df['fold'] = self.cf.fold
            if self.mode == "val_patient" or self.mode == "test":
                # need to make it more conform between sampling and patient-mode
                self.seg_df["pid"] = [pid for pix, pid in enumerate(pid_list)]  # for b_inst in batch_inst_boxes[pix]]
            else:
                self.seg_df["pid"] = np.nan

    def eval_boxes(self, batch_res_dicts, pid_list, obj_cl_dict,
                   obj_cl_identifiers={"gt":'class_targets', "pred":'box_pred_class_id'}):
        """

        :param batch_res_dicts:
        :param pid_list: [pid_0, pid_1, ...]
        :return:
        """
        if self.mode == 'train' or self.mode == 'val_sampling':
            # one pid per batch element
            # batch_size > 1, with varying patients across batch:
            # [[[results_0, ...], [pid_0, ...]], [[results_n, ...], [pid_n, ...]], ...]
            # -> [results_0, results_1, ..]
            batch_inst_boxes = [b_res_dict['boxes'] for b_res_dict in batch_res_dicts]  # len: nr of batches in epoch
            batch_inst_boxes = [[b_inst_boxes] for whole_batch_boxes in batch_inst_boxes for b_inst_boxes in
                                whole_batch_boxes]  # len: batch instances of whole epoch
            assert np.all(len(b_boxes_list) == self.cf.batch_size for b_boxes_list in batch_inst_boxes)
        elif self.mode == "val_patient" or self.mode == "test":
            # patient processing, one element per batch = one patient.
            # [[results_0, pid_0], [results_1, pid_1], ...] -> [results_0, results_1, ..]
            # in patientbatchiterator there is only one pid per batch
            batch_inst_boxes = [b_res_dict['boxes'] for b_res_dict in batch_res_dicts]
            # in patient mode not actually per batch instance, but per whole batch!
            if hasattr(self.cf, "eval_test_separately") and self.cf.eval_test_separately:
                """ you could write your own routines to add GTs to raw predictions for evaluation.
                    implemented standard is: cf.eval_test_separately = False or not set --> GTs are saved at same time 
                     and in same file as raw prediction results. 
                """
                raise NotImplementedError
        assert len(batch_inst_boxes) == len(pid_list)

        df_list_preds = []
        df_list_labels = []
        df_list_class_preds = []
        df_list_pids = []
        df_list_type = []
        df_list_match_iou = []
        df_list_n_missing = []
        df_list_regressions = []
        df_list_rg_targets = []
        df_list_rg_bins = []
        df_list_rg_bin_targets = []
        df_list_rg_uncs = []

        for match_iou in self.cf.ap_match_ious:
            self.logger.info('evaluating with ap_match_iou: {}'.format(match_iou))
            for cl in list(obj_cl_dict.keys()):
                for pix, pid in enumerate(pid_list):
                    len_df_list_before_patient = len(df_list_pids)
                    # input of each batch element is a list of boxes, where each box is a dictionary.
                    for b_inst_ix, b_boxes_list in enumerate(batch_inst_boxes[pix]):

                        b_tar_boxes = []
                        b_cand_boxes, b_cand_scores, b_cand_n_missing = [], [], []
                        if self.regress_flag:
                            b_tar_regs, b_tar_rg_bins = [], []
                            b_cand_regs, b_cand_rg_bins, b_cand_rg_uncs = [], [], []
                        for box in b_boxes_list:
                            # each box is either gt or detection or proposal/anchor
                            # we need all gts in the same order & all dets in same order
                            if box['box_type'] == 'gt' and box[obj_cl_identifiers["gt"]] == cl:
                                b_tar_boxes.append(box["box_coords"])
                                if self.regress_flag:
                                    b_tar_regs.append(np.array(box['regression_targets'], dtype='float32'))
                                    b_tar_rg_bins.append(box['rg_bin_targets'])

                            if box['box_type'] == 'det' and box[obj_cl_identifiers["pred"]] == cl:
                                b_cand_boxes.append(box["box_coords"])
                                b_cand_scores.append(box["box_score"])
                                b_cand_n_missing.append(box["cluster_n_missing"] if 'cluster_n_missing' in box.keys() else np.nan)
                                if self.regress_flag:
                                    b_cand_regs.append(box["regression"])
                                    b_cand_rg_bins.append(box["rg_bin"])
                                    b_cand_rg_uncs.append(box["rg_uncertainty"] if 'rg_uncertainty' in box.keys() else np.nan)
                        b_tar_boxes = np.array(b_tar_boxes)
                        b_cand_boxes, b_cand_scores, b_cand_n_missing = np.array(b_cand_boxes), np.array(b_cand_scores), np.array(b_cand_n_missing)
                        if self.regress_flag:
                            b_tar_regs, b_tar_rg_bins = np.array(b_tar_regs), np.array(b_tar_rg_bins)
                            b_cand_regs, b_cand_rg_bins, b_cand_rg_uncs = np.array(b_cand_regs), np.array(b_cand_rg_bins), np.array(b_cand_rg_uncs)

                        # check if predictions and ground truth boxes exist and match them according to match_iou.
                        if not 0 in b_cand_boxes.shape and not 0 in b_tar_boxes.shape:
                            assert np.all(np.round(b_cand_scores,6) <= 1.), "there is a box score>1: {}".format(b_cand_scores[~(b_cand_scores<=1.)])
                            #coords_check = np.array([len(coords)==self.cf.dim*2 for coords in b_cand_boxes])
                            #assert np.all(coords_check), "cand box with wrong bcoords dim: {}, mode: {}".format(b_cand_boxes[~coords_check], self.mode)
                            expected_dim = len(b_cand_boxes[0])
                            assert np.all([len(coords) == expected_dim for coords in b_tar_boxes]), \
                                "gt/cand box coords mismatch, expected dim: {}.".format(expected_dim)

                            # overlaps: shape len(cand_boxes) x len(tar_boxes)
                            overlaps = mutils.compute_overlaps(b_cand_boxes, b_tar_boxes)

                            # match_cand_ixs: shape (nr_of_matches,)
                            # theses indices are the indices of b_cand_boxes
                            match_cand_ixs = np.argwhere(np.max(overlaps, axis=1) > match_iou)[:, 0]

                            non_match_cand_ixs = np.argwhere(np.max(overlaps, 1) <= match_iou)[:, 0]
                            # the corresponding gt assigned to the pred boxes by highest iou overlap,
                            # i.e., match_gt_ixs holds index into b_tar_boxes for each entry in match_cand_ixs,
                            # i.e., gt_ixs and cand_ixs are paired via their position in their list
                            # (cand_ixs[j] corresponds to gt_ixs[j])
                            match_gt_ixs = np.argmax(overlaps[match_cand_ixs, :], axis=1) if \
                                not 0 in match_cand_ixs.shape else np.array([])
                            assert len(match_gt_ixs)==len(match_cand_ixs)

                            #match_gt_ixs: shape (nr_of_matches,) or 0
                            non_match_gt_ixs = np.array(
                                [ii for ii in np.arange(b_tar_boxes.shape[0]) if ii not in match_gt_ixs])
                            unique, counts = np.unique(match_gt_ixs, return_counts=True)

                            # check for double assignments, i.e. two predictions having been assigned to the same gt.
                            # according to the COCO-metrics, only one prediction counts as true positive, the rest counts as
                            # false positive. This case is supposed to be avoided by the model itself by,
                            #  e.g. using a low enough NMS threshold.
                            if np.any(counts > 1):
                                double_match_gt_ixs = unique[np.argwhere(counts > 1)[:, 0]]
                                keep_max = []
                                double_match_list = []
                                for dg in double_match_gt_ixs:
                                    double_match_cand_ixs = match_cand_ixs[np.argwhere(match_gt_ixs == dg)]
                                    keep_max.append(double_match_cand_ixs[np.argmax(b_cand_scores[double_match_cand_ixs])])
                                    double_match_list += [ii for ii in double_match_cand_ixs]

                                fp_ixs = np.array([ii for ii in match_cand_ixs if
                                                     (ii in double_match_list and ii not in keep_max)])
                                # count as fp: boxes that match gt above match_iou threshold but have not highest class confidence score
                                match_gt_ixs = np.array([gt_ix for ii, gt_ix in enumerate(match_gt_ixs) if match_cand_ixs[ii] not in fp_ixs])
                                match_cand_ixs = np.array([cand_ix for cand_ix in match_cand_ixs if cand_ix not in fp_ixs])
                                assert len(match_gt_ixs) == len(match_cand_ixs)

                                df_list_preds += [ii for ii in b_cand_scores[fp_ixs]]
                                df_list_labels += [0] * fp_ixs.shape[0]  # means label==gt==0==bg for all these fp_ixs
                                df_list_class_preds += [cl] * fp_ixs.shape[0]
                                df_list_n_missing += [n for n in b_cand_n_missing[fp_ixs]]
                                if self.regress_flag:
                                    df_list_regressions += [r for r in b_cand_regs[fp_ixs]]
                                    df_list_rg_bins += [r for r in b_cand_rg_bins[fp_ixs]]
                                    df_list_rg_uncs += [r for r in b_cand_rg_uncs[fp_ixs]]
                                    df_list_rg_targets += [[0.]*self.cf.regression_n_features] * fp_ixs.shape[0]
                                    df_list_rg_bin_targets += [0.] * fp_ixs.shape[0]
                                df_list_pids += [pid] * fp_ixs.shape[0]
                                df_list_type += ['det_fp'] * fp_ixs.shape[0]

                            # matched/tp:
                            if not 0 in match_cand_ixs.shape:
                                df_list_preds += list(b_cand_scores[match_cand_ixs])
                                df_list_labels += [1] * match_cand_ixs.shape[0]
                                df_list_class_preds += [cl] * match_cand_ixs.shape[0]
                                df_list_n_missing += list(b_cand_n_missing[match_cand_ixs])
                                if self.regress_flag:
                                    df_list_regressions += list(b_cand_regs[match_cand_ixs])
                                    df_list_rg_bins += list(b_cand_rg_bins[match_cand_ixs])
                                    df_list_rg_uncs += list(b_cand_rg_uncs[match_cand_ixs])
                                    assert len(match_cand_ixs)==len(match_gt_ixs)
                                    df_list_rg_targets += list(b_tar_regs[match_gt_ixs])
                                    df_list_rg_bin_targets += list(b_tar_rg_bins[match_gt_ixs])
                                df_list_pids += [pid] * match_cand_ixs.shape[0]
                                df_list_type += ['det_tp'] * match_cand_ixs.shape[0]
                            # rest fp:
                            if not 0 in non_match_cand_ixs.shape:
                                df_list_preds += list(b_cand_scores[non_match_cand_ixs])
                                df_list_labels += [0] * non_match_cand_ixs.shape[0]
                                df_list_class_preds += [cl] * non_match_cand_ixs.shape[0]
                                df_list_n_missing += list(b_cand_n_missing[non_match_cand_ixs])
                                if self.regress_flag:
                                    df_list_regressions += list(b_cand_regs[non_match_cand_ixs])
                                    df_list_rg_bins += list(b_cand_rg_bins[non_match_cand_ixs])
                                    df_list_rg_uncs += list(b_cand_rg_uncs[non_match_cand_ixs])
                                    df_list_rg_targets += [[0.]*self.cf.regression_n_features] * non_match_cand_ixs.shape[0]
                                    df_list_rg_bin_targets += [0.] * non_match_cand_ixs.shape[0]
                                df_list_pids += [pid] * non_match_cand_ixs.shape[0]
                                df_list_type += ['det_fp'] * non_match_cand_ixs.shape[0]
                            # fn:
                            if not 0 in non_match_gt_ixs.shape:
                                df_list_preds += [0] * non_match_gt_ixs.shape[0]
                                df_list_labels += [1] * non_match_gt_ixs.shape[0]
                                df_list_class_preds += [cl] * non_match_gt_ixs.shape[0]
                                df_list_n_missing += [np.nan] * non_match_gt_ixs.shape[0]
                                if self.regress_flag:
                                    df_list_regressions += [[0.]*self.cf.regression_n_features] * non_match_gt_ixs.shape[0]
                                    df_list_rg_bins += [0.] * non_match_gt_ixs.shape[0]
                                    df_list_rg_uncs += [np.nan] * non_match_gt_ixs.shape[0]
                                    df_list_rg_targets += list(b_tar_regs[non_match_gt_ixs])
                                    df_list_rg_bin_targets += list(b_tar_rg_bins[non_match_gt_ixs])
                                df_list_pids += [pid]  * non_match_gt_ixs.shape[0]
                                df_list_type += ['det_fn']  * non_match_gt_ixs.shape[0]
                        # only fp:
                        if not 0 in b_cand_boxes.shape and 0 in b_tar_boxes.shape:
                            # means there is no gt in all samples! any preds have to be fp.
                            df_list_preds += list(b_cand_scores)
                            df_list_labels += [0] * b_cand_boxes.shape[0]
                            df_list_class_preds += [cl] * b_cand_boxes.shape[0]
                            df_list_n_missing += list(b_cand_n_missing)
                            if self.regress_flag:
                                df_list_regressions += list(b_cand_regs)
                                df_list_rg_bins += list(b_cand_rg_bins)
                                df_list_rg_uncs += list(b_cand_rg_uncs)
                                df_list_rg_targets += [[0.]*self.cf.regression_n_features] * b_cand_boxes.shape[0]
                                df_list_rg_bin_targets += [0.] * b_cand_boxes.shape[0]
                            df_list_pids += [pid] * b_cand_boxes.shape[0]
                            df_list_type += ['det_fp'] * b_cand_boxes.shape[0]
                        # only fn:
                        if 0 in b_cand_boxes.shape and not 0 in b_tar_boxes.shape:
                            df_list_preds += [0] * b_tar_boxes.shape[0]
                            df_list_labels += [1] * b_tar_boxes.shape[0]
                            df_list_class_preds += [cl] * b_tar_boxes.shape[0]
                            df_list_n_missing += [np.nan] * b_tar_boxes.shape[0]
                            if self.regress_flag:
                                df_list_regressions += [[0.]*self.cf.regression_n_features] * b_tar_boxes.shape[0]
                                df_list_rg_bins += [0.] * b_tar_boxes.shape[0]
                                df_list_rg_uncs += [np.nan] * b_tar_boxes.shape[0]
                                df_list_rg_targets += list(b_tar_regs)
                                df_list_rg_bin_targets += list(b_tar_rg_bins)
                            df_list_pids += [pid] * b_tar_boxes.shape[0]
                            df_list_type += ['det_fn'] * b_tar_boxes.shape[0]

                    # empty patient with 0 detections needs empty patient score, in order to not disappear from stats.
                    # filtered out for roi-level evaluation later. During training (and val_sampling),
                    # tn are assigned per sample independently of associated patients.
                    # i.e., patient_tn is also meant as sample_tn if a list of samples is evaluated instead of whole patient
                    if len(df_list_pids) == len_df_list_before_patient:
                        df_list_preds += [0]
                        df_list_labels += [0]
                        df_list_class_preds += [cl]
                        df_list_n_missing += [np.nan]
                        if self.regress_flag:
                            df_list_regressions += [[0.]*self.cf.regression_n_features]
                            df_list_rg_bins += [0.]
                            df_list_rg_uncs += [np.nan]
                            df_list_rg_targets += [[0.]*self.cf.regression_n_features]
                            df_list_rg_bin_targets += [0.]
                        df_list_pids += [pid]
                        df_list_type += ['patient_tn'] # true negative: no ground truth boxes, no detections.

            df_list_match_iou += [match_iou] * (len(df_list_preds) - len(df_list_match_iou))

        self.test_df = pd.DataFrame()
        self.test_df['pred_score'] = df_list_preds
        self.test_df['class_label'] = df_list_labels
        # class labels are gt, 0,1, only indicate neg/pos (or bg/fg) remapped from all classes
        self.test_df['pred_class'] = df_list_class_preds # can be diff than 0,1
        self.test_df['pid'] = df_list_pids
        self.test_df['det_type'] = df_list_type
        self.test_df['fold'] = self.cf.fold
        self.test_df['match_iou'] = df_list_match_iou
        self.test_df['cluster_n_missing'] = df_list_n_missing
        if self.regress_flag:
            self.test_df['regressions'] = df_list_regressions
            self.test_df['rg_targets'] = df_list_rg_targets
            self.test_df['rg_uncertainties'] = df_list_rg_uncs
            self.test_df['rg_bins'] = df_list_rg_bins
            # super weird error: pandas does not properly add an attribute if column is named "rg_bin_targets" ... ?!?
            self.test_df['rg_bin_target'] = df_list_rg_bin_targets
            assert hasattr(self.test_df, "rg_bin_target")

        #fn_df = self.test_df[self.test_df["det_type"] == "det_fn"]

        pass

    def evaluate_predictions(self, results_list, monitor_metrics=None):
        """
        Performs the matching of predicted boxes and ground truth boxes. Loops over list of matching IoUs and foreground classes.
        Resulting info of each prediction is stored as one line in an internal dataframe, with the keys:
        det_type: 'tp' (true positive), 'fp' (false positive), 'fn' (false negative), 'tn' (true negative)
        pred_class: foreground class which the object predicts.
        pid: corresponding patient-id.
        pred_score: confidence score [0, 1]
        fold: corresponding fold of CV.
        match_iou: utilized IoU for matching.
        :param results_list: list of model predictions. Either from train/val_sampling (patch processing) for monitoring with form:
        [[[results_0, ...], [pid_0, ...]], [[results_n, ...], [pid_n, ...]], ...]
        Or from val_patient/testing (patient processing), with form: [[results_0, pid_0], [results_1, pid_1], ...])
        :param monitor_metrics (optional):  dict of dicts with all metrics of previous epochs.
        :return monitor_metrics: if provided (during training), return monitor_metrics now including results of current epoch.
        """
        # gets results_list = [[batch_instances_box_lists], [batch_instances_pids]]*n_batches
        # we want to evaluate one batch_instance (= 2D or 3D image) at a time.


        self.logger.info('evaluating in mode {}'.format(self.mode))

        batch_res_dicts = [batch[0] for batch in results_list]  # len: nr of batches in epoch
        if self.mode == 'train' or self.mode=='val_sampling':
            # one pid per batch element
            # [[[results_0, ...], [pid_0, ...]], [[results_n, ...], [pid_n, ...]], ...]
            # -> [pid_0, pid_1, ...]
            # additional list wrapping to make conform with below per-patient batches, where one pid is linked to more than one batch instance
            pid_list = [batch_instance_pid for batch in results_list for batch_instance_pid in batch[1]]
        elif self.mode == "val_patient" or self.mode=="test":
            # [[results_0, pid_0], [results_1, pid_1], ...] -> [pid_0, pid_1, ...]
            # in patientbatchiterator there is only one pid per batch
            pid_list = [np.unique(batch[1]) for batch in results_list]
            assert np.all([len(pid)==1 for pid in pid_list]), "pid list in patient-eval mode, should only contain a single scalar per patient: {}".format(pid_list)
            pid_list = [pid[0] for pid in pid_list]
        else:
            raise Exception("undefined run mode encountered")

        self.eval_losses(batch_res_dicts)
        self.eval_segmentations(batch_res_dicts, pid_list)
        self.eval_boxes(batch_res_dicts, pid_list, self.cf.class_dict)

        if monitor_metrics is not None:
            # return all_stats, updated monitor_metrics
            return self.return_metrics(self.test_df, self.cf.class_dict, monitor_metrics)

    def return_metrics(self, df, obj_cl_dict, monitor_metrics=None, boxes_only=False):
        """
        Calculates metric scores for internal data frame. Called directly from evaluate_predictions during training for
        monitoring, or from score_test_df during inference (for single folds or aggregated test set).
        Loops over foreground classes and score_levels ('roi' and/or 'patient'), gets scores and stores them.
        Optionally creates plots of prediction histograms and ROC/PR curves.
        :param df: Data frame that holds evaluated predictions.
        :param obj_cl_dict: Dict linking object-class ids to object-class names. E.g., {1: "bikes", 2 : "cars"}. Set in
            configs as cf.class_dict.
        :param monitor_metrics: dict of dicts with all metrics of previous epochs. This function adds metrics for
         current epoch and returns the same object.
        :param boxes_only: whether to produce metrics only for the boxes, not the segmentations.
        :return: all_stats: list. Contains dicts with resulting scores for each combination of foreground class and
        score_level.
        :return: monitor_metrics
        """

        # -------------- monitoring independent of class, score level ------------
        if monitor_metrics is not None:
            for l_name in self.epoch_losses:
                monitor_metrics[l_name] = [self.epoch_losses[l_name]]

        # -------------- metrics calc dependent on class, score level ------------

        all_stats = [] # all_stats: one entry per score_level per class

        for cl in list(obj_cl_dict.keys()):    # bg eval is neglected
            cl_name = obj_cl_dict[cl]
            cl_df = df[df.pred_class == cl]

            if hasattr(self, "seg_df") and not boxes_only:
                dice_col = self.cf.seg_id2label[cl].name+"_dice"
                seg_cl_df = self.seg_df.loc[:,['pid', dice_col, 'fold']]

            for score_level in self.cf.report_score_level:

                stats_dict = {}
                stats_dict['name'] = 'fold_{} {} {}'.format(self.cf.fold, score_level, cl_name)

                # -------------- RoI-based -----------------
                if score_level == 'rois':

                    stats_dict['auc'] = np.nan
                    stats_dict['roc'] = np.nan

                    if monitor_metrics is not None:
                        tn = len(cl_df[cl_df.det_type == "patient_tn"])
                        tp = len(cl_df[(cl_df.det_type == "det_tp")&(cl_df.pred_score>self.cf.min_det_thresh)])
                        fp = len(cl_df[(cl_df.det_type == "det_fp")&(cl_df.pred_score>self.cf.min_det_thresh)])
                        fn = len(cl_df[cl_df.det_type == "det_fn"])
                        sens = np.divide(tp, (fn + tp))
                        monitor_metrics.update({"Bin_Stats/" + cl_name + "_fp": [fp], "Bin_Stats/" + cl_name + "_tp": [tp],
                                                 "Bin_Stats/" + cl_name + "_fn": [fn], "Bin_Stats/" + cl_name + "_tn": [tn],
                                                 "Bin_Stats/" + cl_name + "_sensitivity": [sens]})
                        # list wrapping only needed bc other metrics are recorded over all epochs;

                    spec_df = cl_df[cl_df.det_type != 'patient_tn']
                    if self.regress_flag:
                        # filter false negatives out for regression-only eval since regressor didn't predict
                        truncd_df = spec_df[(((spec_df.det_type == "det_fp") | (
                                    spec_df.det_type == "det_tp")) & spec_df.pred_score > self.cf.min_det_thresh)]
                        truncd_df_tp = truncd_df[truncd_df.det_type == "det_tp"]
                        weights, weights_tp = truncd_df.pred_score.tolist(), truncd_df_tp.pred_score.tolist()

                        y_true, y_pred = truncd_df.rg_targets.tolist(), truncd_df.regressions.tolist()
                        stats_dict["rg_RMSE"] = RMSE(y_true, y_pred)
                        stats_dict["rg_MAE"] = MAE(y_true, y_pred)
                        stats_dict["rg_RMSE_weighted"] = RMSE(y_true, y_pred, weights)
                        stats_dict["rg_MAE_weighted"] = MAE(y_true, y_pred, weights)
                        y_true, y_pred = truncd_df_tp.rg_targets.tolist(), truncd_df_tp.regressions.tolist()
                        stats_dict["rg_MAE_weighted_tp"] = MAE(y_true, y_pred, weights_tp)
                        stats_dict["rg_MAE_w_std_weighted_tp"] = MAE_w_std(y_true, y_pred, weights_tp)

                        y_true, y_pred = truncd_df.rg_bin_target.tolist(), truncd_df.rg_bins.tolist()
                        stats_dict["rg_bin_accuracy"] = accuracy(y_true, y_pred)
                        stats_dict["rg_bin_accuracy_weighted"] = accuracy(y_true, y_pred, weights)

                        y_true, y_pred = truncd_df_tp.rg_bin_target.tolist(), truncd_df_tp.rg_bins.tolist()
                        stats_dict["rg_bin_accuracy_weighted_tp"] = accuracy(y_true, y_pred, weights_tp)
                        if np.any(~truncd_df.rg_uncertainties.isna()):
                            # det_fn are expected to be NaN so they drop out in means
                            stats_dict.update({"rg_uncertainty": truncd_df.rg_uncertainties.mean(),
                                               "rg_uncertainty_tp": truncd_df_tp.rg_uncertainties.mean(),
                                               "rg_uncertainty_tp_weighted": (truncd_df_tp.rg_uncertainties * truncd_df_tp.pred_score).sum()
                                                                             / truncd_df_tp.pred_score.sum()
                                               })

                    if (spec_df.class_label==1).any():
                        stats_dict['ap'] = get_roi_ap_from_df((spec_df, self.cf.min_det_thresh, self.cf.per_patient_ap))
                        stats_dict['prc'] = precision_recall_curve(spec_df.class_label.tolist(), spec_df.pred_score.tolist())
                        if self.regress_flag:
                            stats_dict['avp'] = roi_avp((spec_df, self.cf.min_det_thresh, self.cf.per_patient_ap))
                    else:
                        stats_dict['ap'] = np.nan
                        stats_dict['prc'] = np.nan
                        stats_dict['avp'] = np.nan
                        # np.nan is formattable by __format__ as a float, None-type is not

                    if hasattr(self, "seg_df") and not boxes_only:
                        stats_dict["dice"] = seg_cl_df.loc[:,dice_col].mean() # mean per all rois in this epoch
                        stats_dict["dice_std"] = seg_cl_df.loc[:,dice_col].std()

                    # for the aggregated test set case, additionally get the scores of averaging over fold results.
                    if self.cf.evaluate_fold_means and len(df.fold.unique()) > 1:
                        aps = []
                        for fold in df.fold.unique():
                            fold_df = spec_df[spec_df.fold == fold]
                            if (fold_df.class_label==1).any():
                                aps.append(get_roi_ap_from_df((fold_df, self.cf.min_det_thresh, self.cf.per_patient_ap)))

                        stats_dict['ap_folds_mean'] = np.mean(aps) if len(aps)>0 else np.nan
                        stats_dict['ap_folds_std'] = np.std(aps) if len(aps)>0 else np.nan
                        stats_dict['auc_folds_mean'] = np.nan
                        stats_dict['auc_folds_std'] = np.nan
                        if self.regress_flag:
                            avps, accuracies, MAEs = [], [], []
                            for fold in df.fold.unique():
                                fold_df = spec_df[spec_df.fold == fold]
                                if (fold_df.class_label == 1).any():
                                    avps.append(roi_avp((fold_df, self.cf.min_det_thresh, self.cf.per_patient_ap)))
                                truncd_df_tp = fold_df[((fold_df.det_type == "det_tp") & fold_df.pred_score > self.cf.min_det_thresh)]
                                weights_tp = truncd_df_tp.pred_score.tolist()
                                y_true, y_pred = truncd_df_tp.rg_bin_target.tolist(), truncd_df_tp.rg_bins.tolist()
                                accuracies.append(accuracy(y_true, y_pred, weights_tp))
                                y_true, y_pred = truncd_df_tp.rg_targets.tolist(), truncd_df_tp.regressions.tolist()
                                MAEs.append(MAE_w_std(y_true, y_pred, weights_tp))

                            stats_dict['avp_folds_mean'] = np.mean(avps) if len(avps) > 0 else np.nan
                            stats_dict['avp_folds_std'] = np.std(avps) if len(avps) > 0 else np.nan
                            stats_dict['rg_bin_accuracy_weighted_tp_folds_mean'] = np.mean(accuracies) if len(accuracies) > 0 else np.nan
                            stats_dict['rg_bin_accuracy_weighted_tp_folds_std'] = np.std(accuracies) if len(accuracies) > 0 else np.nan
                            stats_dict['rg_MAE_w_std_weighted_tp_folds_mean'] = np.mean(MAEs, axis=0) if len(MAEs) > 0 else np.nan
                            stats_dict['rg_MAE_w_std_weighted_tp_folds_std'] = np.std(MAEs, axis=0) if len(MAEs) > 0 else np.nan

                    if hasattr(self, "seg_df") and not boxes_only and self.cf.evaluate_fold_means and len(seg_cl_df.fold.unique()) > 1:
                        fold_means = seg_cl_df.groupby(['fold'], as_index=True).agg({dice_col:"mean"})
                        stats_dict["dice_folds_mean"] = float(fold_means.mean())
                        stats_dict["dice_folds_std"] = float(fold_means.std())

                # -------------- patient-based -----------------
                # on patient level, aggregate predictions per patient (pid): The patient predicted score is the highest
                # confidence prediction for this class. The patient class label is 1 if roi of this class exists in patient, else 0.
                if score_level == 'patient':
                    #this is the critical part in patient scoring: only the max gt and max pred score are taken per patient!
                    #--> does mix up values from separate detections
                    spec_df = cl_df.groupby(['pid'], as_index=False)
                    agg_args = {'class_label': 'max', 'pred_score': 'max', 'fold': 'first'}
                    if self.regress_flag:
                        # pandas throws error if aggregated value is np.array, not if is list.
                        agg_args.update({'regressions': lambda series: list(series.iloc[np.argmax(series.apply(np.linalg.norm).values)]),
                                         'rg_targets': lambda series: list(series.iloc[np.argmax(series.apply(np.linalg.norm).values)]),
                                         'rg_bins': 'max', 'rg_bin_target': 'max',
                                         'rg_uncertainties': 'max'
                                         })
                    if hasattr(cl_df, "cluster_n_missing"):
                        agg_args.update({'cluster_n_missing': 'mean'})
                    spec_df = spec_df.agg(agg_args)

                    if len(spec_df.class_label.unique()) > 1:
                        stats_dict['auc'] = roc_auc_score(spec_df.class_label.tolist(), spec_df.pred_score.tolist())
                        stats_dict['roc'] = roc_curve(spec_df.class_label.tolist(), spec_df.pred_score.tolist())
                    else:
                        stats_dict['auc'] = np.nan
                        stats_dict['roc'] = np.nan

                    if (spec_df.class_label == 1).any():
                        patient_cl_labels = spec_df.class_label.tolist()
                        stats_dict['ap'] = average_precision_score(patient_cl_labels, spec_df.pred_score.tolist())
                        stats_dict['prc'] = precision_recall_curve(patient_cl_labels, spec_df.pred_score.tolist())
                        if self.regress_flag:
                            avp_scores = spec_df[spec_df.rg_bins == spec_df.rg_bin_target].pred_score.tolist()
                            avp_scores += [0.] * (len(patient_cl_labels) - len(avp_scores))
                            stats_dict['avp'] = average_precision_score(patient_cl_labels, avp_scores)
                    else:
                        stats_dict['ap'] = np.nan
                        stats_dict['prc'] = np.nan
                        stats_dict['avp'] = np.nan
                    if self.regress_flag:
                        y_true, y_pred = spec_df.rg_targets.tolist(), spec_df.regressions.tolist()
                        stats_dict["rg_RMSE"] = RMSE(y_true, y_pred)
                        stats_dict["rg_MAE"] = MAE(y_true, y_pred)
                        stats_dict["rg_bin_accuracy"] = accuracy(spec_df.rg_bin_target.tolist(), spec_df.rg_bins.tolist())
                        stats_dict["rg_uncertainty"] = spec_df.rg_uncertainties.mean()
                    if hasattr(self, "seg_df") and not boxes_only:
                        seg_cl_df = seg_cl_df.groupby(['pid'], as_index=False).agg(
                            {dice_col: "mean", "fold": "first"})  # mean of all rois per patient in this epoch
                        stats_dict["dice"] = seg_cl_df.loc[:,dice_col].mean() #mean of all patients
                        stats_dict["dice_std"] = seg_cl_df.loc[:, dice_col].std()


                    # for the aggregated test set case, additionally get the scores for averaging over fold results.
                    if self.cf.evaluate_fold_means and len(df.fold.unique()) > 1 and self.mode in ["test", "analysis"]:
                        aucs = []
                        aps = []
                        for fold in df.fold.unique():
                            fold_df = spec_df[spec_df.fold == fold]
                            if (fold_df.class_label==1).any():
                                aps.append(
                                    average_precision_score(fold_df.class_label.tolist(), fold_df.pred_score.tolist()))
                            if len(fold_df.class_label.unique())>1:
                                aucs.append(roc_auc_score(fold_df.class_label.tolist(), fold_df.pred_score.tolist()))
                        stats_dict['auc_folds_mean'] = np.mean(aucs)
                        stats_dict['auc_folds_std'] = np.std(aucs)
                        stats_dict['ap_folds_mean'] = np.mean(aps)
                        stats_dict['ap_folds_std'] = np.std(aps)
                    if hasattr(self, "seg_df") and not boxes_only and self.cf.evaluate_fold_means and len(seg_cl_df.fold.unique()) > 1:
                        fold_means = seg_cl_df.groupby(['fold'], as_index=True).agg({dice_col:"mean"})
                        stats_dict["dice_folds_mean"] = float(fold_means.mean())
                        stats_dict["dice_folds_std"] = float(fold_means.std())

                all_stats.append(stats_dict)

                # -------------- monitoring, visualisation -----------------
                # fill new results into monitor_metrics dict. for simplicity, only one class (of interest) is monitored on patient level.
                patient_interests = [self.cf.class_dict[self.cf.patient_class_of_interest],]
                if hasattr(self.cf, "bin_dict"):
                    patient_interests += [self.cf.bin_dict[self.cf.patient_bin_of_interest]]
                if monitor_metrics is not None and (score_level != 'patient' or cl_name in patient_interests):
                    name = 'patient_'+cl_name if score_level == 'patient' else cl_name
                    for metric in self.cf.metrics:
                        if metric in stats_dict.keys():
                            monitor_metrics[name + '_'+metric].append(stats_dict[metric])
                        else:
                            print("WARNING: skipped monitor metric {}_{} since not avail".format(name, metric))

                # histograms
                if self.cf.plot_prediction_histograms:
                    out_filename = os.path.join(self.hist_dir, 'pred_hist_{}_{}_{}_{}'.format(
                            self.cf.fold, self.mode, score_level, cl_name))
                    plg.plot_prediction_hist(self.cf, spec_df, out_filename)

                # analysis of the  hyper-parameter cf.min_det_thresh, for optimization on validation set.
                if self.cf.scan_det_thresh and "val" in self.mode:
                    conf_threshs = list(np.arange(0.8, 1, 0.02))
                    pool = Pool(processes=self.cf.n_workers)
                    mp_inputs = [[spec_df, ii, self.cf.per_patient_ap] for ii in conf_threshs]
                    aps = pool.map(get_roi_ap_from_df, mp_inputs, chunksize=1)
                    pool.close()
                    pool.join()
                    self.logger.info('results from scanning over det_threshs: {}'.format([[i, j] for i, j in zip(conf_threshs, aps)]))

        class_means = pd.DataFrame(columns=self.cf.report_score_level)
        for slevel in self.cf.report_score_level:
            level_stats = pd.DataFrame([stats for stats in all_stats if slevel in stats["name"]])[self.cf.metrics]
            class_means.loc[:, slevel] = level_stats.mean()
        all_stats.extend([{"name": 'fold_{} {} {}'.format(self.cf.fold, slevel, "class_means"), **level_means} for
                          slevel, level_means in class_means.to_dict().items()])

        if self.cf.plot_stat_curves:
            out_filename = os.path.join(self.curves_dir, '{}_{}_stat_curves'.format(self.cf.fold, self.mode))
            plg.plot_stat_curves(self.cf, all_stats, out_filename)
        if self.cf.plot_prediction_histograms and hasattr(df, "cluster_n_missing") and df.cluster_n_missing.notna().any():
            out_filename = os.path.join(self.hist_dir, 'n_missing_hist_{}_{}.png'.format(self.cf.fold, self.mode))
            plg.plot_wbc_n_missing(self.cf, df, outfile=out_filename)

        return all_stats, monitor_metrics

    def write_to_results_table(self, stats, metrics_to_score):
        """Write overall results to a common inter-experiment table.
        :param metrics_to_score:
        :return:
        """
        results_table_path = os.path.join(self.cf.test_dir, "../../", 'results_table.csv')
        with open(results_table_path, 'a') as handle:
            # ---column headers---
            handle.write('\n{},'.format("Experiment Name"))
            handle.write('{},'.format("Time Stamp"))
            handle.write('{},'.format("Samples Seen"))
            handle.write('{},'.format("Spatial Dim"))
            handle.write('{},'.format("Patch Size"))
            handle.write('{},'.format("CV Folds"))
            handle.write('{},'.format("{}-clustering IoU".format(self.cf.clustering)))
            handle.write('{},'.format("Merge-2D-to-3D IoU"))
            if hasattr(self.cf, "test_against_exact_gt"):
                handle.write('{},'.format('Exact GT'))
            for s in stats:
                if self.cf.class_dict[self.cf.patient_class_of_interest] in s['name'] or "mean" in s["name"]:
                    for metric in metrics_to_score:
                        if metric in s.keys() and not np.isnan(s[metric]):
                            if metric == 'ap':
                                handle.write('{}_{} : {}_{},'.format(*s['name'].split(" ")[1:], metric,
                                                                     int(np.mean(self.cf.ap_match_ious) * 100)))
                            elif not "folds_std" in metric:
                                handle.write('{}_{} : {},'.format(*s['name'].split(" ")[1:], metric))
                        else:
                            print("WARNING: skipped metric {} since not avail".format(metric))
            handle.write('\n')

            # --- columns content---
            handle.write('{},'.format(self.cf.exp_dir.split(os.sep)[-1]))
            handle.write('{},'.format(time.strftime("%d%b%y %H:%M:%S")))
            handle.write('{},'.format(self.cf.num_epochs * self.cf.num_train_batches * self.cf.batch_size))
            handle.write('{}D,'.format(self.cf.dim))
            handle.write('{},'.format("x".join([str(self.cf.patch_size[i]) for i in range(self.cf.dim)])))
            handle.write('{},'.format(str(self.test_df.fold.unique().tolist()).replace(",", "")))
            handle.write('{},'.format(self.cf.clustering_iou if self.cf.clustering else str("N/A")))
            handle.write('{},'.format(self.cf.merge_3D_iou if self.cf.merge_2D_to_3D_preds else str("N/A")))
            if hasattr(self.cf, "test_against_exact_gt"):
                handle.write('{},'.format(self.cf.test_against_exact_gt))
            for s in stats:
                if self.cf.class_dict[self.cf.patient_class_of_interest] in s['name'] or "mean" in s["name"]:
                    for metric in metrics_to_score:
                        if metric in s.keys() and not np.isnan(
                                s[metric]):  # needed as long as no dice on patient level possible
                            if "folds_mean" in metric:
                                handle.write('{:0.3f}\u00B1{:0.3f}, '.format(s[metric],
                                                                             s["_".join(
                                                                                 (*metric.split("_")[:-1], "std"))]))
                            elif not "folds_std" in metric:
                                handle.write('{:0.3f}, '.format(s[metric]))

            handle.write('\n')

    def score_test_df(self, max_fold=None, internal_df=True):
        """
        Writes out resulting scores to text files: First checks for class-internal-df (typically current) fold,
        gets resulting scores, writes them to a text file and pickles data frame. Also checks if data-frame pickles of
        all folds of cross-validation exist in exp_dir. If true, loads all dataframes, aggregates test sets over folds,
        and calculates and writes out overall metrics.
        """
        # this should maybe be extended to auc, ap stds.
        metrics_to_score = self.cf.metrics.copy() # + [ m+ext for m in self.cf.metrics if "dice" in m for ext in ["_std"]]

        if internal_df:

            self.test_df.to_pickle(os.path.join(self.cf.test_dir, '{}_test_df.pkl'.format(self.cf.fold)))
            if hasattr(self, "seg_df"):
                self.seg_df.to_pickle(os.path.join(self.cf.test_dir, '{}_test_seg_df.pkl'.format(self.cf.fold)))
            stats, _ = self.return_metrics(self.test_df, self.cf.class_dict)

            with open(os.path.join(self.cf.test_dir, 'results.txt'), 'a') as handle:
                handle.write('\n****************************\n')
                handle.write('\nresults for fold {}, {} \n'.format(self.cf.fold, time.strftime("%d/%m/%y %H:%M:%S")))
                handle.write('\n****************************\n')
                handle.write('\nfold df shape {}\n  \n'.format(self.test_df.shape))
                for s in stats:
                    for metric in metrics_to_score:
                        if metric in s.keys():  #needed as long as no dice on patient level poss
                            if "accuracy" in metric:
                                handle.write('{} {:0.4f}  '.format(metric, s[metric]))
                            else:
                                handle.write('{} {:0.3f}  '.format(metric, s[metric]))
                        else:
                            print("WARNING: skipped metric {} since not avail".format(metric))
                    handle.write('{} \n'.format(s['name']))


        if max_fold is None:
            max_fold = self.cf.n_cv_splits-1
        if self.cf.fold == max_fold:
            print("max fold/overall stats triggered")
            self.cf.fold = 'overall'
            if self.cf.evaluate_fold_means:
                metrics_to_score += [m + ext for m in self.cf.metrics for ext in ("_folds_mean", "_folds_std")]

            if not self.cf.hold_out_test_set or not self.cf.ensemble_folds:
                fold_df_paths = sorted([ii for ii in os.listdir(self.cf.test_dir)
                                        if 'test_df.pkl' in ii and not "overall" in ii])
                fold_seg_df_paths = sorted([ii for ii in os.listdir(self.cf.test_dir)
                                            if 'test_seg_df.pkl' in ii and not "overall" in ii])
                for paths in [fold_df_paths, fold_seg_df_paths]:
                    assert len(paths) <= self.cf.n_cv_splits, "found {} > nr of cv splits results dfs in {}".format(
                        len(paths), self.cf.test_dir)
                with open(os.path.join(self.cf.test_dir, 'results.txt'), 'a') as handle:
                    dfs_list = [pd.read_pickle(os.path.join(self.cf.test_dir, ii)) for ii in fold_df_paths]
                    seg_dfs_list = [pd.read_pickle(os.path.join(self.cf.test_dir, ii)) for ii in fold_seg_df_paths]

                    self.test_df = pd.concat(dfs_list, sort=True)
                    if len(seg_dfs_list)>0:
                        self.seg_df = pd.concat(seg_dfs_list, sort=True)
                    stats, _ = self.return_metrics(self.test_df, self.cf.class_dict)

                    handle.write('\n****************************\n')
                    handle.write('\nOVERALL RESULTS \n')
                    handle.write('\n****************************\n')
                    handle.write('\ndf shape \n  \n'.format(self.test_df.shape))
                    for s in stats:
                        for metric in metrics_to_score:
                            if metric in s.keys():
                                handle.write('{} {:0.3f}  '.format(metric, s[metric]))
                        handle.write('{} \n'.format(s['name']))

            self.write_to_results_table(stats, metrics_to_score)

            with open(os.path.join(self.cf.test_dir, 'results_extr_scores.txt'), 'w') as handle:
                handle.write('\n****************************\n')
                handle.write('\nextremal scores for fold {} \n'.format(self.cf.fold))
                handle.write('\n****************************\n')
                # want: pid & fold (&other) of highest scoring tp & fp in test_df
                for cl in self.cf.class_dict.keys():
                    print("\nClass {}".format(self.cf.class_dict[cl]), file=handle)
                    cl_df = self.test_df[self.test_df.pred_class == cl] #.dropna(axis=1)
                    for det_type in ['det_tp', 'det_fp']:
                        filtered_df = cl_df[cl_df.det_type==det_type]
                        print("\nHighest scoring {} of class {}".format(det_type, self.cf.class_dict[cl]), file=handle)
                        if len(filtered_df)>0:
                            print(filtered_df.loc[filtered_df.pred_score.idxmax()], file=handle)
                        else:
                            print("No detections of type {} for class {} in this df".format(det_type, self.cf.class_dict[cl]), file=handle)
                    handle.write('\n****************************\n')
