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

"""Retina Net. According to https://arxiv.org/abs/1708.02002"""

import utils.model_utils as mutils
import utils.exp_utils as utils
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

sys.path.append('..')
from custom_extensions.nms import nms

class Classifier(nn.Module):


    def __init__(self, cf, conv):
        """
        Builds the classifier sub-network.
        """
        super(Classifier, self).__init__()
        self.dim = conv.dim
        self.n_classes = cf.head_classes
        n_input_channels = cf.end_filts
        n_features = cf.n_rpn_features
        n_output_channels = cf.n_anchors_per_pos * cf.head_classes
        anchor_stride = cf.rpn_anchor_stride

        self.conv_1 = conv(n_input_channels, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu, norm=cf.norm)
        self.conv_2 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu, norm=cf.norm)
        self.conv_3 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu, norm=cf.norm)
        self.conv_4 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu, norm=cf.norm)
        self.conv_final = conv(n_features, n_output_channels, ks=3, stride=anchor_stride, pad=1, relu=None)


    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: class_logits (b, n_anchors, n_classes)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)

        class_logits = self.conv_final(x)
        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        class_logits = class_logits.permute(*axes)
        class_logits = class_logits.contiguous()
        class_logits = class_logits.view(x.shape[0], -1, self.n_classes)

        return [class_logits]

class BBRegressor(nn.Module):


    def __init__(self, cf, conv):
        """
        Builds the bb-regression sub-network.
        """
        super(BBRegressor, self).__init__()
        self.dim = conv.dim
        n_input_channels = cf.end_filts
        n_features = cf.n_rpn_features
        n_output_channels = cf.n_anchors_per_pos * self.dim * 2
        anchor_stride = cf.rpn_anchor_stride

        self.conv_1 = conv(n_input_channels, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu, norm=cf.norm)
        self.conv_2 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu, norm=cf.norm)
        self.conv_3 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu, norm=cf.norm)
        self.conv_4 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu, norm=cf.norm)
        self.conv_final = conv(n_features, n_output_channels, ks=3, stride=anchor_stride, pad=1, relu=None)

    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: bb_logits (b, n_anchors, dim * 2)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        bb_logits = self.conv_final(x)

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        bb_logits = bb_logits.permute(*axes)
        bb_logits = bb_logits.contiguous()
        bb_logits = bb_logits.view(x.shape[0], -1, self.dim * 2)

        return [bb_logits]


class RoIRegressor(nn.Module):


    def __init__(self, cf, conv, rg_feats):
        """
        Builds the RoI-item-regression sub-network. Regression items can be, e.g., malignancy scores of tumors.
        """
        super(RoIRegressor, self).__init__()
        self.dim = conv.dim
        n_input_channels = cf.end_filts
        n_features = cf.n_rpn_features
        self.rg_feats = rg_feats
        n_output_channels = cf.n_anchors_per_pos * self.rg_feats
        anchor_stride = cf.rpn_anchor_stride
        self.conv_1 = conv(n_input_channels, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu, norm=cf.norm)
        self.conv_2 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu, norm=cf.norm)
        self.conv_3 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu, norm=cf.norm)
        self.conv_4 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu, norm=cf.norm)
        self.conv_final = conv(n_features, n_output_channels, ks=3, stride=anchor_stride,
                               pad=1, relu=None)

    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: bb_logits (b, n_anchors, dim * 2)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_final(x)

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        x = x.permute(*axes)
        x = x.contiguous()
        x = x.view(x.shape[0], -1, self.rg_feats)

        return [x]



############################################################
#  Loss Functions
############################################################
#
def compute_class_loss(anchor_matches, class_pred_logits, shem_poolsize=20):
    """
    :param anchor_matches: (n_anchors). [-1, 0, 1] for negative, neutral, and positive matched anchors.
    :param class_pred_logits: (n_anchors, n_classes). logits from classifier sub-network.
    :param shem_poolsize: int. factor of top-k candidates to draw from per negative sample (online-hard-example-mining).
    :return: loss: torch tensor
    :return: np_neg_ix: 1D array containing indices of the neg_roi_logits, which have been sampled for training.
    """
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    pos_indices = torch.nonzero(anchor_matches > 0)
    neg_indices = torch.nonzero(anchor_matches == -1)

    # get positive samples and calucalte loss.
    if not 0 in pos_indices.size():
        pos_indices = pos_indices.squeeze(1)
        roi_logits_pos = class_pred_logits[pos_indices]
        targets_pos = anchor_matches[pos_indices].detach()
        pos_loss = F.cross_entropy(roi_logits_pos, targets_pos.long())
    else:
        pos_loss = torch.FloatTensor([0]).cuda()

    # get negative samples, such that the amount matches the number of positive samples, but at least 1.
    # get high scoring negatives by applying online-hard-example-mining.
    if not 0 in neg_indices.size():
        neg_indices = neg_indices.squeeze(1)
        roi_logits_neg = class_pred_logits[neg_indices]
        negative_count = np.max((1, pos_indices.cpu().data.numpy().size))
        roi_probs_neg = F.softmax(roi_logits_neg, dim=1)
        neg_ix = mutils.shem(roi_probs_neg, negative_count, shem_poolsize)
        neg_loss = F.cross_entropy(roi_logits_neg[neg_ix], torch.LongTensor([0] * neg_ix.shape[0]).cuda())
        # return the indices of negative samples, who contributed to the loss for monitoring plots.
        np_neg_ix = neg_ix.cpu().data.numpy()
    else:
        neg_loss = torch.FloatTensor([0]).cuda()
        np_neg_ix = np.array([]).astype('int32')

    loss = (pos_loss + neg_loss) / 2
    return loss, np_neg_ix


def compute_bbox_loss(target_deltas, pred_deltas, anchor_matches):
    """
    :param target_deltas:   (b, n_positive_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd)))).
    Uses 0 padding to fill in unused bbox deltas.
    :param pred_deltas: predicted deltas from bbox regression head. (b, n_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd))))
    :param anchor_matches: tensor (n_anchors). value in [-1, 0, class_ids] for negative, neutral, and positive matched anchors.
        i.e., positively matched anchors are marked by class_id >0
    :return: loss: torch 1D tensor.
    """
    if not 0 in torch.nonzero(anchor_matches>0).shape:
        indices = torch.nonzero(anchor_matches>0).squeeze(1)

        # Pick bbox deltas that contribute to the loss
        pred_deltas = pred_deltas[indices]
        # Trim target bounding box deltas to the same length as pred_deltas.
        target_deltas = target_deltas[:pred_deltas.shape[0], :].detach()
        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_deltas, target_deltas)
    else:
        loss = torch.FloatTensor([0]).cuda()

    return loss

def compute_rg_loss(tasks, target, pred, anchor_matches):
    """
    :param target_deltas:   (b, n_positive_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd)))).
    Uses 0 padding to fill in unsed bbox deltas.
    :param pred_deltas: predicted deltas from bbox regression head. (b, n_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd))))
    :param anchor_matches: (n_anchors). [-1, 0, 1] for negative, neutral, and positive matched anchors.
    :return: loss: torch 1D tensor.
    """
    if not 0 in target.shape and not 0 in torch.nonzero(anchor_matches>0).shape:
        indices = torch.nonzero(anchor_matches>0).squeeze(1)
        # Pick rgs that contribute to the loss
        pred = pred[indices]
        # Trim target
        target = target[:pred.shape[0]].detach()
        if 'regression_bin' in tasks:
            loss = F.cross_entropy(pred, target.long())
        else:
            loss = F.smooth_l1_loss(pred, target)
    else:
        loss = torch.FloatTensor([0]).cuda()

    return loss

def compute_focal_class_loss(anchor_matches, class_pred_logits, gamma=2.):
    """ Focal Loss :math:`FL = -(1-q)^g log(q)` with q = pred class probability, g = gamma hyperparameter.

    :param anchor_matches: (n_anchors). [-1, 0, class] for negative, neutral, and positive matched anchors.
    :param class_pred_logits: (n_anchors, n_classes). logits from classifier sub-network.
    :param gamma: g in above formula, good results with g=2 in original paper.
    :return: loss: torch tensor
    :return: focal loss
    """
    # Positive and Negative anchors contribute to the loss but neutral anchors (match value = 0) don't.
    pos_indices = torch.nonzero(anchor_matches > 0).squeeze(-1) # dim=-1 instead of 1 or 0 to cover empty matches.
    neg_indices = torch.nonzero(anchor_matches == -1).squeeze(-1)
    target_classes  = torch.cat( (anchor_matches[pos_indices].long(), torch.LongTensor([0] * neg_indices.shape[0]).cuda()) )

    non_neutral_indices = torch.cat( (pos_indices, neg_indices) )
    # q shape: (n_non_neutral_anchors, n_classes)
    q = F.softmax(class_pred_logits[non_neutral_indices], dim=1)

    # one-hot encoded target classes: keep only the pred probs of the correct class.
    # that class will receive the incentive to be maximized.
    # log(q_i) where i = target class --> FL shape (n_anchors,)
    # need to transform to indices into flattened tensor to use torch.take
    target_locs_flat = q.shape[1] * torch.arange(q.shape[0]).cuda() + target_classes
    q = torch.take(q, target_locs_flat)

    FL = torch.log(q) # element-wise log
    FL *= -(1.-q)**gamma

    # take mean over all considered anchors
    FL = FL.sum() / FL.shape[0]
    return FL



def refine_detections(anchors, probs, deltas, regressions, batch_ixs, cf):
    """Refine classified proposals, filter overlaps and return final
    detections. n_proposals here is typically a very large number: batch_size * n_anchors.
    This function is hence optimized on trimming down n_proposals.
    :param anchors: (n_anchors, 2 * dim)
    :param probs: (n_proposals, n_classes) softmax probabilities for all rois as predicted by classifier head.
    :param deltas: (n_proposals, n_classes, 2 * dim) box refinement deltas as predicted by bbox regressor head.
    :param regressions: (n_proposals, n_classes, n_rg_feats)
    :param batch_ixs: (n_proposals) batch element assignemnt info for re-allocation.
    :return: result: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score, pred_regr))
    """
    anchors = anchors.repeat(batch_ixs.unique().shape[0], 1)

    #flatten foreground probabilities, sort and trim down to highest confidences by pre_nms limit.
    fg_probs = probs[:, 1:].contiguous()
    flat_probs, flat_probs_order = fg_probs.view(-1).sort(descending=True)
    keep_ix = flat_probs_order[:cf.pre_nms_limit]
    # reshape indices to 2D index array with shape like fg_probs.
    keep_arr = torch.cat(((keep_ix / fg_probs.shape[1]).unsqueeze(1), (keep_ix % fg_probs.shape[1]).unsqueeze(1)), 1)

    pre_nms_scores = flat_probs[:cf.pre_nms_limit]
    pre_nms_class_ids = keep_arr[:, 1] + 1 # add background again.
    pre_nms_batch_ixs = batch_ixs[keep_arr[:, 0]]
    pre_nms_anchors = anchors[keep_arr[:, 0]]
    pre_nms_deltas = deltas[keep_arr[:, 0]]
    pre_nms_regressions = regressions[keep_arr[:, 0]]
    keep = torch.arange(pre_nms_scores.size()[0]).long().cuda()

    # apply bounding box deltas. re-scale to image coordinates.
    std_dev = torch.from_numpy(np.reshape(cf.rpn_bbox_std_dev, [1, cf.dim * 2])).float().cuda()
    scale = torch.from_numpy(cf.scale).float().cuda()
    refined_rois = mutils.apply_box_deltas_2D(pre_nms_anchors / scale, pre_nms_deltas * std_dev) * scale \
        if cf.dim == 2 else mutils.apply_box_deltas_3D(pre_nms_anchors / scale, pre_nms_deltas * std_dev) * scale

    # round and cast to int since we're deadling with pixels now
    refined_rois = mutils.clip_to_window(cf.window, refined_rois)
    pre_nms_rois = torch.round(refined_rois)
    for j, b in enumerate(mutils.unique1d(pre_nms_batch_ixs)):

        bixs = torch.nonzero(pre_nms_batch_ixs == b)[:, 0]
        bix_class_ids = pre_nms_class_ids[bixs]
        bix_rois = pre_nms_rois[bixs]
        bix_scores = pre_nms_scores[bixs]

        for i, class_id in enumerate(mutils.unique1d(bix_class_ids)):

            ixs = torch.nonzero(bix_class_ids == class_id)[:, 0]
            # nms expects boxes sorted by score.
            ix_rois = bix_rois[ixs]
            ix_scores = bix_scores[ixs]
            ix_scores, order = ix_scores.sort(descending=True)
            ix_rois = ix_rois[order, :]
            ix_scores = ix_scores

            class_keep = nms.nms(ix_rois, ix_scores, cf.detection_nms_threshold)
            # map indices back.
            class_keep = keep[bixs[ixs[order[class_keep]]]]
            # merge indices over classes for current batch element
            b_keep = class_keep if i == 0 else mutils.unique1d(torch.cat((b_keep, class_keep)))

        # only keep top-k boxes of current batch-element.
        top_ids = pre_nms_scores[b_keep].sort(descending=True)[1][:cf.model_max_instances_per_batch_element]
        b_keep = b_keep[top_ids]
        # merge indices over batch elements.
        batch_keep = b_keep if j == 0 else mutils.unique1d(torch.cat((batch_keep, b_keep)))

    keep = batch_keep

    # arrange output.
    result = torch.cat((pre_nms_rois[keep],
                        pre_nms_batch_ixs[keep].unsqueeze(1).float(),
                        pre_nms_class_ids[keep].unsqueeze(1).float(),
                        pre_nms_scores[keep].unsqueeze(1),
                        pre_nms_regressions[keep]), dim=1)

    return result



def gt_anchor_matching(cf, anchors, gt_boxes, gt_class_ids=None, gt_regressions=None):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2, (z1), (z2))]
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2, (z1), (z2))]
    gt_class_ids (optional): [num_gt_boxes] Integer class IDs for one stage detectors. in RPN case of Mask R-CNN,
    set all positive matches to 1 (foreground)
    gt_regressions: [num_gt_rgs, n_rg_feats], if None empty rg_targets are returned

    Returns:
    anchor_class_matches: [N] (int32) matches between anchors and GT boxes. class_id = positive anchor,
     -1 = negative anchor, 0 = neutral. i.e., positively matched anchors are marked by class_id (which is >0).
    anchor_delta_targets: [N, (dy, dx, (dz), log(dh), log(dw), (log(dd)))] Anchor bbox deltas.
    anchor_rg_targets: [n_anchors, n_rg_feats]
    """

    anchor_class_matches = np.zeros([anchors.shape[0]], dtype=np.int32)
    anchor_delta_targets = np.zeros((cf.rpn_train_anchors_per_image, 2*cf.dim))
    if gt_regressions is not None:
        if 'regression_bin' in cf.prediction_tasks:
            anchor_rg_targets = np.zeros((cf.rpn_train_anchors_per_image,))
        else:
            anchor_rg_targets = np.zeros((cf.rpn_train_anchors_per_image,  cf.regression_n_features))
    else:
        anchor_rg_targets = np.array([])

    anchor_matching_iou = cf.anchor_matching_iou

    if gt_boxes is None:
        anchor_class_matches = np.full(anchor_class_matches.shape, fill_value=-1)
        return anchor_class_matches, anchor_delta_targets, anchor_rg_targets

    # for mrcnn: anchor matching is done for RPN loss, so positive labels are all 1 (foreground)
    if gt_class_ids is None:
        gt_class_ids = np.array([1] * len(gt_boxes))

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = mutils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= anchor_matching_iou then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.1 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.1).

    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    if anchors.shape[1] == 4:
        anchor_class_matches[(anchor_iou_max < 0.1)] = -1
    elif anchors.shape[1] == 6:
        anchor_class_matches[(anchor_iou_max < 0.01)] = -1
    else:
        raise ValueError('anchor shape wrong {}'.format(anchors.shape))

    # 2. Set an anchor for each GT box (regardless of IoU value).
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    for ix, ii in enumerate(gt_iou_argmax):
        anchor_class_matches[ii] = gt_class_ids[ix]

    # 3. Set anchors with high overlap as positive.
    above_thresh_ixs = np.argwhere(anchor_iou_max >= anchor_matching_iou)
    anchor_class_matches[above_thresh_ixs] = gt_class_ids[anchor_iou_argmax[above_thresh_ixs]]

    # Subsample to balance positive anchors.
    ids = np.where(anchor_class_matches > 0)[0]
    extra = len(ids) - (cf.rpn_train_anchors_per_image // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        anchor_class_matches[ids] = 0

    # Leave all negative proposals negative for now and sample from them later in online hard example mining.
    # For positive anchors, compute shift and scale needed to transform them to match the corresponding GT boxes.
    ids = np.where(anchor_class_matches > 0)[0]
    ix = 0  # index into anchor_delta_targets
    for i, a in zip(ids, anchors[ids]):
        # closest gt box (it might have IoU < anchor_matching_iou)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # convert coordinates to center plus width/height.
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        if cf.dim == 2:
            anchor_delta_targets[ix] = [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                np.log(gt_h / a_h),
                np.log(gt_w / a_w)]
        else:
            gt_d = gt[5] - gt[4]
            gt_center_z = gt[4] + 0.5 * gt_d
            a_d = a[5] - a[4]
            a_center_z = a[4] + 0.5 * a_d
            anchor_delta_targets[ix] = [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                (gt_center_z - a_center_z) / a_d,
                np.log(gt_h / a_h),
                np.log(gt_w / a_w),
                np.log(gt_d / a_d)]

        # normalize.
        anchor_delta_targets[ix] /= cf.rpn_bbox_std_dev
        if gt_regressions is not None:
            anchor_rg_targets[ix] = gt_regressions[anchor_iou_argmax[i]]

        ix += 1

    return anchor_class_matches, anchor_delta_targets, anchor_rg_targets

############################################################
#  RetinaNet Class
############################################################


class net(nn.Module):
    """Encapsulates the RetinaNet model functionality.
    """

    def __init__(self, cf, logger):
        """
        cf: A Sub-class of the cf class
        model_dir: Directory to save training logs and trained weights
        """
        super(net, self).__init__()
        self.cf = cf
        self.logger = logger
        self.build()
        if self.cf.weight_init is not None:
            mutils.initialize_weights(self)
        else:
            logger.info("using default pytorch weight init")

        self.debug_acm = []

    def build(self):
        """Build Retina Net architecture."""

        # Image size must be dividable by 2 multiple times.
        h, w = self.cf.patch_size[:2]
        if h / 2 ** 5 != int(h / 2 ** 5) or w / 2 ** 5 != int(w / 2 ** 5):
            raise Exception("Image size must be divisible by 2 at least 5 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        backbone = utils.import_module('bbone', self.cf.backbone_path)
        self.logger.info("loaded backbone from {}".format(self.cf.backbone_path))
        conv = backbone.ConvGenerator(self.cf.dim)


        # build Anchors, FPN, Classifier / Bbox-Regressor -head
        self.np_anchors = mutils.generate_pyramid_anchors(self.logger, self.cf)
        self.anchors = torch.from_numpy(self.np_anchors).float().cuda()
        self.fpn = backbone.FPN(self.cf, conv, operate_stride1=self.cf.operate_stride1).cuda()
        self.classifier = Classifier(self.cf, conv).cuda()
        self.bb_regressor = BBRegressor(self.cf, conv).cuda()

        if 'regression' in self.cf.prediction_tasks:
            self.roi_regressor = RoIRegressor(self.cf, conv, self.cf.regression_n_features).cuda()
        elif 'regression_bin' in self.cf.prediction_tasks:
            # classify into bins of regression values
            self.roi_regressor = RoIRegressor(self.cf, conv, len(self.cf.bin_labels)).cuda()
        else:
            self.roi_regressor = lambda x: [torch.tensor([]).cuda()]

        if self.cf.model == 'retina_unet':
            self.final_conv = conv(self.cf.end_filts, self.cf.num_seg_classes, ks=1, pad=0, norm=None, relu=None)

    def forward(self, img):
        """
        :param img: input img (b, c, y, x, (z)).
        """
        # Feature extraction
        fpn_outs = self.fpn(img)
        if self.cf.model == 'retina_unet':
            seg_logits = self.final_conv(fpn_outs[0])
            selected_fmaps = [fpn_outs[i + 1] for i in self.cf.pyramid_levels]
        else:
            seg_logits = None
            selected_fmaps = [fpn_outs[i] for i in self.cf.pyramid_levels]

        # Loop through pyramid layers
        class_layer_outputs, bb_reg_layer_outputs, roi_reg_layer_outputs = [], [], []  # list of lists
        for p in selected_fmaps:
            class_layer_outputs.append(self.classifier(p))
            bb_reg_layer_outputs.append(self.bb_regressor(p))
            roi_reg_layer_outputs.append(self.roi_regressor(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        class_logits = list(zip(*class_layer_outputs))
        class_logits = [torch.cat(list(o), dim=1) for o in class_logits][0]
        bb_outputs = list(zip(*bb_reg_layer_outputs))
        bb_outputs = [torch.cat(list(o), dim=1) for o in bb_outputs][0]
        if not 0 == roi_reg_layer_outputs[0][0].shape[0]:
            rg_outputs = list(zip(*roi_reg_layer_outputs))
            rg_outputs = [torch.cat(list(o), dim=1) for o in rg_outputs][0]
        else:
            if self.cf.dim == 2:
                n_feats = np.array([p.shape[-2] * p.shape[-1] * self.cf.n_anchors_per_pos for p in selected_fmaps]).sum()
            else:
                n_feats = np.array([p.shape[-3]*p.shape[-2]*p.shape[-1]*self.cf.n_anchors_per_pos for p in selected_fmaps]).sum()
            rg_outputs = torch.zeros((selected_fmaps[0].shape[0], n_feats, self.cf.regression_n_features),
                                     dtype=torch.float32).fill_(float('NaN')).cuda()

        # merge batch_dimension and store info in batch_ixs for re-allocation.
        batch_ixs = torch.arange(class_logits.shape[0]).unsqueeze(1).repeat(1, class_logits.shape[1]).view(-1).cuda()
        flat_class_softmax = F.softmax(class_logits.view(-1, class_logits.shape[-1]), 1)
        flat_bb_outputs = bb_outputs.view(-1, bb_outputs.shape[-1])
        flat_rg_outputs = rg_outputs.view(-1, rg_outputs.shape[-1])

        detections = refine_detections(self.anchors, flat_class_softmax, flat_bb_outputs, flat_rg_outputs, batch_ixs,
                                       self.cf)

        return detections, class_logits, bb_outputs, rg_outputs, seg_logits


    def get_results(self, img_shape, detections, seg_logits, box_results_list=None):
        """
        Restores batch dimension of merged detections, unmolds detections, creates and fills results dict.
        :param img_shape:
        :param detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score,
            pred_regression)
        :param box_results_list: None or list of output boxes for monitoring/plotting.
        each element is a list of boxes per batch element.
        :return: results_dict: dictionary with keys:
                 'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                          [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
                 'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, 1] only fg. vs. bg for now.
                 class-specific return of masks will come with implementation of instance segmentation evaluation.
        """
        detections = detections.cpu().data.numpy()
        batch_ixs = detections[:, self.cf.dim*2]
        detections = [detections[batch_ixs == ix] for ix in range(img_shape[0])]

        if box_results_list == None:  # for test_forward, where no previous list exists.
            box_results_list = [[] for _ in range(img_shape[0])]

        for ix in range(img_shape[0]):

            if not 0 in detections[ix].shape:

                boxes = detections[ix][:, :2 * self.cf.dim].astype(np.int32)
                class_ids = detections[ix][:, 2 * self.cf.dim + 1].astype(np.int32)
                scores = detections[ix][:, 2 * self.cf.dim + 2]
                regressions = detections[ix][:, 2 * self.cf.dim + 3:]

                # Filter out detections with zero area. Often only happens in early
                # stages of training when the network weights are still a bit random.
                if self.cf.dim == 2:
                    exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
                else:
                    exclude_ix = np.where(
                        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4]) <= 0)[0]

                if exclude_ix.shape[0] > 0:
                    boxes = np.delete(boxes, exclude_ix, axis=0)
                    class_ids = np.delete(class_ids, exclude_ix, axis=0)
                    scores = np.delete(scores, exclude_ix, axis=0)
                    regressions = np.delete(regressions, exclude_ix, axis=0)

                if not 0 in boxes.shape:
                    for ix2, score in enumerate(scores):
                        if score >= self.cf.model_min_confidence:
                            box = {'box_type': 'det', 'box_coords': boxes[ix2], 'box_score': score,
                                   'box_pred_class_id': class_ids[ix2]}
                            if "regression_bin" in self.cf.prediction_tasks:
                                # in this case, regression preds are actually the rg_bin_ids --> map to rg value the bin stands for
                                box['rg_bin'] = regressions[ix2].argmax()
                                box['regression'] = self.cf.bin_id2rg_val[box['rg_bin']]
                            else:
                                box['regression'] = regressions[ix2]
                                if hasattr(self.cf, "rg_val_to_bin_id") and \
                                        any(['regression' in task for task in self.cf.prediction_tasks]):
                                    box['rg_bin'] = self.cf.rg_val_to_bin_id(regressions[ix2])
                            box_results_list[ix].append(box)


        results_dict = {}
        results_dict['boxes'] = box_results_list
        if seg_logits is None:
            # output dummy segmentation for retina_net.
            out_logits_shape = list(img_shape)
            out_logits_shape[1] = self.cf.num_seg_classes
            results_dict['seg_preds'] = np.zeros(out_logits_shape, dtype=np.float16)
            #todo: try with seg_preds=None? as to not carry heavy dummy preds.
        else:
            # output label maps for retina_unet.
            results_dict['seg_preds'] = F.softmax(seg_logits, 1).cpu().data.numpy()

        return results_dict


    def train_forward(self, batch, is_validation=False):
        """
        train method (also used for validation monitoring). wrapper around forward pass of network. prepares input data
        for processing, computes losses, and stores outputs in a dictionary.
        :param batch: dictionary containing 'data', 'seg', etc.
        :return: results_dict: dictionary with keys:
                'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                        [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
                'seg_preds': pixelwise segmentation output (b, c, y, x, (z)) with values [0, .., n_classes].
                'torch_loss': 1D torch tensor for backprop.
                'class_loss': classification loss for monitoring.
        """
        img = batch['data']
        gt_class_ids = batch['class_targets']
        gt_boxes = batch['bb_target']
        if 'regression' in self.cf.prediction_tasks:
            gt_regressions = batch["regression_targets"]
        elif 'regression_bin' in self.cf.prediction_tasks:
            gt_regressions = batch["rg_bin_targets"]
        else:
            gt_regressions = None
        if self.cf.model == 'retina_unet':
            var_seg_ohe = torch.FloatTensor(mutils.get_one_hot_encoding(batch['seg'], self.cf.num_seg_classes)).cuda()
            var_seg = torch.LongTensor(batch['seg']).cuda()

        img = torch.from_numpy(img).float().cuda()
        torch_loss = torch.FloatTensor([0]).cuda()

        # list of output boxes for monitoring/plotting. each element is a list of boxes per batch element.
        box_results_list = [[] for _ in range(img.shape[0])]
        detections, class_logits, pred_deltas, pred_rgs, seg_logits = self.forward(img)
        # loop over batch
        for b in range(img.shape[0]):
            # add gt boxes to results dict for monitoring.
            if len(gt_boxes[b]) > 0:
                for tix in range(len(gt_boxes[b])):
                    gt_box = {'box_type': 'gt', 'box_coords': batch['bb_target'][b][tix]}
                    for name in self.cf.roi_items:
                        gt_box.update({name: batch[name][b][tix]})
                    box_results_list[b].append(gt_box)

                # match gt boxes with anchors to generate targets.
                anchor_class_match, anchor_target_deltas, anchor_target_rgs = gt_anchor_matching(
                    self.cf, self.np_anchors, gt_boxes[b], gt_class_ids[b], gt_regressions[b] if gt_regressions is not None else None)

                # add positive anchors used for loss to results_dict for monitoring.
                pos_anchors = mutils.clip_boxes_numpy(
                    self.np_anchors[np.argwhere(anchor_class_match > 0)][:, 0], img.shape[2:])
                for p in pos_anchors:
                    box_results_list[b].append({'box_coords': p, 'box_type': 'pos_anchor'})

            else:
                anchor_class_match = np.array([-1]*self.np_anchors.shape[0])
                anchor_target_deltas = np.array([])
                anchor_target_rgs = np.array([])

            anchor_class_match = torch.from_numpy(anchor_class_match).cuda()
            anchor_target_deltas = torch.from_numpy(anchor_target_deltas).float().cuda()
            anchor_target_rgs = torch.from_numpy(anchor_target_rgs).float().cuda()

            if self.cf.focal_loss:
                # compute class loss as focal loss as suggested in original publication, but multi-class.
                class_loss = compute_focal_class_loss(anchor_class_match, class_logits[b], gamma=self.cf.focal_loss_gamma)
                # sparing appendix of negative anchors for monitoring as not really relevant
            else:
                # compute class loss with SHEM.
                class_loss, neg_anchor_ix = compute_class_loss(anchor_class_match, class_logits[b])
                # add negative anchors used for loss to results_dict for monitoring.
                neg_anchors = mutils.clip_boxes_numpy(
                    self.np_anchors[np.argwhere(anchor_class_match.cpu().numpy() == -1)][neg_anchor_ix, 0],
                    img.shape[2:])
                for n in neg_anchors:
                    box_results_list[b].append({'box_coords': n, 'box_type': 'neg_anchor'})
            rg_loss = compute_rg_loss(self.cf.prediction_tasks, anchor_target_rgs, pred_rgs[b], anchor_class_match)
            bbox_loss = compute_bbox_loss(anchor_target_deltas, pred_deltas[b], anchor_class_match)
            torch_loss += (class_loss + bbox_loss + rg_loss) / img.shape[0]


        results_dict = self.get_results(img.shape, detections, seg_logits, box_results_list)
        results_dict['seg_preds'] = results_dict['seg_preds'].argmax(axis=1).astype('uint8')[:, np.newaxis]

        if self.cf.model == 'retina_unet':
            seg_loss_dice = 1 - mutils.batch_dice(F.softmax(seg_logits, dim=1),var_seg_ohe)
            seg_loss_ce = F.cross_entropy(seg_logits, var_seg[:, 0])
            torch_loss += (seg_loss_dice + seg_loss_ce) / 2
            #self.logger.info("loss: {0:.2f}, class: {1:.2f}, bbox: {2:.2f}, seg dice: {3:.3f}, seg ce: {4:.3f}, "
            #                 "mean pixel preds: {5:.5f}".format(torch_loss.item(), batch_class_loss.item(), batch_bbox_loss.item(),
            #                                                   seg_loss_dice.item(), seg_loss_ce.item(), np.mean(results_dict['seg_preds'])))
        if 'dice' in self.cf.metrics:
            results_dict['batch_dices'] = mutils.dice_per_batch_and_class(
                results_dict['seg_preds'], batch["seg"], self.cf.num_seg_classes, convert_to_ohe=True)
        #else:
            #self.logger.info("loss: {0:.2f}, class: {1:.2f}, bbox: {2:.2f}".format(
        #        torch_loss.item(), class_loss.item(), bbox_loss.item()))


        results_dict['torch_loss'] = torch_loss
        results_dict['class_loss'] = class_loss.item()

        return results_dict

    def test_forward(self, batch, **kwargs):
        """
        test method. wrapper around forward pass of network without usage of any ground truth information.
        prepares input data for processing and stores outputs in a dictionary.
        :param batch: dictionary containing 'data'
        :return: results_dict: dictionary with keys:
               'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                       [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
               'seg_preds': actually contain seg probabilities since evaluated to seg_preds (via argmax) in predictor.
                or dummy seg logits for real retina net (detection only)
        """
        img = torch.from_numpy(batch['data']).float().cuda()
        detections, _, _, _, seg_logits = self.forward(img)
        results_dict = self.get_results(img.shape, detections, seg_logits)
        return results_dict