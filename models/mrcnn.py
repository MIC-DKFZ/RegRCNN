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

"""
Parts are based on https://github.com/multimodallearning/pytorch-mask-rcnn
published under MIT license.
"""
import os
from multiprocessing import  Pool
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

import utils.model_utils as mutils
import utils.exp_utils as utils



class RPN(nn.Module):
    """
    Region Proposal Network.
    """

    def __init__(self, cf, conv):

        super(RPN, self).__init__()
        self.dim = conv.dim

        self.conv_shared = conv(cf.end_filts, cf.n_rpn_features, ks=3, stride=cf.rpn_anchor_stride, pad=1, relu=cf.relu)
        self.conv_class = conv(cf.n_rpn_features, 2 * len(cf.rpn_anchor_ratios), ks=1, stride=1, relu=None)
        self.conv_bbox = conv(cf.n_rpn_features, 2 * self.dim * len(cf.rpn_anchor_ratios), ks=1, stride=1, relu=None)


    def forward(self, x):
        """
        :param x: input feature maps (b, in_channels, y, x, (z))
        :return: rpn_class_logits (b, 2, n_anchors)
        :return: rpn_probs_logits (b, 2, n_anchors)
        :return: rpn_bbox (b, 2 * dim, n_anchors)
        """

        # Shared convolutional base of the RPN.
        x = self.conv_shared(x)

        # Anchor Score. (batch, anchors per location * 2, y, x, (z)).
        rpn_class_logits = self.conv_class(x)
        # Reshape to (batch, 2, anchors)
        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        rpn_class_logits = rpn_class_logits.permute(*axes)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        # Softmax on last dimension (fg vs. bg).
        rpn_probs = F.softmax(rpn_class_logits, dim=2)

        # Bounding box refinement. (batch, anchors_per_location * (y, x, (z), log(h), log(w), (log(d)), y, x, (z))
        rpn_bbox = self.conv_bbox(x)

        # Reshape to (batch, 2*dim, anchors)
        rpn_bbox = rpn_bbox.permute(*axes)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, self.dim * 2)

        return [rpn_class_logits, rpn_probs, rpn_bbox]



class Classifier(nn.Module):
    """
    Head network for classification and bounding box refinement. Performs RoiAlign, processes resulting features through a
    shared convolutional base and finally branches off the classifier- and regression head.
    """
    def __init__(self, cf, conv):
        super(Classifier, self).__init__()

        self.cf = cf
        self.dim = conv.dim
        self.in_channels = cf.end_filts
        self.pool_size = cf.pool_size
        self.pyramid_levels = cf.pyramid_levels
        # instance_norm does not work with spatial dims (1, 1, (1))
        norm = cf.norm if cf.norm != 'instance_norm' else None

        self.conv1 = conv(cf.end_filts, cf.end_filts * 4, ks=self.pool_size, stride=1, norm=norm, relu=cf.relu)
        self.conv2 = conv(cf.end_filts * 4, cf.end_filts * 4, ks=1, stride=1, norm=norm, relu=cf.relu)
        self.linear_bbox = nn.Linear(cf.end_filts * 4, cf.head_classes * 2 * self.dim)


        if 'regression' in self.cf.prediction_tasks:
            self.linear_regressor = nn.Linear(cf.end_filts * 4, cf.head_classes * cf.regression_n_features)
            self.rg_n_feats = cf.regression_n_features
        #classify into bins of regression values
        elif 'regression_bin' in self.cf.prediction_tasks:
            self.linear_regressor = nn.Linear(cf.end_filts * 4, cf.head_classes * len(cf.bin_labels))
            self.rg_n_feats = len(cf.bin_labels)
        else:
            self.linear_regressor = lambda x: torch.zeros((x.shape[0], cf.head_classes * 1), dtype=torch.float32).fill_(float('NaN')).cuda()
            self.rg_n_feats = 1 #cf.regression_n_features
        if 'class' in self.cf.prediction_tasks:
            self.linear_class = nn.Linear(cf.end_filts * 4, cf.head_classes)
        else:
            assert cf.head_classes == 2, "#head classes {} needs to be 2 (bg/fg) when not predicting classes".format(cf.head_classes)
            self.linear_class = lambda x: torch.zeros((x.shape[0], cf.head_classes), dtype=torch.float64).cuda()


    def forward(self, x, rois):
        """
        :param x: input feature maps (b, in_channels, y, x, (z))
        :param rois: normalized box coordinates as proposed by the RPN to be forwarded through
        the second stage (n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ix). Proposals of all batch elements
        have been merged to one vector, while the origin info has been stored for re-allocation.
        :return: mrcnn_class_logits (n_proposals, n_head_classes)
        :return: mrcnn_bbox (n_proposals, n_head_classes, 2 * dim) predicted corrections to be applied to proposals for refinement.
        """
        x = mutils.pyramid_roi_align(x, rois, self.pool_size, self.pyramid_levels, self.dim)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.in_channels * 4)

        mrcnn_bbox = self.linear_bbox(x)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, self.dim * 2)
        mrcnn_class_logits = self.linear_class(x)
        mrcnn_regress = self.linear_regressor(x)
        mrcnn_regress = mrcnn_regress.view(mrcnn_regress.size()[0], -1, self.rg_n_feats)

        return [mrcnn_bbox, mrcnn_class_logits, mrcnn_regress]


class Mask(nn.Module):
    """
    Head network for proposal-based mask segmentation. Performs RoiAlign, some convolutions and applies sigmoid on the
    output logits to allow for overlapping classes.
    """
    def __init__(self, cf, conv):
        super(Mask, self).__init__()
        self.pool_size = cf.mask_pool_size
        self.pyramid_levels = cf.pyramid_levels
        self.dim = conv.dim
        self.conv1 = conv(cf.end_filts, cf.end_filts, ks=3, stride=1, pad=1, norm=cf.norm, relu=cf.relu)
        self.conv2 = conv(cf.end_filts, cf.end_filts, ks=3, stride=1, pad=1, norm=cf.norm, relu=cf.relu)
        self.conv3 = conv(cf.end_filts, cf.end_filts, ks=3, stride=1, pad=1, norm=cf.norm, relu=cf.relu)
        self.conv4 = conv(cf.end_filts, cf.end_filts, ks=3, stride=1, pad=1, norm=cf.norm, relu=cf.relu)
        if conv.dim == 2:
            self.deconv = nn.ConvTranspose2d(cf.end_filts, cf.end_filts, kernel_size=2, stride=2) # todo why no norm here?
        else:
            self.deconv = nn.ConvTranspose3d(cf.end_filts, cf.end_filts, kernel_size=2, stride=2)

        self.relu = nn.ReLU(inplace=True) if cf.relu == 'relu' else nn.LeakyReLU(inplace=True)
        self.conv5 = conv(cf.end_filts, cf.head_classes, ks=1, stride=1, relu=None)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, rois):
        """
        :param x: input feature maps (b, in_channels, y, x, (z))
        :param rois: normalized box coordinates as proposed by the RPN to be forwarded through
        the second stage (n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ix). Proposals of all batch elements
        have been merged to one vector, while the origin info has been stored for re-allocation.
        :return: x: masks (n_sampled_proposals (n_detections in inference), n_classes, y, x, (z))
        """
        x = mutils.pyramid_roi_align(x, rois, self.pool_size, self.pyramid_levels, self.dim)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.relu(self.deconv(x))
        x = self.conv5(x)
        x = self.sigmoid(x)
        return x


############################################################
#  Loss Functions
############################################################

def compute_rpn_class_loss(rpn_class_logits, rpn_match, shem_poolsize):
    """
    :param rpn_match: (n_anchors). [-1, 0, 1] for negative, neutral, and positive matched anchors.
    :param rpn_class_logits: (n_anchors, 2). logits from RPN classifier.
    :param SHEM_poolsize: int. factor of top-k candidates to draw from per negative sample (stochastic-hard-example-mining).
    :return: loss: torch tensor
    :return: np_neg_ix: 1D array containing indices of the neg_roi_logits, which have been sampled for training.
    """

    # Filter out netural anchors
    pos_indices = torch.nonzero(rpn_match == 1)
    neg_indices = torch.nonzero(rpn_match == -1)

    # loss for positive samples
    if not 0 in pos_indices.size():
        pos_indices = pos_indices.squeeze(1)
        roi_logits_pos = rpn_class_logits[pos_indices]
        pos_loss = F.cross_entropy(roi_logits_pos, torch.LongTensor([1] * pos_indices.shape[0]).cuda())
    else:
        pos_loss = torch.FloatTensor([0]).cuda()

    # loss for negative samples: draw hard negative examples (SHEM)
    # that match the number of positive samples, but at least 1.
    if not 0 in neg_indices.size():
        neg_indices = neg_indices.squeeze(1)
        roi_logits_neg = rpn_class_logits[neg_indices]
        negative_count = np.max((1, pos_indices.cpu().data.numpy().size))
        roi_probs_neg = F.softmax(roi_logits_neg, dim=1)
        neg_ix = mutils.shem(roi_probs_neg, negative_count, shem_poolsize)
        neg_loss = F.cross_entropy(roi_logits_neg[neg_ix], torch.LongTensor([0] * neg_ix.shape[0]).cuda())
        np_neg_ix = neg_ix.cpu().data.numpy()
        #print("pos, neg count", pos_indices.cpu().data.numpy().size, negative_count)
    else:
        neg_loss = torch.FloatTensor([0]).cuda()
        np_neg_ix = np.array([]).astype('int32')

    loss = (pos_loss + neg_loss) / 2
    return loss, np_neg_ix


def compute_rpn_bbox_loss(rpn_pred_deltas, rpn_target_deltas, rpn_match):
    """
    :param rpn_target_deltas:   (b, n_positive_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd)))).
    Uses 0 padding to fill in unsed bbox deltas.
    :param rpn_pred_deltas: predicted deltas from RPN. (b, n_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd))))
    :param rpn_match: (n_anchors). [-1, 0, 1] for negative, neutral, and positive matched anchors.
    :return: loss: torch 1D tensor.
    """
    if not 0 in torch.nonzero(rpn_match == 1).size():

        indices = torch.nonzero(rpn_match == 1).squeeze(1)
        # Pick bbox deltas that contribute to the loss
        rpn_pred_deltas = rpn_pred_deltas[indices]
        # Trim target bounding box deltas to the same length as rpn_bbox.
        target_deltas = rpn_target_deltas[:rpn_pred_deltas.size()[0], :]
        # Smooth L1 loss
        loss = F.smooth_l1_loss(rpn_pred_deltas, target_deltas)
    else:
        loss = torch.FloatTensor([0]).cuda()

    return loss

def compute_mrcnn_bbox_loss(mrcnn_pred_deltas, mrcnn_target_deltas, target_class_ids):
    """
    :param mrcnn_target_deltas: (n_sampled_rois, (dy, dx, (dz), log(dh), log(dw), (log(dh)))
    :param mrcnn_pred_deltas: (n_sampled_rois, n_classes, (dy, dx, (dz), log(dh), log(dw), (log(dh)))
    :param target_class_ids: (n_sampled_rois)
    :return: loss: torch 1D tensor.
    """
    if not 0 in torch.nonzero(target_class_ids > 0).size():
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix].long()
        target_bbox = mrcnn_target_deltas[positive_roi_ix, :].detach()
        pred_bbox = mrcnn_pred_deltas[positive_roi_ix, positive_roi_class_ids, :]
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = torch.FloatTensor([0]).cuda()

    return loss

def compute_mrcnn_mask_loss(pred_masks, target_masks, target_class_ids):
    """
    :param target_masks: (n_sampled_rois, y, x, (z)) A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    :param pred_masks: (n_sampled_rois, n_classes, y, x, (z)) float32 tensor with values between [0, 1].
    :param target_class_ids: (n_sampled_rois)
    :return: loss: torch 1D tensor.
    """
    #print("targ masks", target_masks.unique(return_counts=True))
    if not 0 in torch.nonzero(target_class_ids > 0).size():
        # Only positive ROIs contribute to the loss. And only
        # the class-specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix].long()
        y_true = target_masks[positive_ix, :, :].detach()
        y_pred = pred_masks[positive_ix, positive_class_ids, :, :]
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = torch.FloatTensor([0]).cuda()

    return loss

def compute_mrcnn_class_loss(tasks, pred_class_logits, target_class_ids):
    """
    :param pred_class_logits: (n_sampled_rois, n_classes)
    :param target_class_ids: (n_sampled_rois) batch dimension was merged into roi dimension.
    :return: loss: torch 1D tensor.
    """
    if 'class' in tasks and not 0 in target_class_ids.size():
        loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
    else:
        loss = torch.FloatTensor([0.]).cuda()

    return loss

def compute_mrcnn_regression_loss(tasks, pred, target, target_class_ids):
    """regression loss is a distance metric between target vector and predicted regression vector.
    :param pred: (n_sampled_rois, n_classes, [n_rg_feats if real regression or 1 if rg_bin task)
    :param target: (n_sampled_rois, [n_rg_feats or n_rg_bins])
    :return: differentiable loss, torch 1D tensor on cuda
    """

    if not 0 in target.shape and not 0 in torch.nonzero(target_class_ids > 0).shape:
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix].long()
        target = target[positive_roi_ix].detach()
        pred = pred[positive_roi_ix, positive_roi_class_ids]
        if "regression_bin" in tasks:
            loss = F.cross_entropy(pred, target.long())
        else:
            loss = F.smooth_l1_loss(pred, target)
            #loss = F.mse_loss(pred, target)
    else:
        loss = torch.FloatTensor([0.]).cuda()

    return loss

############################################################
#  Detection Layer
############################################################

def compute_roi_scores(tasks, batch_rpn_proposals, mrcnn_cl_logits):
    """ Depending on the predicition tasks: if no class prediction beyong fg/bg (--> means no additional class
        head was applied) use RPN objectness scores as roi scores, otherwise class head scores.
    :param cf:
    :param batch_rpn_proposals:
    :param mrcnn_cl_logits:
    :return:
    """
    if not 'class' in tasks:
        scores = batch_rpn_proposals[:, :, -1].view(-1, 1)
        scores = torch.cat((1 - scores, scores), dim=1)
    else:
        scores = F.softmax(mrcnn_cl_logits, dim=1)

    return scores

############################################################
#  MaskRCNN Class
############################################################

class net(nn.Module):


    def __init__(self, cf, logger):

        super(net, self).__init__()
        self.cf = cf
        self.logger = logger
        self.build()

        loss_order = ['rpn_class', 'rpn_bbox', 'mrcnn_bbox', 'mrcnn_mask', 'mrcnn_class', 'mrcnn_rg']
        if hasattr(cf, "mrcnn_loss_weights"):
            # bring into right order
            self.loss_weights = np.array([cf.mrcnn_loss_weights[k] for k in loss_order])
        else:
            self.loss_weights = np.array([1.]*len(loss_order))

        if self.cf.weight_init=="custom":
            logger.info("Tried to use custom weight init which is not defined. Using pytorch default.")
        elif self.cf.weight_init:
            mutils.initialize_weights(self)
        else:
            logger.info("using default pytorch weight init")

    def build(self):
        """Build Mask R-CNN architecture."""

        # Image size must be dividable by 2 multiple times.
        h, w = self.cf.patch_size[:2]
        if h / 2**5 != int(h / 2**5) or w / 2**5 != int(w / 2**5):
            raise Exception("Image size must be divisible by 2 at least 5 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 288, 320, 384, 448, 512, ... etc.,i.e.,"
                            "any number x*32 will do!")

        # instantiate abstract multi-dimensional conv generator and load backbone module.
        backbone = utils.import_module('bbone', self.cf.backbone_path)
        self.logger.info("loaded backbone from {}".format(self.cf.backbone_path))
        conv = backbone.ConvGenerator(self.cf.dim)

        # build Anchors, FPN, RPN, Classifier / Bbox-Regressor -head, Mask-head
        self.np_anchors = mutils.generate_pyramid_anchors(self.logger, self.cf)
        self.anchors = torch.from_numpy(self.np_anchors).float().cuda()
        self.fpn = backbone.FPN(self.cf, conv, relu_enc=self.cf.relu, operate_stride1=False).cuda()
        self.rpn = RPN(self.cf, conv)
        self.classifier = Classifier(self.cf, conv)
        self.mask = Mask(self.cf, conv)

    def forward(self, img, is_training=True):
        """
        :param img: input images (b, c, y, x, (z)).
        :return: rpn_pred_logits: (b, n_anchors, 2)
        :return: rpn_pred_deltas: (b, n_anchors, (y, x, (z), log(h), log(w), (log(d))))
        :return: batch_proposal_boxes: (b, n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ix)) only for monitoring/plotting.
        :return: detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
        :return: detection_masks: (n_final_detections, n_classes, y, x, (z)) raw molded masks as returned by mask-head.
        """
        # extract features.
        fpn_outs = self.fpn(img)
        rpn_feature_maps = [fpn_outs[i] for i in self.cf.pyramid_levels]
        self.mrcnn_feature_maps = rpn_feature_maps

        # loop through pyramid layers and apply RPN.
        layer_outputs = [ self.rpn(p_feats) for p_feats in rpn_feature_maps ]

        # concatenate layer outputs.
        # convert from list of lists of level outputs to list of lists of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_pred_logits, rpn_pred_probs, rpn_pred_deltas = outputs
        #
        # # generate proposals: apply predicted deltas to anchors and filter by foreground scores from RPN classifier.
        proposal_count = self.cf.post_nms_rois_training if is_training else self.cf.post_nms_rois_inference
        batch_normed_props, batch_unnormed_props = mutils.refine_proposals(rpn_pred_probs, rpn_pred_deltas,
                                                                            proposal_count, self.anchors, self.cf)

        # merge batch dimension of proposals while storing allocation info in coordinate dimension.
        batch_ixs = torch.arange(
            batch_normed_props.shape[0]).cuda().unsqueeze(1).repeat(1,batch_normed_props.shape[1]).view(-1).float()
        rpn_rois = batch_normed_props[:, :, :-1].view(-1, batch_normed_props[:, :, :-1].shape[2])
        self.rpn_rois_batch_info = torch.cat((rpn_rois, batch_ixs.unsqueeze(1)), dim=1)

        # this is the first of two forward passes in the second stage, where no activations are stored for backprop.
        # here, all proposals are forwarded (with virtual_batch_size = batch_size * post_nms_rois.)
        # for inference/monitoring as well as sampling of rois for the loss functions.
        # processed in chunks of roi_chunk_size to re-adjust to gpu-memory.
        chunked_rpn_rois = self.rpn_rois_batch_info.split(self.cf.roi_chunk_size)
        bboxes_list, class_logits_list, regressions_list = [], [], []
        with torch.no_grad():
            for chunk in chunked_rpn_rois:
                chunk_bboxes, chunk_class_logits, chunk_regressions = self.classifier(self.mrcnn_feature_maps, chunk)
                bboxes_list.append(chunk_bboxes)
                class_logits_list.append(chunk_class_logits)
                regressions_list.append(chunk_regressions)
        mrcnn_bbox = torch.cat(bboxes_list, 0)
        mrcnn_class_logits = torch.cat(class_logits_list, 0)
        mrcnn_regressions = torch.cat(regressions_list, 0)
        self.mrcnn_roi_scores = compute_roi_scores(self.cf.prediction_tasks, batch_normed_props, mrcnn_class_logits)

        # refine classified proposals, filter and return final detections.
        # returns (cf.max_inst_per_batch_element, n_coords+1+...)
        detections = mutils.refine_detections(self.cf, batch_ixs, rpn_rois, mrcnn_bbox, self.mrcnn_roi_scores,
                                       mrcnn_regressions)

        # forward remaining detections through mask-head to generate corresponding masks.
        scale = [img.shape[2]] * 4 + [img.shape[-1]] * 2
        scale = torch.from_numpy(np.array(scale[:self.cf.dim * 2] + [1])[None]).float().cuda()

        # first self.cf.dim * 2 entries on axis 1 are always the box coords, +1 is batch_ix
        detection_boxes = detections[:, :self.cf.dim * 2 + 1] / scale
        with torch.no_grad():
            detection_masks = self.mask(self.mrcnn_feature_maps, detection_boxes)

        return [rpn_pred_logits, rpn_pred_deltas, batch_unnormed_props, detections, detection_masks]


    def loss_samples_forward(self, batch_gt_boxes, batch_gt_masks, batch_gt_class_ids, batch_gt_regressions=None):
        """
        this is the second forward pass through the second stage (features from stage one are re-used).
        samples few rois in loss_example_mining and forwards only those for loss computation.
        :param batch_gt_class_ids: list over batch elements. Each element is a list over the corresponding roi target labels.
        :param batch_gt_boxes: list over batch elements. Each element is a list over the corresponding roi target coordinates.
        :param batch_gt_masks: (b, n(b), c, y, x (,z)) list over batch elements. Each element holds n_gt_rois(b)
                (i.e., dependent on the batch element) binary masks of shape (c, y, x, (z)).
        :return: sample_logits: (n_sampled_rois, n_classes) predicted class scores.
        :return: sample_deltas: (n_sampled_rois, n_classes, 2 * dim) predicted corrections to be applied to proposals for refinement.
        :return: sample_mask: (n_sampled_rois, n_classes, y, x, (z)) predicted masks per class and proposal.
        :return: sample_target_class_ids: (n_sampled_rois) target class labels of sampled proposals.
        :return: sample_target_deltas: (n_sampled_rois, 2 * dim) target deltas of sampled proposals for box refinement.
        :return: sample_target_masks: (n_sampled_rois, y, x, (z)) target masks of sampled proposals.
        :return: sample_proposals: (n_sampled_rois, 2 * dim) RPN output for sampled proposals. only for monitoring/plotting.
        """
        # sample rois for loss and get corresponding targets for all Mask R-CNN head network losses.
        sample_ics, sample_target_deltas, sample_target_mask, sample_target_class_ids, sample_target_regressions = \
            mutils.loss_example_mining(self.cf, self.rpn_rois_batch_info, batch_gt_boxes, batch_gt_masks,
                                       self.mrcnn_roi_scores, batch_gt_class_ids, batch_gt_regressions)

        # re-use feature maps and RPN output from first forward pass.
        sample_proposals = self.rpn_rois_batch_info[sample_ics]
        if not 0 in sample_proposals.size():
            sample_deltas, sample_logits, sample_regressions = self.classifier(self.mrcnn_feature_maps, sample_proposals)
            sample_mask = self.mask(self.mrcnn_feature_maps, sample_proposals)
        else:
            sample_logits = torch.FloatTensor().cuda()
            sample_deltas = torch.FloatTensor().cuda()
            sample_regressions = torch.FloatTensor().cuda()
            sample_mask = torch.FloatTensor().cuda()

        return [sample_deltas, sample_mask, sample_logits, sample_regressions, sample_proposals,
                sample_target_deltas, sample_target_mask, sample_target_class_ids, sample_target_regressions]

    def get_results(self, img_shape, detections, detection_masks, box_results_list=None, return_masks=True):
        """
        Restores batch dimension of merged detections, unmolds detections, creates and fills results dict.
        :param img_shape:
        :param detections: shape (n_final_detections, len(info)), where
            info=( y1, x1, y2, x2, (z1,z2), batch_ix, pred_class_id, pred_score )
        :param detection_masks: (n_final_detections, n_classes, y, x, (z)) raw molded masks as returned by mask-head.
        :param box_results_list: None or list of output boxes for monitoring/plotting.
        each element is a list of boxes per batch element.
        :param return_masks: boolean. If True, full resolution masks are returned for all proposals (speed trade-off).
        :return: results_dict: dictionary with keys:
                 'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                          [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
                 'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, 1] only fg. vs. bg for now.
                 class-specific return of masks will come with implementation of instance segmentation evaluation.
        """

        detections = detections.cpu().data.numpy()
        if self.cf.dim == 2:
            detection_masks = detection_masks.permute(0, 2, 3, 1).cpu().data.numpy()
        else:
            detection_masks = detection_masks.permute(0, 2, 3, 4, 1).cpu().data.numpy()
        # det masks shape now (n_dets, y,x(,z), n_classes)
        # restore batch dimension of merged detections using the batch_ix info.
        batch_ixs = detections[:, self.cf.dim*2]
        detections = [detections[batch_ixs == ix] for ix in range(img_shape[0])]
        mrcnn_mask = [detection_masks[batch_ixs == ix] for ix in range(img_shape[0])]
        # mrcnn_mask: shape (b_size, variable, variable, n_classes), variable bc depends on single instance mask size

        if box_results_list == None: # for test_forward, where no previous list exists.
            box_results_list =  [[] for _ in range(img_shape[0])]
        # seg_logits == seg_probs in mrcnn since mask head finishes with sigmoid (--> image space = [0,1])
        seg_probs = []
        # loop over batch and unmold detections.
        for ix in range(img_shape[0]):

            # final masks are one-hot encoded (b, n_classes, y, x, (z))
            final_masks = np.zeros((self.cf.num_classes + 1, *img_shape[2:]))
            #+1 for bg, 0.5 bc mask head classifies only bg/fg with logits between 0,1--> bg is <0.5
            if self.cf.num_classes + 1 != self.cf.num_seg_classes:
                self.logger.warning("n of roi-classifier head classes {} doesnt match cf.num_seg_classes {}".format(
                    self.cf.num_classes + 1, self.cf.num_seg_classes))

            if not 0 in detections[ix].shape:
                boxes = detections[ix][:, :self.cf.dim*2].astype(np.int32)
                class_ids = detections[ix][:, self.cf.dim*2 + 1].astype(np.int32)
                scores = detections[ix][:, self.cf.dim*2 + 2]
                masks = mrcnn_mask[ix][np.arange(boxes.shape[0]), ..., class_ids]
                regressions = detections[ix][:,self.cf.dim*2+3:]

                # Filter out detections with zero area. Often only happens in early
                # stages of training when the network weights are still a bit random.
                if self.cf.dim == 2:
                    exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
                else:
                    exclude_ix = np.where(
                        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4]) <= 0)[0]

                if exclude_ix.shape[0] > 0:
                    boxes = np.delete(boxes, exclude_ix, axis=0)
                    masks = np.delete(masks, exclude_ix, axis=0)
                    class_ids = np.delete(class_ids, exclude_ix, axis=0)
                    scores = np.delete(scores, exclude_ix, axis=0)
                    regressions = np.delete(regressions, exclude_ix, axis=0)

                # Resize masks to original image size and set boundary threshold.
                if return_masks:
                    for i in range(masks.shape[0]): #masks per this batch instance/element/image
                        # Convert neural network mask to full size mask
                        if self.cf.dim == 2:
                            full_mask = mutils.unmold_mask_2D(masks[i], boxes[i], img_shape[2:])
                        else:
                            full_mask = mutils.unmold_mask_3D(masks[i], boxes[i], img_shape[2:])
                        # take the maximum seg_logits per class of instances in that class, i.e., a pixel in a class
                        # has the max seg_logit value over all instances of that class in one sample
                        final_masks[class_ids[i]] = np.max((final_masks[class_ids[i]], full_mask), axis=0)
                    final_masks[0] = np.full(final_masks[0].shape, 0.49999999) #effectively min_det_thres at 0.5 per pixel

                # add final predictions to results.
                if not 0 in boxes.shape:
                    for ix2, coords in enumerate(boxes):
                        box = {'box_coords': coords, 'box_type': 'det', 'box_score': scores[ix2],
                               'box_pred_class_id': class_ids[ix2]}
                        #if (hasattr(self.cf, "convert_cl_to_rg") and self.cf.convert_cl_to_rg):
                        if "regression_bin" in self.cf.prediction_tasks:
                            # in this case, regression preds are actually the rg_bin_ids --> map to rg value the bin represents
                            box['rg_bin'] = regressions[ix2].argmax()
                            box['regression'] = self.cf.bin_id2rg_val[box['rg_bin']]
                        else:
                            box['regression'] = regressions[ix2]
                            if hasattr(self.cf, "rg_val_to_bin_id") and \
                                    any(['regression' in task for task in self.cf.prediction_tasks]):
                                box.update({'rg_bin': self.cf.rg_val_to_bin_id(regressions[ix2])})

                        box_results_list[ix].append(box)

            # if no detections were made--> keep full bg mask (zeros).
            seg_probs.append(final_masks)

        # create and fill results dictionary.
        results_dict = {}
        results_dict['boxes'] = box_results_list
        results_dict['seg_preds'] = np.array(seg_probs)

        return results_dict


    def train_forward(self, batch, is_validation=False):
        """
        train method (also used for validation monitoring). wrapper around forward pass of network. prepares input data
        for processing, computes losses, and stores outputs in a dictionary.
        :param batch: dictionary containing 'data', 'seg', etc.
            batch['roi_masks']: (b, n(b), c, h(n), w(n) (z(n))) list like roi_labels but with arrays (masks) inplace of
        integers. c==channels of the raw segmentation.
        :return: results_dict: dictionary with keys:
                'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                        [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
                'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, n_classes].
                'torch_loss': 1D torch tensor for backprop.
                'class_loss': classification loss for monitoring.
        """
        img = batch['data']
        gt_boxes = batch['bb_target']
        #axes = (0, 2, 3, 1) if self.cf.dim == 2 else (0, 2, 3, 4, 1)
        #gt_masks = [np.transpose(batch['roi_masks'][ii], axes=axes) for ii in range(len(batch['roi_masks']))]
        gt_masks = batch['roi_masks']
        gt_class_ids = batch['class_targets']
        if 'regression' in self.cf.prediction_tasks:
            gt_regressions = batch["regression_targets"]
        elif 'regression_bin' in self.cf.prediction_tasks:
            gt_regressions = batch["rg_bin_targets"]
        else:
            gt_regressions = None

        img = torch.from_numpy(img).cuda().float()
        batch_rpn_class_loss = torch.FloatTensor([0]).cuda()
        batch_rpn_bbox_loss = torch.FloatTensor([0]).cuda()

        # list of output boxes for monitoring/plotting. each element is a list of boxes per batch element.
        box_results_list = [[] for _ in range(img.shape[0])]

        #forward passes. 1. general forward pass, where no activations are saved in second stage (for performance
        # monitoring and loss sampling). 2. second stage forward pass of sampled rois with stored activations for backprop.
        rpn_class_logits, rpn_pred_deltas, proposal_boxes, detections, detection_masks = self.forward(img)

        mrcnn_pred_deltas, mrcnn_pred_mask, mrcnn_class_logits, mrcnn_regressions, sample_proposals, \
        mrcnn_target_deltas, target_mask, target_class_ids, target_regressions = \
            self.loss_samples_forward(gt_boxes, gt_masks, gt_class_ids, gt_regressions)
        # loop over batch
        for b in range(img.shape[0]):
            if len(gt_boxes[b]) > 0:
                # add gt boxes to output list
                for tix in range(len(gt_boxes[b])):
                    gt_box = {'box_type': 'gt', 'box_coords': batch['bb_target'][b][tix]}
                    for name in self.cf.roi_items:
                        gt_box.update({name: batch[name][b][tix]})
                    box_results_list[b].append(gt_box)

                # match gt boxes with anchors to generate targets for RPN losses.
                rpn_match, rpn_target_deltas = mutils.gt_anchor_matching(self.cf, self.np_anchors, gt_boxes[b])

                # add positive anchors used for loss to output list for monitoring.
                pos_anchors = mutils.clip_boxes_numpy(self.np_anchors[np.argwhere(rpn_match == 1)][:, 0], img.shape[2:])
                for p in pos_anchors:
                    box_results_list[b].append({'box_coords': p, 'box_type': 'pos_anchor'})

            else:
                rpn_match = np.array([-1]*self.np_anchors.shape[0])
                rpn_target_deltas = np.array([0])

            rpn_match_gpu = torch.from_numpy(rpn_match).cuda()
            rpn_target_deltas = torch.from_numpy(rpn_target_deltas).float().cuda()

            # compute RPN losses.
            rpn_class_loss, neg_anchor_ix = compute_rpn_class_loss(rpn_class_logits[b], rpn_match_gpu, self.cf.shem_poolsize)
            rpn_bbox_loss = compute_rpn_bbox_loss(rpn_pred_deltas[b], rpn_target_deltas, rpn_match_gpu)
            batch_rpn_class_loss += rpn_class_loss /img.shape[0]
            batch_rpn_bbox_loss += rpn_bbox_loss /img.shape[0]

            # add negative anchors used for loss to output list for monitoring.
            # neg_anchor_ix = neg_ix come from shem and mark positions in roi_probs_neg = rpn_class_logits[neg_indices]
            # with neg_indices = rpn_match == -1
            neg_anchors = mutils.clip_boxes_numpy(self.np_anchors[rpn_match == -1][neg_anchor_ix], img.shape[2:])
            for n in neg_anchors:
                box_results_list[b].append({'box_coords': n, 'box_type': 'neg_anchor'})

            # add highest scoring proposals to output list for monitoring.
            rpn_proposals = proposal_boxes[b][proposal_boxes[b, :, -1].argsort()][::-1]
            for r in rpn_proposals[:self.cf.n_plot_rpn_props, :-1]:
                box_results_list[b].append({'box_coords': r, 'box_type': 'prop'})

        # add positive and negative roi samples used for mrcnn losses to output list for monitoring.
        if not 0 in sample_proposals.shape:
            rois = mutils.clip_to_window(self.cf.window, sample_proposals).cpu().data.numpy()
            for ix, r in enumerate(rois):
                box_results_list[int(r[-1])].append({'box_coords': r[:-1] * self.cf.scale,
                                            'box_type': 'pos_class' if target_class_ids[ix] > 0 else 'neg_class'})

        # compute mrcnn losses.
        mrcnn_class_loss = compute_mrcnn_class_loss(self.cf.prediction_tasks, mrcnn_class_logits, target_class_ids)
        mrcnn_bbox_loss = compute_mrcnn_bbox_loss(mrcnn_pred_deltas, mrcnn_target_deltas, target_class_ids)
        mrcnn_regressions_loss = compute_mrcnn_regression_loss(self.cf.prediction_tasks, mrcnn_regressions, target_regressions, target_class_ids)
        # mrcnn can be run without pixelwise annotations available (Faster R-CNN mode).
        # In this case, the mask_loss is taken out of training.
        if self.cf.frcnn_mode:
            mrcnn_mask_loss = torch.FloatTensor([0]).cuda()
        else:
            mrcnn_mask_loss = compute_mrcnn_mask_loss(mrcnn_pred_mask, target_mask, target_class_ids)

        loss = batch_rpn_class_loss + batch_rpn_bbox_loss +\
               mrcnn_bbox_loss + mrcnn_mask_loss +  mrcnn_class_loss + mrcnn_regressions_loss

        # run unmolding of predictions for monitoring and merge all results to one dictionary.
        return_masks = self.cf.return_masks_in_val if is_validation else self.cf.return_masks_in_train
        results_dict = self.get_results(img.shape, detections, detection_masks, box_results_list,
                                        return_masks=return_masks)
        results_dict['seg_preds'] = results_dict['seg_preds'].argmax(axis=1).astype('uint8')[:,np.newaxis]

        if 'dice' in self.cf.metrics:
            results_dict['batch_dices'] = mutils.dice_per_batch_and_class(
                results_dict['seg_preds'], batch["seg"], self.cf.num_seg_classes, convert_to_ohe=True)

        results_dict['torch_loss'] = loss
        results_dict['class_loss'] = mrcnn_class_loss.item()
        results_dict['bbox_loss'] = mrcnn_bbox_loss.item()
        results_dict['mask_loss'] = mrcnn_mask_loss.item()
        results_dict['rg_loss'] = mrcnn_regressions_loss.item()
        results_dict['rpn_class_loss'] = rpn_class_loss.item()
        results_dict['rpn_bbox_loss'] = rpn_bbox_loss.item()
        return results_dict


    def test_forward(self, batch, return_masks=True):
        """
        test method. wrapper around forward pass of network without usage of any ground truth information.
        prepares input data for processing and stores outputs in a dictionary.
        :param batch: dictionary containing 'data'
        :param return_masks: boolean. If True, full resolution masks are returned for all proposals (speed trade-off).
        :return: results_dict: dictionary with keys:
               'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                       [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
               'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, n_classes]
        """
        img = batch['data']
        img = torch.from_numpy(img).float().cuda()
        _, _, _, detections, detection_masks = self.forward(img)
        results_dict = self.get_results(img.shape, detections, detection_masks, return_masks=return_masks)

        return results_dict