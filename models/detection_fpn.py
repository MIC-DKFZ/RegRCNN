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
Unet-like Backbone architecture, with non-parametric heuristics for box detection on semantic segmentation outputs.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.ndimage.measurements import label as lb

import utils.exp_utils as utils
import utils.model_utils as mutils


class net(nn.Module):

    def __init__(self, cf, logger):

        super(net, self).__init__()
        self.cf = cf
        self.logger = logger
        backbone = utils.import_module('bbone', cf.backbone_path)
        self.logger.info("loaded backbone from {}".format(self.cf.backbone_path))
        conv_gen = backbone.ConvGenerator(cf.dim)

        # set operate_stride1=True to generate a unet-like FPN.)
        self.fpn = backbone.FPN(cf, conv=conv_gen, relu_enc=cf.relu, operate_stride1=True)
        self.conv_final = conv_gen(cf.end_filts, cf.num_seg_classes, ks=1, pad=0, norm=None, relu=None)

        #initialize parameters
        if self.cf.weight_init=="custom":
            logger.info("Tried to use custom weight init which is not defined. Using pytorch default.")
        elif self.cf.weight_init:
            mutils.initialize_weights(self)
        else:
            logger.info("using default pytorch weight init")


    def forward(self, x):
        """
        forward pass of network.
        :param x: input image. shape (b, c, y, x, (z))
        :return: seg_logits: shape (b, n_classes, y, x, (z))
        :return: out_box_coords: list over n_classes. elements are arrays(b, n_rois, (y1, x1, y2, x2, (z1), (z2)))
        :return: out_max_scores: list over n_classes. elements are arrays(b, n_rois)
        """

        out_features = self.fpn(x)[0] #take only pyramid output of stride 1

        seg_logits = self.conv_final(out_features)
        out_box_coords, out_max_scores = [], []
        smax = F.softmax(seg_logits.detach(), dim=1).cpu().data.numpy()

        for cl in range(1, len(self.cf.class_dict.keys()) + 1):
            hard_mask = np.copy(smax).argmax(1)
            hard_mask[hard_mask != cl] = 0
            hard_mask[hard_mask == cl] = 1
            # perform connected component analysis on argmaxed predictions,
            # draw boxes around components and return coordinates.
            box_coords, rois = mutils.get_coords(hard_mask, self.cf.n_roi_candidates, self.cf.dim)

            # for each object, choose the highest softmax score (in the respective class)
            # of all pixels in the component as object score.
            max_scores = [[] for _ in range(x.shape[0])]
            for bix, broi in enumerate(rois):
                for nix, nroi in enumerate(broi):
                    score_det = np.max if self.cf.score_det=="max" else np.median #score determination
                    max_scores[bix].append(score_det(smax[bix, cl][nroi > 0]))
            out_box_coords.append(box_coords)
            out_max_scores.append(max_scores)
        return seg_logits, out_box_coords, out_max_scores

    def train_forward(self, batch, **kwargs):
        """
        train method (also used for validation monitoring). wrapper around forward pass of network. prepares input data
        for processing, computes losses, and stores outputs in a dictionary.
        :param batch: dictionary containing 'data', 'seg', etc.
        :param kwargs:
        :return: results_dict: dictionary with keys:
                'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                        [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
                'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, n_classes]
                'torch_loss': 1D torch tensor for backprop.
                'class_loss': classification loss for monitoring. here: dummy array, since no classification conducted.
        """

        img = torch.from_numpy(batch['data']).cuda().float()
        seg = torch.from_numpy(batch['seg']).cuda().long()
        seg_ohe = torch.from_numpy(mutils.get_one_hot_encoding(batch['seg'], self.cf.num_seg_classes)).cuda()
        results_dict = {}
        seg_logits, box_coords, max_scores = self.forward(img)

        # no extra class loss applied in this model. pass dummy tensor for monitoring.
        results_dict['class_loss'] = np.nan

        results_dict['boxes'] = [[] for _ in range(img.shape[0])]
        for cix in range(len(self.cf.class_dict.keys())):
            for bix in range(img.shape[0]):
                for rix in range(len(max_scores[cix][bix])):
                    if max_scores[cix][bix][rix] > self.cf.detection_min_confidence:
                        results_dict['boxes'][bix].append({'box_coords': np.copy(box_coords[cix][bix][rix]),
                                    'box_score': max_scores[cix][bix][rix],
                                    'box_pred_class_id': cix + 1, # add 0 for background.
                                    'box_type': 'det'})

        for bix in range(img.shape[0]):
            for tix in range(len(batch['bb_target'][bix])):
                gt_box = {'box_coords': batch['bb_target'][bix][tix], 'box_type': 'gt'}
                for name in self.cf.roi_items:
                    gt_box.update({name: batch[name][bix][tix]})

                results_dict['boxes'][bix].append(gt_box)

        # compute segmentation loss as either weighted cross entropy, dice loss, or the sum of both.
        loss = torch.tensor([0.], dtype=torch.float, requires_grad=False).cuda()
        seg_pred = F.softmax(seg_logits, dim=1)
        if self.cf.seg_loss_mode == 'dice' or self.cf.seg_loss_mode == 'dice_wce':
            loss += 1 - mutils.batch_dice(seg_pred, seg_ohe.float(), false_positive_weight=float(self.cf.fp_dice_weight))

        if self.cf.seg_loss_mode == 'wce' or self.cf.seg_loss_mode == 'dice_wce':
            loss += F.cross_entropy(seg_logits, seg[:, 0], weight=torch.FloatTensor(self.cf.wce_weights).cuda())

        results_dict['torch_loss'] = loss
        seg_pred = seg_pred.argmax(dim=1).unsqueeze(dim=1).cpu().data.numpy()
        results_dict['seg_preds'] = seg_pred
        if 'dice' in self.cf.metrics:
            results_dict['batch_dices'] = mutils.dice_per_batch_and_class(seg_pred, batch["seg"],
                                                                           self.cf.num_seg_classes, convert_to_ohe=True)
        #self.logger.info("loss: {0:.2f}".format(loss.item()))
        return results_dict


    def test_forward(self, batch, **kwargs):
        """
        test method. wrapper around forward pass of network without usage of any ground truth information.
        prepares input data for processing and stores outputs in a dictionary.
        :param batch: dictionary containing 'data'
        :param kwargs:
        :return: results_dict: dictionary with keys:
               'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                       [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
               'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, n_classes]
        """
        img = torch.FloatTensor(batch['data']).cuda()
        seg_logits, box_coords, max_scores = self.forward(img)

        results_dict = {}
        results_dict['boxes'] = [[] for _ in range(img.shape[0])]
        for cix in range(len(box_coords)):
            for bix in range(img.shape[0]):
                for rix in range(len(max_scores[cix][bix])):
                    if max_scores[cix][bix][rix] > self.cf.detection_min_confidence:
                        results_dict['boxes'][bix].append({'box_coords': np.copy(box_coords[cix][bix][rix]),
                                    'box_score': max_scores[cix][bix][rix],
                                    'box_pred_class_id': cix + 1,
                                    'box_type': 'det'})
        results_dict['seg_preds'] = F.softmax(seg_logits, dim=1).cpu().data.numpy()

        return results_dict

