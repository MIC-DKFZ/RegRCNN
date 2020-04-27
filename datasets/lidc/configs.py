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

import sys
import os
from collections import namedtuple
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../..")
from default_configs import DefaultConfigs

# legends, nested classes are not handled well in multiprocessing! hence, Label class def in outer scope
Label = namedtuple("Label", ['id', 'name', 'color', 'm_scores']) # m_scores = malignancy scores
binLabel = namedtuple("binLabel", ['id', 'name', 'color', 'm_scores', 'bin_vals'])


class Configs(DefaultConfigs):

    def __init__(self, server_env=None):
        super(Configs, self).__init__(server_env)

        #########################
        #    Preprocessing      #
        #########################

        self.root_dir = '/home/gregor/networkdrives/E130-Personal/Goetz/Datenkollektive/Lungendaten/Nodules_LIDC_IDRI'
        self.raw_data_dir = '{}/new_nrrd'.format(self.root_dir)
        self.pp_dir = '/media/gregor/HDD2TB/data/lidc/pp_20200309_dev'
        # 'merged' for one gt per image, 'single_annotator' for four gts per image.
        self.gts_to_produce = ["single_annotator", "merged"]

        self.target_spacing = (0.7, 0.7, 1.25)

        #########################
        #         I/O          #
        #########################

        # path to preprocessed data.
        self.pp_name = 'pp_20190805'

        self.input_df_name = 'info_df.pickle'
        self.data_sourcedir = '/media/gregor/HDD2TB/data/lidc/{}/'.format(self.pp_name)
        #self.data_sourcedir = '/home/gregor/networkdrives/E132-Cluster-Projects/lidc/data/{}/'.format(self.pp_name)

        # settings for deployment on cluster.
        if server_env:
            # path to preprocessed data.
            self.data_sourcedir = '/datasets/datasets_ramien/lidc/data/{}_npz/'.format(self.pp_name)

        # one out of ['mrcnn', 'retina_net', 'retina_unet', 'detection_fpn'].
        self.model = 'detection_fpn'
        self.model_path = 'models/{}.py'.format(self.model if not 'retina' in self.model else 'retina_net')
        self.model_path = os.path.join(self.source_dir, self.model_path)


        #########################
        #      Architecture     #
        #########################

        # dimension the model operates in. one out of [2, 3].
        self.dim = 3

        # 'class': standard object classification per roi, pairwise combinable with each of below tasks.
        # if 'class' is omitted from tasks, object classes will be fg/bg (1/0) from RPN.
        # 'regression': regress some vector per each roi
        # 'regression_ken_gal': use kendall-gal uncertainty sigma
        # 'regression_bin': classify each roi into a bin related to a regression scale
        self.prediction_tasks = ['class']

        self.start_filts = 48 if self.dim == 2 else 18
        self.end_filts = self.start_filts * 4 if self.dim == 2 else self.start_filts * 2
        self.res_architecture = 'resnet50' # 'resnet101' , 'resnet50'
        self.norm = "instance_norm" # one of None, 'instance_norm', 'batch_norm'

        # one of 'xavier_uniform', 'xavier_normal', or 'kaiming_normal', None (=default = 'kaiming_uniform')
        self.weight_init = None

        self.regression_n_features = 1

        #########################
        #      Data Loader      #
        #########################

        # distorted gt experiments: train on single-annotator gts in a random fashion to investigate network's
        # handling of noisy gts.
        # choose 'merged' for single, merged gt per image, or 'single_annotator' for four gts per image.
        # validation is always performed on same gt kind as training, testing always on merged gt.
        self.training_gts = "merged"

        # select modalities from preprocessed data
        self.channels = [0]
        self.n_channels = len(self.channels)

        # patch_size to be used for training. pre_crop_size is the patch_size before data augmentation.
        self.pre_crop_size_2D = [320, 320]
        self.patch_size_2D = [320, 320]
        self.pre_crop_size_3D = [160, 160, 96]
        self.patch_size_3D = [160, 160, 96]

        self.patch_size = self.patch_size_2D if self.dim == 2 else self.patch_size_3D
        self.pre_crop_size = self.pre_crop_size_2D if self.dim == 2 else self.pre_crop_size_3D

        # ratio of free sampled batch elements before class balancing is triggered
        self.batch_random_ratio = 0.1
        self.balance_target =  "class_targets" if 'class' in self.prediction_tasks else 'rg_bin_targets'

        # set 2D network to match 3D gt boxes.
        self.merge_2D_to_3D_preds = self.dim==2

        self.observables_rois = []

        #self.rg_map = {1:1, 2:2, 3:3, 4:4, 5:5}

        #########################
        #   Colors and Legends  #
        #########################
        self.plot_frequency = 5

        binary_cl_labels = [Label(1, 'benign',  (*self.dark_green, 1.),  (1, 2)),
                            Label(2, 'malignant', (*self.red, 1.),  (3, 4, 5))]
        quintuple_cl_labels = [Label(1, 'MS1',  (*self.dark_green, 1.),      (1,)),
                               Label(2, 'MS2',  (*self.dark_yellow, 1.),     (2,)),
                               Label(3, 'MS3',  (*self.orange, 1.),     (3,)),
                               Label(4, 'MS4',  (*self.bright_red, 1.), (4,)),
                               Label(5, 'MS5',  (*self.red, 1.),        (5,))]
        # choose here if to do 2-way or 5-way regression-bin classification
        task_spec_cl_labels = quintuple_cl_labels

        self.class_labels = [
            #       #id #name     #color              #malignancy score
            Label(  0,  'bg',     (*self.gray, 0.),  (0,))]
        if "class" in self.prediction_tasks:
            self.class_labels += task_spec_cl_labels

        else:
            self.class_labels += [Label(1, 'lesion', (*self.orange, 1.), (1,2,3,4,5))]

        if any(['regression' in task for task in self.prediction_tasks]):
            self.bin_labels = [binLabel(0, 'MS0', (*self.gray, 1.), (0,), (0,))]
            self.bin_labels += [binLabel(cll.id, cll.name, cll.color, cll.m_scores,
                                         tuple([ms for ms in cll.m_scores])) for cll in task_spec_cl_labels]
            self.bin_id2label = {label.id: label for label in self.bin_labels}
            self.ms2bin_label = {ms: label for label in self.bin_labels for ms in label.m_scores}
            bins = [(min(label.bin_vals), max(label.bin_vals)) for label in self.bin_labels]
            self.bin_id2rg_val = {ix: [np.mean(bin)] for ix, bin in enumerate(bins)}
            self.bin_edges = [(bins[i][1] + bins[i + 1][0]) / 2 for i in range(len(bins) - 1)]

        if self.class_specific_seg:
            self.seg_labels = self.class_labels
        else:
            self.seg_labels = [  # id      #name           #color
                Label(0, 'bg', (*self.gray, 0.)),
                Label(1, 'fg', (*self.orange, 1.))
            ]

        self.class_id2label = {label.id: label for label in self.class_labels}
        self.class_dict = {label.id: label.name for label in self.class_labels if label.id != 0}
        # class_dict is used in evaluator / ap, auc, etc. statistics, and class 0 (bg) only needs to be
        # evaluated in debugging
        self.class_cmap = {label.id: label.color for label in self.class_labels}

        self.seg_id2label = {label.id: label for label in self.seg_labels}
        self.cmap = {label.id: label.color for label in self.seg_labels}

        self.plot_prediction_histograms = True
        self.plot_stat_curves = False
        self.has_colorchannels = False
        self.plot_class_ids = True

        self.num_classes = len(self.class_dict)  # for instance classification (excl background)
        self.num_seg_classes = len(self.seg_labels)  # incl background


        #########################
        #   Data Augmentation   #
        #########################

        self.da_kwargs={
            'mirror': True,
            'mirror_axes': tuple(np.arange(0, self.dim, 1)),
            'do_elastic_deform': True,
            'alpha':(0., 1500.),
            'sigma':(30., 50.),
            'do_rotation':True,
            'angle_x': (0., 2 * np.pi),
            'angle_y': (0., 0),
            'angle_z': (0., 0),
            'do_scale': True,
            'scale':(0.8, 1.1),
            'random_crop':False,
            'rand_crop_dist':  (self.patch_size[0] / 2. - 3, self.patch_size[1] / 2. - 3),
            'border_mode_data': 'constant',
            'border_cval_data': 0,
            'order_data': 1}

        if self.dim == 3:
            self.da_kwargs['do_elastic_deform'] = False
            self.da_kwargs['angle_x'] = (0, 0.0)
            self.da_kwargs['angle_y'] = (0, 0.0) #must be 0!!
            self.da_kwargs['angle_z'] = (0., 2 * np.pi)

        #################################
        #  Schedule / Selection / Optim #
        #################################

        self.num_epochs = 130 if self.dim == 2 else 150
        self.num_train_batches = 200 if self.dim == 2 else 200
        self.batch_size = 20 if self.dim == 2 else 8

        # decide whether to validate on entire patient volumes (like testing) or sampled patches (like training)
        # the former is morge accurate, while the latter is faster (depending on volume size)
        self.val_mode = 'val_sampling' # only 'val_sampling', 'val_patient' not implemented
        if self.val_mode == 'val_patient':
            raise NotImplementedError
        if self.val_mode == 'val_sampling':
            self.num_val_batches = 70

        self.save_n_models = 4
        # set a minimum epoch number for saving in case of instabilities in the first phase of training.
        self.min_save_thresh = 0 if self.dim == 2 else 0
        # criteria to average over for saving epochs, 'criterion':weight.
        if "class" in self.prediction_tasks:
            # 'criterion': weight
            if len(self.class_labels)==3:
                self.model_selection_criteria = {"benign_ap": 0.5, "malignant_ap": 0.5}
            elif len(self.class_labels)==6:
                self.model_selection_criteria = {str(label.name)+"_ap": 1./5 for label in self.class_labels if label.id!=0}
        elif any("regression" in task for task in self.prediction_tasks):
            self.model_selection_criteria = {"lesion_ap": 0.2, "lesion_avp": 0.8}

        self.weight_decay = 1e-5
        self.exclude_from_wd = []
        self.clip_norm = 200 if 'regression_ken_gal' in self.prediction_tasks else None  # number or None

        # int in [0, dataset_size]. select n patients from dataset for prototyping. If None, all data is used.
        self.select_prototype_subset = None #self.batch_size

        #########################
        #        Testing        #
        #########################

        # set the top-n-epochs to be saved for temporal averaging in testing.
        self.test_n_epochs = self.save_n_models

        self.test_aug_axes = (0,1,(0,1))  # None or list: choices are 0,1,(0,1) (0==spatial y, 1== spatial x).
        self.hold_out_test_set = False
        self.max_test_patients = "all"  # "all" or number

        self.report_score_level = ['rois', 'patient']  # choose list from 'patient', 'rois'
        self.patient_class_of_interest = 2 if 'class' in self.prediction_tasks else 1

        self.metrics = ['ap', 'auc']
        if any(['regression' in task for task in self.prediction_tasks]):
            self.metrics += ['avp', 'rg_MAE_weighted', 'rg_MAE_weighted_tp',
                             'rg_bin_accuracy_weighted', 'rg_bin_accuracy_weighted_tp']
        if 'aleatoric' in self.model:
            self.metrics += ['rg_uncertainty', 'rg_uncertainty_tp', 'rg_uncertainty_tp_weighted']
        self.evaluate_fold_means = True

        self.ap_match_ious = [0.1]  # list of ious to be evaluated for ap-scoring.
        self.min_det_thresh = 0.1  # minimum confidence value to select predictions for evaluation.

        # aggregation method for test and val_patient predictions.
        # wbc = weighted box clustering as in https://arxiv.org/pdf/1811.08661.pdf,
        # nms = standard non-maximum suppression, or None = no clustering
        self.clustering = 'wbc'
        # iou thresh (exclusive!) for regarding two preds as concerning the same ROI
        self.clustering_iou = 0.1  # has to be larger than desired possible overlap iou of model predictions

        self.plot_prediction_histograms = True
        self.plot_stat_curves = False
        self.n_test_plots = 1

        #########################
        #   Assertions          #
        #########################
        if not 'class' in self.prediction_tasks:
            assert self.num_classes == 1

        #########################
        #   Add model specifics #
        #########################

        {'detection_fpn': self.add_det_fpn_configs,
         'mrcnn': self.add_mrcnn_configs,
         'retina_net': self.add_mrcnn_configs,
         'retina_unet': self.add_mrcnn_configs,
        }[self.model]()

    def rg_val_to_bin_id(self, rg_val):
        return float(np.digitize(np.mean(rg_val), self.bin_edges))

    def add_det_fpn_configs(self):

        self.learning_rate = [3e-4] * self.num_epochs
        self.dynamic_lr_scheduling = False

        # RoI score assigned to aggregation from pixel prediction (connected component). One of ['max', 'median'].
        self.score_det = 'max'

        # max number of roi candidates to identify per batch element and class.
        self.n_roi_candidates = 10 if self.dim == 2 else 30

        # loss mode: either weighted cross entropy ('wce'), batch-wise dice loss ('dice), or the sum of both ('dice_wce')
        self.seg_loss_mode = 'wce'

        # if <1, false positive predictions in foreground are penalized less.
        self.fp_dice_weight = 1 if self.dim == 2 else 1
        if len(self.class_labels)==3:
            self.wce_weights = [1., 1., 1.] if self.seg_loss_mode=="dice_wce" else [0.1, 1., 1.]
        elif len(self.class_labels)==6:
            self.wce_weights = [1., 1., 1., 1., 1., 1.] if self.seg_loss_mode == "dice_wce" else [0.1, 1., 1., 1., 1., 1.]
        else:
            raise Exception("mismatch loss weights & nr of classes")
        self.detection_min_confidence = self.min_det_thresh

        self.head_classes = self.num_seg_classes

    def add_mrcnn_configs(self):

        # learning rate is a list with one entry per epoch.
        self.learning_rate = [3e-4] * self.num_epochs
        self.dynamic_lr_scheduling = False
        # disable the re-sampling of mask proposals to original size for speed-up.
        # since evaluation is detection-driven (box-matching) and not instance segmentation-driven (iou-matching),
        # mask-outputs are optional.
        self.return_masks_in_train = False
        self.return_masks_in_val = True
        self.return_masks_in_test = False

        # set number of proposal boxes to plot after each epoch.
        self.n_plot_rpn_props = 5 if self.dim == 2 else 30

        # number of classes for network heads: n_foreground_classes + 1 (background)
        self.head_classes = self.num_classes + 1

        self.frcnn_mode = False

        # feature map strides per pyramid level are inferred from architecture.
        self.backbone_strides = {'xy': [4, 8, 16, 32], 'z': [1, 2, 4, 8]}

        # anchor scales are chosen according to expected object sizes in data set. Default uses only one anchor scale
        # per pyramid level. (outer list are pyramid levels (corresponding to BACKBONE_STRIDES), inner list are scales per level.)
        self.rpn_anchor_scales = {'xy': [[8], [16], [32], [64]], 'z': [[2], [4], [8], [16]]}

        # choose which pyramid levels to extract features from: P2: 0, P3: 1, P4: 2, P5: 3.
        self.pyramid_levels = [0, 1, 2, 3]

        # number of feature maps in rpn. typically lowered in 3D to save gpu-memory.
        self.n_rpn_features = 512 if self.dim == 2 else 128

        # anchor ratios and strides per position in feature maps.
        self.rpn_anchor_ratios = [0.5, 1, 2]
        self.rpn_anchor_stride = 1

        # Threshold for first stage (RPN) non-maximum suppression (NMS):  LOWER == HARDER SELECTION
        self.rpn_nms_threshold = 0.7 if self.dim == 2 else 0.7

        # loss sampling settings.
        self.rpn_train_anchors_per_image = 64  #per batch element
        self.train_rois_per_image = 6 #per batch element
        self.roi_positive_ratio = 0.5
        self.anchor_matching_iou = 0.7

        # factor of top-k candidates to draw from  per negative sample (stochastic-hard-example-mining).
        # poolsize to draw top-k candidates from will be shem_poolsize * n_negative_samples.
        self.shem_poolsize = 10

        self.pool_size = (7, 7) if self.dim == 2 else (7, 7, 3)
        self.mask_pool_size = (14, 14) if self.dim == 2 else (14, 14, 5)
        self.mask_shape = (28, 28) if self.dim == 2 else (28, 28, 10)

        self.rpn_bbox_std_dev = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
        self.bbox_std_dev = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
        self.window = np.array([0, 0, self.patch_size[0], self.patch_size[1], 0, self.patch_size_3D[2]])
        self.scale = np.array([self.patch_size[0], self.patch_size[1], self.patch_size[0], self.patch_size[1],
                               self.patch_size_3D[2], self.patch_size_3D[2]])
        if self.dim == 2:
            self.rpn_bbox_std_dev = self.rpn_bbox_std_dev[:4]
            self.bbox_std_dev = self.bbox_std_dev[:4]
            self.window = self.window[:4]
            self.scale = self.scale[:4]

        # pre-selection in proposal-layer (stage 1) for NMS-speedup. applied per batch element.
        self.pre_nms_limit = 3000 if self.dim == 2 else 6000

        # n_proposals to be selected after NMS per batch element. too high numbers blow up memory if "detect_while_training" is True,
        # since proposals of the entire batch are forwarded through second stage in as one "batch".
        self.roi_chunk_size = 2500 if self.dim == 2 else 600
        self.post_nms_rois_training = 500 if self.dim == 2 else 75
        self.post_nms_rois_inference = 500

        # Final selection of detections (refine_detections)
        self.model_max_instances_per_batch_element = 10 if self.dim == 2 else 30  # per batch element and class.
        self.detection_nms_threshold = 1e-5  # needs to be > 0, otherwise all predictions are one cluster.
        self.model_min_confidence = 0.1

        if self.dim == 2:
            self.backbone_shapes = np.array(
                [[int(np.ceil(self.patch_size[0] / stride)),
                  int(np.ceil(self.patch_size[1] / stride))]
                 for stride in self.backbone_strides['xy']])
        else:
            self.backbone_shapes = np.array(
                [[int(np.ceil(self.patch_size[0] / stride)),
                  int(np.ceil(self.patch_size[1] / stride)),
                  int(np.ceil(self.patch_size[2] / stride_z))]
                 for stride, stride_z in zip(self.backbone_strides['xy'], self.backbone_strides['z']
                                             )])

        if self.model == 'retina_net' or self.model == 'retina_unet':

            self.focal_loss = True

            # implement extra anchor-scales according to retina-net publication.
            self.rpn_anchor_scales['xy'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                            self.rpn_anchor_scales['xy']]
            self.rpn_anchor_scales['z'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                           self.rpn_anchor_scales['z']]
            self.n_anchors_per_pos = len(self.rpn_anchor_ratios) * 3

            self.n_rpn_features = 256 if self.dim == 2 else 128

            # pre-selection of detections for NMS-speedup. per entire batch.
            self.pre_nms_limit = (500 if self.dim == 2 else 6250) * self.batch_size

            # anchor matching iou is lower than in Mask R-CNN according to https://arxiv.org/abs/1708.02002
            self.anchor_matching_iou = 0.5

            if self.model == 'retina_unet':
                self.operate_stride1 = True

