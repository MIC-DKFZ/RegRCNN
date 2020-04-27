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
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
from default_configs import DefaultConfigs
from collections import namedtuple

boxLabel = namedtuple('boxLabel', ["name", "color"])
Label = namedtuple("Label", ['id', 'name', 'shape', 'radius', 'color', 'regression', 'ambiguities', 'gt_distortion'])
binLabel = namedtuple("binLabel", ['id', 'name', 'color', 'bin_vals'])

class Configs(DefaultConfigs):

    def __init__(self, server_env=None):
        super(Configs, self).__init__(server_env)

        #########################
        #         Prepro        #
        #########################

        self.pp_rootdir = os.path.join('/home/gregor/datasets/toy', "cyl1ps_dev")
        self.pp_npz_dir = self.pp_rootdir+"_npz"

        self.pre_crop_size = [320,320,8] #y,x,z; determines pp data shape (2D easily implementable, but only 3D for now)
        self.min_2d_radius = 6 #in pixels
        self.n_train_samples, self.n_test_samples = 1200, 1000

        # not actually real one-hot encoding (ohe) but contains more info: roi-overlap only within classes.
        self.pp_create_ohe_seg = False
        self.pp_empty_samples_ratio = 0.1

        self.pp_place_radii_mid_bin = True
        self.pp_only_distort_2d = True
        # outer-most intensity of blurred radii, relative to inner-object intensity. <1 for decreasing, > 1 for increasing.
        # e.g.: setting 0.1 means blurred edge has min intensity 10% as large as inner-object intensity.
        self.pp_blur_min_intensity = 0.2

        self.max_instances_per_sample = 1 #how many max instances over all classes per sample (img if 2d, vol if 3d)
        self.max_instances_per_class = self.max_instances_per_sample  # how many max instances per image per class
        self.noise_scale = 0.  # std-dev of gaussian noise

        self.ambigs_sampling = "gaussian" #"gaussian" or "uniform"
        """ radius_calib: gt distort for calibrating uncertainty. Range of gt distortion is inferable from
            image by distinguishing it from the rest of the object.
            blurring width around edge will be shifted so that symmetric rel to orig radius.
            blurring scale: if self.ambigs_sampling is uniform, distribution's non-zero range (b-a) will be sqrt(12)*scale
            since uniform dist has variance (b-a)²/12. b,a will be placed symmetrically around unperturbed radius.
            if sampling is gaussian, then scale parameter sets one std dev, i.e., blurring width will be orig_radius * std_dev * 2.
        """
        self.ambiguities = {
             #set which classes to apply which ambs to below in class labels
             #choose out of: 'outer_radius', 'inner_radius', 'radii_relations'.
             #kind              #probability   #scale (gaussian std, relative to unperturbed value)
            #"outer_radius":     (1.,            0.5),
            #"outer_radius_xy":  (1.,            0.5),
            #"inner_radius":     (0.5,            0.1),
            #"radii_relations":  (0.5,            0.1),
            "radius_calib":     (1.,            1./6)
        }

        # shape choices: 'cylinder', 'block'
        #                        id,    name,       shape,      radius,                 color,              regression,     ambiguities,    gt_distortion
        self.pp_classes = [Label(1,     'cylinder', 'cylinder', ((6,6,1),(40,40,8)),    (*self.blue, 1.),   "radius_2d",    (),             ()),
                           #Label(2,      'block',      'block',        ((6,6,1),(40,40,8)),  (*self.aubergine,1.),  "radii_2d", (), ('radius_calib',))
            ]


        #########################
        #         I/O           #
        #########################

        self.data_sourcedir = '/home/gregor/datasets/toy/cyl1ps_dev'

        if server_env:
            self.data_sourcedir = '/datasets/datasets_ramien/toy/data/cyl1ps_dev_npz'


        self.test_data_sourcedir = os.path.join(self.data_sourcedir, 'test')
        self.data_sourcedir = os.path.join(self.data_sourcedir, "train")

        self.info_df_name = 'info_df.pickle'

        # one out of ['mrcnn', 'retina_net', 'retina_unet', 'detection_unet', 'ufrcnn', 'detection_fpn'].
        self.model = 'mrcnn'
        self.model_path = 'models/{}.py'.format(self.model if not 'retina' in self.model else 'retina_net')
        self.model_path = os.path.join(self.source_dir, self.model_path)


        #########################
        #      Architecture     #
        #########################

        # one out of [2, 3]. dimension the model operates in.
        self.dim = 2

        # 'class', 'regression', 'regression_bin', 'regression_ken_gal'
        # currently only tested mode is a single-task at a time (i.e., only one task in below list)
        # but, in principle, tasks could be combined (e.g., object classes and regression per class)
        self.prediction_tasks = ['class', ]

        self.start_filts = 48 if self.dim == 2 else 18
        self.end_filts = self.start_filts * 4 if self.dim == 2 else self.start_filts * 2
        self.res_architecture = 'resnet50' # 'resnet101' , 'resnet50'
        self.norm = 'instance_norm' # one of None, 'instance_norm', 'batch_norm'
        self.relu = 'relu'
        # one of 'xavier_uniform', 'xavier_normal', or 'kaiming_normal', None (=default = 'kaiming_uniform')
        self.weight_init = None

        self.regression_n_features = 1  # length of regressor target vector


        #########################
        #      Data Loader      #
        #########################

        self.num_epochs = 24
        self.num_train_batches = 100 if self.dim == 2 else 180
        self.batch_size = 20 if self.dim == 2 else 8

        self.n_cv_splits = 4
        # select modalities from preprocessed data
        self.channels = [0]
        self.n_channels = len(self.channels)

        # which channel (mod) to show as bg in plotting, will be extra added to batch if not in self.channels
        self.plot_bg_chan = 0
        self.crop_margin = [20, 20, 1]  # has to be smaller than respective patch_size//2
        self.patch_size_2D = self.pre_crop_size[:2]
        self.patch_size_3D = self.pre_crop_size[:2]+[8]

        # patch_size to be used for training. pre_crop_size is the patch_size before data augmentation.
        self.patch_size = self.patch_size_2D if self.dim == 2 else self.patch_size_3D

        # ratio of free sampled batch elements before class balancing is triggered
        # (>0 to include "empty"/background patches.)
        self.batch_random_ratio = 0.2
        self.balance_target = "class_targets" if 'class' in self.prediction_tasks else "rg_bin_targets"

        self.observables_patient = []
        self.observables_rois = []

        self.seed = 3 #for generating folds

        #############################
        # Colors, Classes, Legends  #
        #############################
        self.plot_frequency = 4

        binary_bin_labels = [binLabel(1,  'r<=25',      (*self.green, 1.),      (1,25)),
                             binLabel(2,  'r>25',       (*self.red, 1.),        (25,))]
        quintuple_bin_labels = [binLabel(1,  'r2-10',   (*self.green, 1.),      (2,10)),
                                binLabel(2,  'r10-20',  (*self.yellow, 1.),     (10,20)),
                                binLabel(3,  'r20-30',  (*self.orange, 1.),     (20,30)),
                                binLabel(4,  'r30-40',  (*self.bright_red, 1.), (30,40)),
                                binLabel(5,  'r>40',    (*self.red, 1.), (40,))]

        # choose here if to do 2-way or 5-way regression-bin classification
        task_spec_bin_labels = quintuple_bin_labels

        self.class_labels = [
            # regression: regression-task label, either value or "(x,y,z)_radius" or "radii".
            # ambiguities: name of above defined ambig to apply to image data (not gt); need to be iterables!
            # gt_distortion: name of ambig to apply to gt only; needs to be iterable!
            #      #id  #name   #shape  #radius     #color              #regression #ambiguities    #gt_distortion
            Label(  0,  'bg',   None,   (0, 0, 0),  (*self.white, 0.),  (0, 0, 0),  (),             ())]
        if "class" in self.prediction_tasks:
            self.class_labels += self.pp_classes
        else:
            self.class_labels += [Label(1, 'object', 'object', ('various',), (*self.orange, 1.), ('radius_2d',), ("various",), ('various',))]


        if any(['regression' in task for task in self.prediction_tasks]):
            self.bin_labels = [binLabel(0,  'bg',       (*self.white, 1.),      (0,))]
            self.bin_labels += task_spec_bin_labels
            self.bin_id2label = {label.id: label for label in self.bin_labels}
            bins = [(min(label.bin_vals), max(label.bin_vals)) for label in self.bin_labels]
            self.bin_id2rg_val = {ix: [np.mean(bin)] for ix, bin in enumerate(bins)}
            self.bin_edges = [(bins[i][1] + bins[i + 1][0]) / 2 for i in range(len(bins) - 1)]
            self.bin_dict = {label.id: label.name for label in self.bin_labels if label.id != 0}

        if self.class_specific_seg:
          self.seg_labels = self.class_labels

        self.box_type2label = {label.name: label for label in self.box_labels}
        self.class_id2label = {label.id: label for label in self.class_labels}
        self.class_dict = {label.id: label.name for label in self.class_labels if label.id != 0}

        self.seg_id2label = {label.id: label for label in self.seg_labels}
        self.cmap = {label.id: label.color for label in self.seg_labels}

        self.plot_prediction_histograms = True
        self.plot_stat_curves = False
        self.has_colorchannels = False
        self.plot_class_ids = True

        self.num_classes = len(self.class_dict)
        self.num_seg_classes = len(self.seg_labels)

        #########################
        #   Data Augmentation   #
        #########################
        self.do_aug = True
        self.da_kwargs = {
            'mirror': True,
            'mirror_axes': tuple(np.arange(0, self.dim, 1)),
            'do_elastic_deform': False,
            'alpha': (500., 1500.),
            'sigma': (40., 45.),
            'do_rotation': False,
            'angle_x': (0., 2 * np.pi),
            'angle_y': (0., 0),
            'angle_z': (0., 0),
            'do_scale': False,
            'scale': (0.8, 1.1),
            'random_crop': False,
            'rand_crop_dist': (self.patch_size[0] / 2. - 3, self.patch_size[1] / 2. - 3),
            'border_mode_data': 'constant',
            'border_cval_data': 0,
            'order_data': 1
        }

        if self.dim == 3:
            self.da_kwargs['do_elastic_deform'] = False
            self.da_kwargs['angle_x'] = (0, 0.0)
            self.da_kwargs['angle_y'] = (0, 0.0)  # must be 0!!
            self.da_kwargs['angle_z'] = (0., 2 * np.pi)

        #########################
        #  Schedule / Selection #
        #########################

        # decide whether to validate on entire patient volumes (like testing) or sampled patches (like training)
        # the former is morge accurate, while the latter is faster (depending on volume size)
        self.val_mode = 'val_sampling' # one of 'val_sampling' , 'val_patient'
        if self.val_mode == 'val_patient':
            self.max_val_patients = 220  # if 'all' iterates over entire val_set once.
        if self.val_mode == 'val_sampling':
            self.num_val_batches = 35 if self.dim==2 else 25

        self.save_n_models = 2
        self.min_save_thresh = 1 if self.dim == 2 else 1  # =wait time in epochs
        if "class" in self.prediction_tasks:
            self.model_selection_criteria = {name + "_ap": 1. for name in self.class_dict.values()}
        elif any("regression" in task for task in self.prediction_tasks):
            self.model_selection_criteria = {name + "_ap": 0.2 for name in self.class_dict.values()}
            self.model_selection_criteria.update({name + "_avp": 0.8 for name in self.class_dict.values()})

        self.lr_decay_factor = 0.25
        self.scheduling_patience = np.ceil(3600 / (self.num_train_batches * self.batch_size))
        self.weight_decay = 3e-5
        self.exclude_from_wd = []
        self.clip_norm = None  # number or None

        #########################
        #   Testing / Plotting  #
        #########################

        self.test_aug_axes = (0,1,(0,1)) # None or list: choices are 0,1,(0,1)
        self.hold_out_test_set = True
        self.max_test_patients = "all"  # number or "all" for all

        self.test_against_exact_gt = True # only True implemented
        self.val_against_exact_gt = False # True is an unrealistic --> irrelevant scenario.
        self.report_score_level = ['rois']  # 'patient' or 'rois' (incl)
        self.patient_class_of_interest = 1
        self.patient_bin_of_interest = 2

        self.eval_bins_separately = False#"additionally" if not 'class' in self.prediction_tasks else False
        self.metrics = ['ap', 'auc', 'dice']
        if any(['regression' in task for task in self.prediction_tasks]):
            self.metrics += ['avp', 'rg_MAE_weighted', 'rg_MAE_weighted_tp',
                             'rg_bin_accuracy_weighted', 'rg_bin_accuracy_weighted_tp']
        if 'aleatoric' in self.model:
            self.metrics += ['rg_uncertainty', 'rg_uncertainty_tp', 'rg_uncertainty_tp_weighted']
        self.evaluate_fold_means = True

        self.ap_match_ious = [0.5]  # threshold(s) for considering a prediction as true positive
        self.min_det_thresh = 0.3

        self.model_max_iou_resolution = 0.2

        # aggregation method for test and val_patient predictions.
        # wbc = weighted box clustering as in https://arxiv.org/pdf/1811.08661.pdf,
        # nms = standard non-maximum suppression, or None = no clustering
        self.clustering = 'wbc'
        # iou thresh (exclusive!) for regarding two preds as concerning the same ROI
        self.clustering_iou = self.model_max_iou_resolution  # has to be larger than desired possible overlap iou of model predictions

        self.merge_2D_to_3D_preds = self.dim==2
        self.merge_3D_iou = self.model_max_iou_resolution
        self.n_test_plots = 1  # per fold and rank

        self.test_n_epochs = self.save_n_models  # should be called n_test_ens, since is number of models to ensemble over during testing
        # is multiplied by (1 + nr of test augs)

        #########################
        #   Assertions          #
        #########################
        if not 'class' in self.prediction_tasks:
            assert self.num_classes == 1

        #########################
        #   Add model specifics #
        #########################

        {'mrcnn': self.add_mrcnn_configs,
         'retina_net': self.add_mrcnn_configs, 'retina_unet': self.add_mrcnn_configs,
         'detection_unet': self.add_det_unet_configs, 'detection_fpn': self.add_det_fpn_configs
         }[self.model]()

    def rg_val_to_bin_id(self, rg_val):
        #only meant for isotropic radii!!
        # only 2D radii (x and y dims) or 1D (x or y) are expected
        return np.round(np.digitize(rg_val, self.bin_edges).mean())


    def add_det_fpn_configs(self):

      self.learning_rate = [3 * 1e-4] * self.num_epochs
      self.dynamic_lr_scheduling = True
      self.scheduling_criterion = 'torch_loss'
      self.scheduling_mode = 'min' if "loss" in self.scheduling_criterion else 'max'

      self.n_roi_candidates = 4 if self.dim == 2 else 6
      # max number of roi candidates to identify per image (slice in 2D, volume in 3D)

      # loss mode: either weighted cross entropy ('wce'), batch-wise dice loss ('dice), or the sum of both ('dice_wce')
      self.seg_loss_mode = 'wce'
      self.wce_weights = [1] * self.num_seg_classes if 'dice' in self.seg_loss_mode else [0.1, 1]

      self.fp_dice_weight = 1 if self.dim == 2 else 1
      # if <1, false positive predictions in foreground are penalized less.

      self.detection_min_confidence = 0.05
      # how to determine score of roi: 'max' or 'median'
      self.score_det = 'max'

    def add_det_unet_configs(self):

      self.learning_rate = [3 * 1e-4] * self.num_epochs
      self.dynamic_lr_scheduling = True
      self.scheduling_criterion = "torch_loss"
      self.scheduling_mode = 'min' if "loss" in self.scheduling_criterion else 'max'

      # max number of roi candidates to identify per image (slice in 2D, volume in 3D)
      self.n_roi_candidates = 4 if self.dim == 2 else 6

      # loss mode: either weighted cross entropy ('wce'), batch-wise dice loss ('dice), or the sum of both ('dice_wce')
      self.seg_loss_mode = 'wce'
      self.wce_weights = [1] * self.num_seg_classes if 'dice' in self.seg_loss_mode else [0.1, 1]
      # if <1, false positive predictions in foreground are penalized less.
      self.fp_dice_weight = 1 if self.dim == 2 else 1

      self.detection_min_confidence = 0.05
      # how to determine score of roi: 'max' or 'median'
      self.score_det = 'max'

      self.init_filts = 32
      self.kernel_size = 3  # ks for horizontal, normal convs
      self.kernel_size_m = 2  # ks for max pool
      self.pad = "same"  # "same" or integer, padding of horizontal convs

    def add_mrcnn_configs(self):

      self.learning_rate = [3e-4] * self.num_epochs
      self.dynamic_lr_scheduling = True  # with scheduler set in exec
      self.scheduling_criterion = max(self.model_selection_criteria, key=self.model_selection_criteria.get)
      self.scheduling_mode = 'min' if "loss" in self.scheduling_criterion else 'max'

      # number of classes for network heads: n_foreground_classes + 1 (background)
      self.head_classes = self.num_classes + 1 if 'class' in self.prediction_tasks else 2

      # feed +/- n neighbouring slices into channel dimension. set to None for no context.
      self.n_3D_context = None
      if self.n_3D_context is not None and self.dim == 2:
        self.n_channels *= (self.n_3D_context * 2 + 1)

      self.detect_while_training = True
      # disable the re-sampling of mask proposals to original size for speed-up.
      # since evaluation is detection-driven (box-matching) and not instance segmentation-driven (iou-matching),
      # mask outputs are optional.
      self.return_masks_in_train = True
      self.return_masks_in_val = True
      self.return_masks_in_test = True

      # feature map strides per pyramid level are inferred from architecture. anchor scales are set accordingly.
      self.backbone_strides = {'xy': [4, 8, 16, 32], 'z': [1, 2, 4, 8]}
      # anchor scales are chosen according to expected object sizes in data set. Default uses only one anchor scale
      # per pyramid level. (outer list are pyramid levels (corresponding to BACKBONE_STRIDES), inner list are scales per level.)
      self.rpn_anchor_scales = {'xy': [[4], [8], [16], [32]], 'z': [[1], [2], [4], [8]]}
      # choose which pyramid levels to extract features from: P2: 0, P3: 1, P4: 2, P5: 3.
      self.pyramid_levels = [0, 1, 2, 3]
      # number of feature maps in rpn. typically lowered in 3D to save gpu-memory.
      self.n_rpn_features = 512 if self.dim == 2 else 64

      # anchor ratios and strides per position in feature maps.
      self.rpn_anchor_ratios = [0.5, 1., 2.]
      self.rpn_anchor_stride = 1
      # Threshold for first stage (RPN) non-maximum suppression (NMS):  LOWER == HARDER SELECTION
      self.rpn_nms_threshold = max(0.7, self.model_max_iou_resolution)

      # loss sampling settings.
      self.rpn_train_anchors_per_image = 32
      self.train_rois_per_image = 6 # per batch_instance
      self.roi_positive_ratio = 0.5
      self.anchor_matching_iou = 0.8

      # k negative example candidates are drawn from a pool of size k*shem_poolsize (stochastic hard-example mining),
      # where k<=#positive examples.
      self.shem_poolsize = 6

      self.pool_size = (7, 7) if self.dim == 2 else (7, 7, 3)
      self.mask_pool_size = (14, 14) if self.dim == 2 else (14, 14, 5)
      self.mask_shape = (28, 28) if self.dim == 2 else (28, 28, 10)

      self.rpn_bbox_std_dev = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
      self.bbox_std_dev = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
      self.window = np.array([0, 0, self.patch_size[0], self.patch_size[1], 0, self.patch_size_3D[2]])
      self.scale = np.array([self.patch_size[0], self.patch_size[1], self.patch_size[0], self.patch_size[1],
                             self.patch_size_3D[2], self.patch_size_3D[2]])  # y1,x1,y2,x2,z1,z2

      if self.dim == 2:
        self.rpn_bbox_std_dev = self.rpn_bbox_std_dev[:4]
        self.bbox_std_dev = self.bbox_std_dev[:4]
        self.window = self.window[:4]
        self.scale = self.scale[:4]

      self.plot_y_max = 1.5
      self.n_plot_rpn_props = 5 if self.dim == 2 else 30  # per batch_instance (slice in 2D / patient in 3D)

      # pre-selection in proposal-layer (stage 1) for NMS-speedup. applied per batch element.
      self.pre_nms_limit = 2000 if self.dim == 2 else 4000

      # n_proposals to be selected after NMS per batch element. too high numbers blow up memory if "detect_while_training" is True,
      # since proposals of the entire batch are forwarded through second stage as one "batch".
      self.roi_chunk_size = 1300 if self.dim == 2 else 500
      self.post_nms_rois_training = 200 * (self.head_classes-1) if self.dim == 2 else 400
      self.post_nms_rois_inference = 200 * (self.head_classes-1)

      # Final selection of detections (refine_detections)
      self.model_max_instances_per_batch_element = 9 if self.dim == 2 else 18 # per batch element and class.
      self.detection_nms_threshold = self.model_max_iou_resolution  # needs to be > 0, otherwise all predictions are one cluster.
      self.model_min_confidence = 0.2  # iou for nms in box refining (directly after heads), should be >0 since ths>=x in mrcnn.py

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
        # whether to use focal loss or SHEM for loss-sample selection
        self.focal_loss = False
        # implement extra anchor-scales according to https://arxiv.org/abs/1708.02002
        self.rpn_anchor_scales['xy'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                        self.rpn_anchor_scales['xy']]
        self.rpn_anchor_scales['z'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                       self.rpn_anchor_scales['z']]
        self.n_anchors_per_pos = len(self.rpn_anchor_ratios) * 3

        # pre-selection of detections for NMS-speedup. per entire batch.
        self.pre_nms_limit = (500 if self.dim == 2 else 6250) * self.batch_size

        # anchor matching iou is lower than in Mask R-CNN according to https://arxiv.org/abs/1708.02002
        self.anchor_matching_iou = 0.7

        if self.model == 'retina_unet':
          self.operate_stride1 = True
