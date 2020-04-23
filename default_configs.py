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

"""Default Configurations script. Avoids changing configs of all experiments if general settings are to be changed."""

import os
from collections import namedtuple

boxLabel = namedtuple('boxLabel', ["name", "color"])

class DefaultConfigs:

    def __init__(self, server_env=None, dim=2):
        self.server_env = server_env
        self.cuda_benchmark = True
        self.sysmetrics_interval = 2 # set > 0 to record system metrics to tboard with this time span in seconds.
        #########################
        #         I/O           #
        #########################

        self.dim = dim
        # int [0 < dataset_size]. select n patients from dataset for prototyping.
        self.select_prototype_subset = None

        # some default paths.
        self.source_dir = os.path.dirname(os.path.realpath(__file__)) # current dir.
        self.backbone_path = os.path.join(self.source_dir, 'models/backbone.py')
        self.input_df_name = 'info_df.pickle'


        if server_env:
            self.select_prototype_subset = None

        #########################
        #      Colors/legends   #
        #########################

        # in part from solarized theme.
        self.black = (0.1, 0.05, 0.)
        self.gray = (0.514, 0.580, 0.588)
        self.beige = (1., 1., 0.85)
        self.white = (0.992, 0.965, 0.890)

        self.green = (0.659, 0.792, 0.251)  # [168, 202, 64]
        self.dark_green = (0.522, 0.600, 0.000) # [133.11, 153.  ,   0.  ]
        self.cyan = (0.165, 0.631, 0.596)  # [ 42.075, 160.905, 151.98 ]
        self.bright_blue = (0.85, 0.95, 1.)
        self.blue = (0.149, 0.545, 0.824) # [ 37.995, 138.975, 210.12 ]
        self.dkfz_blue = (0, 75. / 255, 142. / 255)
        self.dark_blue = (0.027, 0.212, 0.259) # [ 6.885, 54.06 , 66.045]
        self.purple = (0.424, 0.443, 0.769) # [108.12 , 112.965, 196.095]
        self.aubergine = (0.62, 0.21, 0.44)  # [ 157,  53 ,  111]
        self.magenta = (0.827, 0.212, 0.510) # [210.885,  54.06 , 130.05 ]
        self.coral = (1., 0.251, 0.4) # [255,64,102]
        self.bright_red = (1., 0.15, 0.1)  # [255, 38.25, 25.5]
        self.brighter_red = (0.863, 0.196, 0.184) # [220.065,  49.98 ,  46.92 ]
        self.red = (0.87, 0.05, 0.01)  # [ 223, 13, 2]
        self.dark_red = (0.6, 0.04, 0.005)
        self.orange = (0.91, 0.33, 0.125)  # [ 232.05 ,   84.15 ,   31.875]
        self.dark_orange = (0.796, 0.294, 0.086) #[202.98,  74.97,  21.93]
        self.yellow = (0.95, 0.9, 0.02)  # [ 242.25,  229.5 ,    5.1 ]
        self.dark_yellow = (0.710, 0.537, 0.000) # [181.05 , 136.935,   0.   ]


        self.color_palette = [self.blue, self.dark_blue, self.aubergine, self.green, self.yellow, self.orange, self.red,
                              self.cyan, self.black]

        self.box_labels = [
            #           name            color
            boxLabel("det", self.blue),
            boxLabel("prop", self.gray),
            boxLabel("pos_anchor", self.cyan),
            boxLabel("neg_anchor", self.cyan),
            boxLabel("neg_class", self.green),
            boxLabel("pos_class", self.aubergine),
            boxLabel("gt", self.red)
        ]  # neg and pos in a medical sense, i.e., pos=positive diagnostic finding

        self.box_type2label = {label.name: label for label in self.box_labels}
        self.box_color_palette = {label.name: label.color for label in self.box_labels}

        # whether the input data is mono-channel or RGB/rgb
        self.has_colorchannels = False

        #########################
        #      Data Loader      #
        #########################

        #random seed for fold_generator and batch_generator.
        self.seed = 0

        #number of threads for multithreaded tasks like batch generation, wcs, merge2dto3d
        self.n_workers = 16 if server_env else os.cpu_count()

        self.create_bounding_box_targets = True
        self.class_specific_seg = True  # False if self.model=="mrcnn" else True
        self.max_val_patients = "all"
        #########################
        #      Architecture      #
        #########################

        self.prediction_tasks = ["class"]  # 'class', 'regression_class', 'regression_kendall', 'regression_feindt'

        self.weight_decay = 0.0

        # nonlinearity to be applied after convs with nonlinearity. one of 'relu' or 'leaky_relu'
        self.relu = 'relu'

        # if True initializes weights as specified in model script. else use default Pytorch init.
        self.weight_init = None

        # if True adds high-res decoder levels to feature pyramid: P1 + P0. (e.g. set to true in retina_unet configs)
        self.operate_stride1 = False

        #########################
        #  Optimization         #
        #########################

        self.optimizer = "ADAMW" # "ADAMW" or "SGD" or implemented additionals

        #########################
        #  Schedule             #
        #########################

        # number of folds in cross validation.
        self.n_cv_splits = 5

        #########################
        #   Testing / Plotting  #
        #########################

        # perform mirroring at test time. (only XY. Z not done to not blow up predictions times).
        self.test_aug = True

        # if True, test data lies in a separate folder and is not part of the cross validation.
        self.hold_out_test_set = False
        # if hold-out test set: if ensemble_folds is True, predictions of all folds on the common hold-out test set
        # are aggregated (like ensemble members). if False, each fold's parameters are evaluated separately on the test
        # set and the evaluations are aggregated (like normal cross-validation folds).
        self.ensemble_folds = False

        # if hold_out_test_set provided, ensemble predictions over models of all trained cv-folds.
        self.ensemble_folds = False

        # what metrics to evaluate
        self.metrics = ['ap']
        # whether to evaluate fold means when evaluating over more than one fold
        self.evaluate_fold_means = False

        # how often (in nr of epochs) to plot example batches during train/val
        self.plot_frequency = 1

        # color specifications for all box_types in prediction_plot.
        self.box_color_palette = {'det': 'b', 'gt': 'r', 'neg_class': 'purple',
                                  'prop': 'w', 'pos_class': 'g', 'pos_anchor': 'c', 'neg_anchor': 'c'}

        # scan over confidence score in evaluation to optimize it on the validation set.
        self.scan_det_thresh = False

        # plots roc-curves / prc-curves in evaluation.
        self.plot_stat_curves = False

        # if True: evaluate average precision per patient id and average over per-pid results,
        #     instead of computing one ap over whole data set.
        self.per_patient_ap = False

        # threshold for clustering 2D box predictions to 3D Cubes. Overlap is computed in XY.
        self.merge_3D_iou = 0.1

        # number or "all" for all
        self.max_test_patients = "all"

        #########################
        #   MRCNN               #
        #########################

        # if True, mask loss is not applied. used for data sets, where no pixel-wise annotations are provided.
        self.frcnn_mode = False

        self.return_masks_in_train = False
        # if True, unmolds masks in Mask R-CNN to full-res for plotting/monitoring.
        self.return_masks_in_val = False
        self.return_masks_in_test = False # needed if doing instance segmentation. evaluation not yet implemented.

        # add P6 to Feature Pyramid Network.
        self.sixth_pooling = False


        #########################
        #   RetinaNet           #
        #########################
        self.focal_loss = False
        self.focal_loss_gamma = 2.
