__author__ = ''
#credit Paul F. Jaeger

#########################
#     Example Config    #
#########################

import os
import sys

import numpy as np
from collections import namedtuple

sys.path.append('../')
from default_configs import DefaultConfigs

class Configs(DefaultConfigs):

    def __init__(self, server_env=None):
        super(Configs, self).__init__(server_env)

        self.dim = 2

        #########################
        #         I/O           #
        #########################

        self.data_sourcedir = "/mnt/HDD2TB/Documents/data/cityscapes/cs_20190715/"
        if server_env:
            #self.source_dir = '/home/ramien/medicaldetectiontoolkit/'
            self.data_sourcedir = '/datasets/data_ramien/cityscapes/cs_20190715_npz/'
            #self.data_sourcedir = "/mnt/HDD2TB/Documents/data/cityscapes/cs_6c_inst_only/"

        self.datapath = "leftImg8bit/"
        self.targetspath = "gtFine/"
        
        self.cities = {'train':['dusseldorf', 'aachen', 'bochum', 'cologne', 'erfurt',
                                'hamburg', 'hanover', 'jena', 'krefeld', 'monchengladbach', 
                                'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar',
                                'zurich'], 
                        'val':['frankfurt', 'munster'], 
                        'test':['bremen', 'darmstadt', 'lindau'] }
        self.set_splits = ["train", "val", "test"] # for training and val, mixed up
        # test cities are not held out

        self.info_dict_name = 'city_info.pkl'
        self.info_dict_path = os.path.join(self.data_sourcedir, self.info_dict_name)
        self.config_path = os.path.realpath(__file__)
        self.backbone_path = 'models/backbone.py'

        # one out of ['mrcnn', 'retina_net', 'retina_unet', 'detection_unet', 'detection_fpn'].
        self.model = 'retina_unet'
        self.model_path = 'models/{}.py'.format(self.model if not 'retina' in self.model else 'retina_net')
        self.model_path = os.path.join(self.source_dir, self.model_path)

        self.select_prototype_subset = None
            
        #########################
        #      Preprocessing    #
        #########################
        self.prepro = {
            
		 'data_dir': '/mnt/HDD2TB/Documents/data/cityscapes_raw/', #raw files (input), needs to end with "/"
         'targettype': "gtFine_instanceIds",
         'set_splits': ["train", "val", "test"],
         
         'img_target_size': np.array([256, 512])*4, #y,x
         
         'output_directory': self.data_sourcedir,
		 
         'center_of_mass_crop': True, #not implemented
         #'pre_crop_size': , #z,y,x
		 'normalization': {'percentiles':[1., 99.]},#not implemented
	     'interpolation': 'nearest', #not implemented
         
         'info_dict_path': self.info_dict_path,
         
         'npz_dir' : self.data_sourcedir[:-1]+"_npz" #if not None: convert to npz, copy data here
         }

        #########################
        #      Architecture     #
        #########################
        # 'class', 'regression', 'regression_ken_gal'
        # 'class': standard object classification per roi, pairwise combinable with each of below tasks.
        # 'class' is only option implemented for CityScapes data set.
        self.prediction_tasks = ['class',]
        self.start_filts = 52
        self.end_filts = self.start_filts * 4
        self.res_architecture = 'resnet101'  # 'resnet101' , 'resnet50'
        self.weight_init = None  # 'kaiming', 'xavier' or None for pytorch default
        self.norm = 'instance_norm'  # 'batch_norm' # one of 'None', 'instance_norm', 'batch_norm'
        self.relu = 'relu'

        #########################
        #      Data Loader      #
        #########################

        self.seed = 17
        self.n_workers = 16 if server_env else os.cpu_count()
        
        self.batch_size = 8
        self.n_cv_splits = 10 #at least 2 (train, val)
        
        self.num_classes = None #set below #for instance classification (excl background)
        self.num_seg_classes = None #set below #incl background
        
        self.create_bounding_box_targets = True
        self.class_specific_seg = True
        
        self.channels = [0,1,2] 
        self.pre_crop_size = self.prepro['img_target_size'] # y,x
        self.crop_margin   = [10,10] #has to be smaller than respective patch_size//2
        self.patch_size_2D = [256, 512] #self.pre_crop_size #would be better to save as tuple since should not be altered
        self.patch_size_3D = self.patch_size_2D + [1]
        self.patch_size = self.patch_size_2D

        self.balance_target = "class_targets"
        # ratio of fully random patients drawn during batch generation
        # resulting batch random count is rounded down to closest integer
        self.batch_random_ratio = 0.2

        self.observables_patient = []
        self.observables_rois = []
        
        #########################
        #   Data Augmentation   #
        #########################
        #the angle rotations are implemented incorrectly in batchgenerators! in 2D,
        #the x-axis angle controls the z-axis angle.
        self.do_aug = True
        self.da_kwargs = {
            'mirror': True,
            'mirror_axes': (1,), #image axes, (batch and channel are ignored, i.e., actual tensor dims are +2)
        	'random_crop': True,
        	'rand_crop_dist': (self.patch_size[0] / 2., self.patch_size[1] / 2.),
        	'do_elastic_deform': True,
        	'alpha': (0., 1000.),
        	'sigma': (28., 30.),
        	'do_rotation': True,
        	'angle_x': (-np.pi / 8., np.pi / 8.),
        	'angle_y': (0.,0.),
        	'angle_z': (0.,0.),
        	'do_scale': True,
        	'scale': (0.6, 1.4),
        	'border_mode_data': 'constant',
            'gamma_range': (0.6, 1.4)
        }        
        
        #################################
        #  Schedule / Selection / Optim #
        #################################
        #mrcnn paper: ~2.56m samples seen during coco-dataset training
        self.num_epochs = 400
        self.num_train_batches = 600
        
        self.do_validation = True
        # decide whether to validate on entire patient volumes (like testing) or sampled patches (like training)
        # the former is morge accurate, while the latter is faster (depending on volume size)
        self.val_mode = 'val_sampling' # one of 'val_sampling', 'val_patient'
        # if 'all' iterates over entire val_set once.
        self.num_val_batches = "all" # for val_sampling
        
        self.save_n_models = 3
        self.min_save_thresh = 1 # in epochs
        self.model_selection_criteria = {"human_ap": 1., "vehicle_ap": 0.9}
        self.warm_up = 0

        self.learning_rate = [5*1e-4] * self.num_epochs
        self.dynamic_lr_scheduling = True #with scheduler set in exec
        self.lr_decay_factor = 0.5
        self.scheduling_patience = int(self.num_epochs//10)
        self.weight_decay = 1e-6
        self.clip_norm = None  # number or None

        #########################
        #   Colors and Legends  #
        #########################
        self.plot_frequency = 5

        #colors
        self.color_palette = [self.red, self.blue, self.green, self.orange, self.aubergine,
                              self.yellow, self.gray, self.cyan, self.black]
        
        #legends
        Label = namedtuple( 'Label' , [
            'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                            # We use them to uniquely name a class
            'ppId'          , # An integer ID that is associated with this label.
                            # The IDs are used to represent the label in ground truth images
                            # An ID of -1 means that this label does not have an ID and thus
                            # is ignored when creating ground truth images (e.g. license plate).
                            # Do not modify these IDs, since exactly these IDs are expected by the
                            # evaluation server.
            'id'     , # Feel free to modify these IDs as suitable for your method.
                            # Max value is 255!
            'category'    , # The name of the category that this label belongs to
            'categoryId'  , # The ID of this category. Used to create ground truth images
                            # on category level.
            'hasInstances', # Whether this label distinguishes between single instances or not
            'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                            # during evaluations or not
            'color'       , # The color of this label
            ] )
        segLabel = namedtuple( "segLabel", ["name", "id", "color"])
        boxLabel = namedtuple( 'boxLabel', [ "name", "color"]) 
        
        self.labels = [
            #       name                   ppId         id   category            catId     hasInstances   ignoreInEval   color
            Label(  'ignore'               ,  0 ,        0 , 'void'            , 0       , False        , True         , (  0.,  0.,  0., 1.) ),
            Label(  'ego vehicle'          ,  1 ,        0 , 'void'            , 0       , False        , True         , (  0.,  0.,  0., 1.) ),
            Label(  'rectification border' ,  2 ,        0 , 'void'            , 0       , False        , True         , (  0.,  0.,  0., 1.) ),
            Label(  'out of roi'           ,  3 ,        0 , 'void'            , 0       , False        , True         , (  0.,  0.,  0., 1.) ),
            Label(  'static'               ,  4 ,        0 , 'void'            , 0       , False        , True         , (  0.,  0.,  0., 1.) ),
            Label(  'dynamic'              ,  5 ,        0 , 'void'            , 0       , False        , True         , (0.44, 0.29,  0., 1.) ),
            Label(  'ground'               ,  6 ,        0 , 'void'            , 0       , False        , True         , ( 0.32,  0., 0.32, 1.) ),
            Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (0.5, 0.25, 0.5, 1.) ),
            Label(  'sidewalk'             ,  8 ,        0 , 'flat'            , 1       , False        , False        , (0.96, 0.14, 0.5, 1.) ),
            Label(  'parking'              ,  9 ,        0 , 'flat'            , 1       , False        , True         , (0.98, 0.67, 0.63, 1.) ),
            Label(  'rail track'           , 10 ,        0 , 'flat'            , 1       , False        , True         , ( 0.9,  0.59, 0.55, 1.) ),
            Label(  'building'             , 11 ,        0 , 'construction'    , 2       , False        , False        , ( 0.27, 0.27, 0.27, 1.) ),
            Label(  'wall'                 , 12 ,        0 , 'construction'    , 2       , False        , False        , (0.4,0.4,0.61, 1.) ),
            Label(  'fence'                , 13 ,        0 , 'construction'    , 2       , False        , False        , (0.75,0.6,0.6, 1.) ),
            Label(  'guard rail'           , 14 ,        0 , 'construction'    , 2       , False        , True         , (0.71,0.65,0.71, 1.) ),
            Label(  'bridge'               , 15 ,        0 , 'construction'    , 2       , False        , True         , (0.59,0.39,0.39, 1.) ),
            Label(  'tunnel'               , 16 ,        0 , 'construction'    , 2       , False        , True         , (0.59,0.47, 0.35, 1.) ),
            Label(  'pole'                 , 17 ,        0 , 'object'          , 3       , False        , False        , (0.6,0.6,0.6, 1.) ),
            Label(  'polegroup'            , 18 ,        0 , 'object'          , 3       , False        , True         , (0.6,0.6,0.6, 1.) ),
            Label(  'traffic light'        , 19 ,        0 , 'object'          , 3       , False        , False        , (0.98,0.67, 0.12, 1.) ),
            Label(  'traffic sign'         , 20 ,        0 , 'object'          , 3       , False        , False        , (0.86,0.86, 0., 1.) ),
            Label(  'vegetation'           , 21 ,        0 , 'nature'          , 4       , False        , False        , (0.42,0.56, 0.14, 1.) ),
            Label(  'terrain'              , 22 ,        0 , 'nature'          , 4       , False        , False        , (0.6, 0.98,0.6, 1.) ),
            Label(  'sky'                  , 23 ,        0 , 'sky'             , 5       , False        , False        , (0.27,0.51,0.71, 1.) ),
            Label(  'person'               , 24 ,        1 , 'human'           , 6       , True         , False        , (0.86, 0.08, 0.24, 1.) ),
            Label(  'rider'                , 25 ,        1 , 'human'           , 6       , True         , False        , (1.,  0.,  0., 1.) ),
            Label(  'car'                  , 26 ,        2 , 'vehicle'         , 7       , True         , False        , (  0., 0.,0.56, 1.) ),
            Label(  'truck'                , 27 ,        2 , 'vehicle'         , 7       , True         , False        , (  0.,  0., 0.27, 1.) ),
            Label(  'bus'                  , 28 ,        2 , 'vehicle'         , 7       , True         , False        , (  0., 0.24,0.39, 1.) ),
            Label(  'caravan'              , 29 ,        2 , 'vehicle'         , 7       , True         , True         , (  0.,  0., 0.35, 1.) ),
            Label(  'trailer'              , 30 ,        2 , 'vehicle'         , 7       , True         , True         , (  0.,  0.,0.43, 1.) ),
            Label(  'train'                , 31 ,        2 , 'vehicle'         , 7       , True         , False        , (  0., 0.31,0.39, 1.) ),
            Label(  'motorcycle'           , 32 ,        2 , 'vehicle'         , 7       , True         , False        , (  0.,  0., 0.9, 1.) ),
            Label(  'bicycle'              , 33 ,        2 , 'vehicle'         , 7       , True         , False        , (0.47, 0.04, 0.13, 1.) ),
            Label(  'license plate'        , -1 ,        0 , 'vehicle'         , 7       , False        , True         , (  0.,  0., 0.56, 1.) ),
            Label(  'background'           , -1 ,        0 , 'void'            , 0       , False        , True         , (  0.,  0., 0.0, 0.) ),
            Label(  'vehicle'              , 33 ,        2 , 'vehicle'         , 7       , True         , False        , (*self.aubergine, 1.)  ),
            Label(  'human'                , 25 ,        1 , 'human'           , 6       , True         , False        , (*self.blue, 1.) )
        ]
        # evtl problem: class-ids (trainIds) don't start with 0 for the first class, 0 is bg.
        #WONT WORK: class ids need to start at 0 (excluding bg!) and be consecutively numbered 

        self.ppId2id = { label.ppId : label.id for label in self.labels}
        self.class_id2label = { label.id : label for label in self.labels}
        self.class_cmap = {label.id : label.color for label in self.labels}
        self.class_dict = {label.id : label.name for label in self.labels if label.id!=0}
        #c_dict: only for evaluation, remove bg class.
        
        self.box_type2label = {label.name : label for label in self.box_labels}
        self.box_color_palette = {label.name:label.color for label in self.box_labels}

        if self.class_specific_seg:
            self.seg_labels = [label for label in self.class_id2label.values()]
        else:
            self.seg_labels = [
                    #           name    id  color
                    segLabel(  "bg" ,   0,  (1.,1.,1.,0.) ),
                    segLabel(  "fg" ,   1,  (*self.orange, .8))
                    ]

        self.seg_id2label = {label.id : label for label in self.seg_labels}
        self.cmap = {label.id : label.color for label in self.seg_labels}
        
        self.plot_prediction_histograms = True
        self.plot_stat_curves = False
        self.has_colorchannels = True
        self.plot_class_ids = True
        
        self.num_classes = len(self.class_dict)
        self.num_seg_classes = len(self.seg_labels)

        #########################
        #   Testing             #
        #########################

        self.test_aug_axes = None #None or list: choices are 2,3,(2,3)
        self.held_out_test_set = False
        self.max_test_patients = 'all' # 'all' for all
        self.report_score_level = ['rois',]  # choose list from 'patient', 'rois'
        self.patient_class_of_interest = 1
        
        self.metrics = ['ap', 'dice']
        self.ap_match_ious = [0.1]  # threshold(s) for considering a prediction as true positive
        # aggregation method for test and val_patient predictions.
        # wbc = weighted box clustering as in https://arxiv.org/pdf/1811.08661.pdf,
        # nms = standard non-maximum suppression, or None = no clustering
        self.clustering = 'wbc'
        # iou thresh (exclusive!) for regarding two preds as concerning the same ROI
        self.clustering_iou = 0.1  # has to be larger than desired possible overlap iou of model predictions

        self.min_det_thresh = 0.06
        self.merge_2D_to_3D_preds = False
        
        self.n_test_plots = 1 #per fold and rankself.ap_match_ious = [0.1] #threshold(s) for considering a prediction as true positive
        self.test_n_epochs = self.save_n_models


        #########################
        # shared model settings #
        #########################

        # max number of roi candidates to identify per image and class (slice in 2D, volume in 3D)
        self.n_roi_candidates = 100
        
        #########################
        #   Add model specifics #
        #########################
        
        {'mrcnn': self.add_mrcnn_configs, 'retina_net': self.add_mrcnn_configs, 'retina_unet': self.add_mrcnn_configs
         }[self.model]()

    def add_mrcnn_configs(self):

        self.scheduling_criterion = max(self.model_selection_criteria, key=self.model_selection_criteria.get)
        self.scheduling_mode = 'min' if "loss" in self.scheduling_criterion else 'max'

        # number of classes for network heads: n_foreground_classes + 1 (background)
        self.head_classes = self.num_classes + 1

        # seg_classes here refers to the first stage classifier (RPN) reallY?

        # feed +/- n neighbouring slices into channel dimension. set to None for no context.
        self.n_3D_context = None


        self.frcnn_mode = False

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
        self.rpn_nms_threshold = 0.7

        # loss sampling settings.
        self.rpn_train_anchors_per_image = 8
        self.train_rois_per_image = 10  # per batch_instance
        self.roi_positive_ratio = 0.5
        self.anchor_matching_iou = 0.8

        # k negative example candidates are drawn from a pool of size k*shem_poolsize (stochastic hard-example mining),
        # where k<=#positive examples.
        self.shem_poolsize = 3

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
        self.n_plot_rpn_props = 5 # per batch_instance (slice in 2D / patient in 3D)

        # pre-selection in proposal-layer (stage 1) for NMS-speedup. applied per batch element.
        self.pre_nms_limit = 3000

        # n_proposals to be selected after NMS per batch element. too high numbers blow up memory if "detect_while_training" is True,
        # since proposals of the entire batch are forwarded through second stage as one "batch".
        self.roi_batch_size = 2500
        self.post_nms_rois_training = 500
        self.post_nms_rois_inference = 500

        # Final selection of detections (refine_detections)
        self.model_max_instances_per_batch_element = 50 # per batch element and class.
        self.detection_nms_threshold = 1e-5  # needs to be > 0, otherwise all predictions are one cluster.
        self.model_min_confidence = 0.05  # iou for nms in box refining (directly after heads), should be >0 since ths>=x in mrcnn.py

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
            # implement extra anchor-scales according to https://arxiv.org/abs/1708.02002
            self.rpn_anchor_scales['xy'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                            self.rpn_anchor_scales['xy']]
            self.rpn_anchor_scales['z'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                           self.rpn_anchor_scales['z']]
            self.n_anchors_per_pos = len(self.rpn_anchor_ratios) * 3

            self.n_rpn_features = 256 if self.dim == 2 else 64

            # pre-selection of detections for NMS-speedup. per entire batch.
            self.pre_nms_limit = 10000 if self.dim == 2 else 30000

            # anchor matching iou is lower than in Mask R-CNN according to https://arxiv.org/abs/1708.02002
            self.anchor_matching_iou = 0.5

            if self.model == 'retina_unet':
                self.operate_stride1 = True