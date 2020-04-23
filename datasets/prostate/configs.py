__author__ = ''
#credit Paul F. Jaeger

#########################
#     Example Config    #
#########################

import os
import sys
import pickle

import numpy as np
import torch

from collections import namedtuple

from default_configs import DefaultConfigs

def load_obj(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

# legends, nested classes are not handled well in multiprocessing! hence, Label class def in outer scope
Label = namedtuple("Label", ['id', 'name', 'color', 'gleasons'])
binLabel = namedtuple("Label", ['id', 'name', 'color', 'gleasons', 'bin_vals'])


class Configs(DefaultConfigs): #todo change to Configs

    def __init__(self, server_env=None):
        #########################
        #         General       #
        #########################
        super(Configs, self).__init__(server_env)

        #########################
        #         I/O           #
        #########################

        self.data_sourcedir = "/mnt/HDD2TB/Documents/data/prostate/data_di_250519_ps384_gs6071/"
        #self.data_sourcedir = "/mnt/HDD2TB/Documents/data/prostate/data_t2_250519_ps384_gs6071/"
        #self.data_sourcedir = "/mnt/HDD2TB/Documents/data/prostate/data_analysis/"

        if server_env:
            self.data_sourcedir = "/datasets/data_ramien/prostate/data_di_250519_ps384_gs6071_npz/"
            #self.data_sourcedir = '/datasets/data_ramien/prostate/data_t2_250519_ps384_gs6071_npz/'
            #self.data_sourcedir = "/mnt/HDD2TB/Documents/data/prostate/data_di_ana_151118_ps384_gs60/"

        self.histo_dir = os.path.join(self.data_sourcedir,"histos/")
        self.info_dict_name = 'master_info.pkl'
        self.info_dict_path = os.path.join(self.data_sourcedir, self.info_dict_name)

        self.config_path = os.path.realpath(__file__)

        # one out of ['mrcnn', 'retina_net', 'retina_unet', 'detection_fpn'].
        self.model = 'detection_fpn'
        self.model_path = 'models/{}.py'.format(self.model if not 'retina' in self.model else 'retina_net')
        self.model_path = os.path.join(self.source_dir,self.model_path)
                       
        self.select_prototype_subset = None

        #########################
        #      Preprocessing    #
        #########################
        self.missing_pz_subjects = [#189, 196, 198, 205, 211, 214, 215, 217, 218, 219, 220,
                                     #223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
                                     #234, 235, 236, 237, 238, 239, 240, 241, 242, 244, 258,
                                     #261, 262, 264, 267, 268, 269, 270, 271, 273, 275, 276,
                                     #277, 278, 283
                                    ]
        self.no_bval_radval_subjects = [57] #this guy has master id 222
        
        self.prepro = {
            'data_dir': '/home/gregor/networkdrives/E132-Projekte/Move_to_E132-Rohdaten/Prisma_Master/Daten/',
            'dir_spec': 'Master',
            #'images': {'t2': 'T2TRA', 'adc': 'ADC1500', 'b50': 'BVAL50', 'b500': 'BVAL500',
            #     'b1000': 'BVAL1000', 'b1500': 'BVAL1500'},
            #'images': {'adc': 'ADC1500', 'b50': 'BVAL50', 'b500': 'BVAL500', 'b1000': 'BVAL1000', 'b1500': 'BVAL1500'},
            'images': {'t2': 'T2TRA'},
            'anatomical_masks': ['seg_T2_PRO'], # try: 'seg_T2_PRO','seg_T2_PZ', 'seg_ADC_PRO', 'seg_ADC_PZ',
            'merge_mode' : 'union', #if registered data w/ two gts: take 'union' or 'adc' or 't2' of gt
            'rename_tags': {'seg_ADC_PRO':"pro", 'seg_T2_PRO':"pro", 'seg_ADC_PZ':"pz", 'seg_T2_PZ':"pz"},
            'lesion_postfix': '_Re', #lesion files are tagged seg_MOD_LESx
            'img_postfix': "_resampled2", #"_resampled2_registered",
            'overall_postfix': ".nrrd", #including filetype ending!

            'histo_dir': '/home/gregor/networkdrives/E132-Projekte/Move_to_E132-Rohdaten/Prisma_Master/Dokumente/',
            'histo_dir_out': self.histo_dir,
            'histo_lesion_based': 'MasterHistoAll.csv',
            'histo_patient_based': 'MasterPatientbasedAll_clean.csv',
            'histo_id_column_name': 'Master_ID',
            'histo_pb_id_column_name': 'Master_ID_Short', #for patient histo

            'excluded_prisma_subjects': [],
            'excluded_radval_subjects': self.no_bval_radval_subjects,
            'excluded_master_subjects': self.missing_pz_subjects,

            'seg_labels': {'tz': 0, 'pz': 0, 'lesions':'roi'},
            #set as hard label or 'roi' to have seg labels represent obj instance count
            #if not given 'lesions' are numbered highest seg label +lesion-nr-in-histofile
            'class_labels': {'lesions':'gleason'}, #0 is not bg, but first fg class!
            #i.e., prepro labels are shifted by -1 towards later training labels in gt, legends, dicts, etc.
            #evtly set lesions to 'gleason' and check gleason remap in prepro
            #'gleason_thresh': 71,
            'gleason_mapping': {0: -1, 60:0, 71:1, 72:1, 80:1, 90:1, 91:1, 92:1},
            'gleason_map': self.gleason_map, #see below
            'color_palette': [self.green, self.red],

            'output_directory': self.data_sourcedir,

            'modalities2concat' : "all", #['t2', 'adc','b50','b500','b1000','b1500'], #will be concatenated on colorchannel
            'center_of_mass_crop': True,
            'mod_scaling' : (1,1,1), #z,y,x
            'pre_crop_size': [20, 384, 384], #z,y,x, z-cropping and non-square not implemented atm!!
            'swap_yx_to_xy': False, #change final spatial shape from z,y,x to z,x,y
            'normalization': {'percentiles':[1., 99.]},
            'interpolation': 'nearest',

            'observables_patient': ['Original_ID', 'GSBx', 'PIRADS2', 'PSA'],
            'observables_rois': ['lesion_gleasons'],

            'info_dict_path': self.info_dict_path,

            'npz_dir' : self.data_sourcedir[:-1]+"_npz" #if not None: convert to npz, copy data here
         }
        if self.prepro["modalities2concat"] == "all":
            self.prepro["modalities2concat"] = list(self.prepro["images"].keys())

        #########################
        #      Architecture     #
        #########################

        # dimension the model operates in. one out of [2, 3].
        self.dim = 2

        # 'class': standard object classification per roi, pairwise combinable with each of below tasks.
        # if 'class' is omitted from tasks, object classes will be fg/bg (1/0) from RPN.
        # 'regression': regress some vector per each roi
        # 'regression_ken_gal': use kendall-gal uncertainty sigma
        # 'regression_bin': classify each roi into a bin related to a regression scale
        self.prediction_tasks = ['class',]

        self.start_filts = 48 if self.dim == 2 else 18
        self.end_filts = self.start_filts * 4 if self.dim == 2 else self.start_filts * 2
        self.res_architecture = 'resnet50' # 'resnet101' or 'resnet50'
        self.weight_init = None #'kaiming_normal' #, 'xavier' or None-->pytorch standard,
        self.norm = None #'instance_norm' # one of 'None', 'instance_norm', 'batch_norm'
        self.relu = 'relu' # 'relu' or 'leaky_relu'

        self.regression_n_features = 1 #length of regressor target vector (always 1D)

        #########################
        #      Data Loader      #
        #########################

        self.seed = 17
        self.n_workers = 16 if server_env else os.cpu_count()

        self.batch_size = 10 if self.dim == 2 else 6

        self.channels = [1, 2, 3, 4]  # modalities2load, see prepo
        self.n_channels = len(self.channels)  # for compatibility, but actually redundant
        # which channel (mod) to show as bg in plotting, will be extra added to batch if not in self.channels
        self.plot_bg_chan = 0
        self.pre_crop_size = list(np.array(self.prepro['pre_crop_size'])[[1, 2, 0]])  # now y,x,z
        self.crop_margin = [20, 20, 1]  # has to be smaller than respective patch_size//2
        self.patch_size_2D = self.pre_crop_size[:2] #[288, 288]
        self.patch_size_3D = self.pre_crop_size[:2] + [8]  # only numbers divisible by 2 multiple times
        # (at least 5 times for x,y, at least 3 for z)!
        # otherwise likely to produce error in crop fct or net
        self.patch_size = self.patch_size_2D if self.dim == 2 else self.patch_size_3D

        self.balance_target = "class_targets" if 'class' in self.prediction_tasks else 'rg_bin_targets'
        # ratio of fully random patients drawn during batch generation
        # resulting batch random count is rounded down to closest integer
        self.batch_random_ratio = 0.2 if self.dim==2 else 0.4

        self.observables_patient = ['Original_ID', 'GSBx', 'PIRADS2']
        self.observables_rois = ['lesion_gleasons']

        self.regression_target = "lesion_gleasons"  # name of the info_dict entry holding regression targets
        # linear mapping
        self.rg_map = {0: 0, 60: 1, 71: 2, 72: 3, 80: 4, 90: 5, 91: 6, 92: 7, None: 0}
        # non-linear mapping
        #self.rg_map = {0: 0, 60: 1, 71: 6, 72: 7.5, 80: 9, 90: 10, 91: 10, 92: 10, None: 0}

        #########################
        #   Colors and Legends  #
        #########################
        self.plot_frequency = 5

        # colors
        self.gravity_col_palette = [self.green, self.yellow, self.orange, self.bright_red, self.red, self.dark_red]

        self.gs_labels = [
            Label(0,    'bg',   self.gray,     (0,)),
            Label(60,   'GS60', self.dark_green,     (60,)),
            Label(71,   'GS71', self.dark_yellow,    (71,)),
            Label(72,   'GS72', self.orange,    (72,)),
            Label(80,   'GS80', self.brighter_red,(80,)),
            Label(90,   'GS90', self.bright_red,       (90,)),
            Label(91,   'GS91', self.red,       (91,)),
            Label(92,   'GS92', self.dark_red,  (92,))
        ]
        self.gs2label = {label.id: label for label in self.gs_labels}


        binary_cl_labels = [Label(1, 'benign',      (*self.green, 1.),  (60,)),
                            Label(2, 'malignant',   (*self.red, 1.),    (71,72,80,90,91,92)),
                            #Label(3, 'pz',          (*self.blue, 1.),   (None,)),
                            #Label(4, 'tz',          (*self.aubergine, 1.), (None,))
                        ]

        self.class_labels = [
                    #id #name           #color              #gleason score
            Label(  0,  'bg',           (*self.gray, 0.),  (0,))]
        if "class" in self.prediction_tasks:
                self.class_labels += binary_cl_labels
                # self.class_labels += [Label(cl, cl_dic["name"], cl_dic["color"], tuple(cl_dic["gleasons"]))
                #                      for cl, cl_dic in
                #                      load_obj(os.path.join(self.data_sourcedir, "pp_class_labels.pkl")).items()]
        else:
            self.class_labels += [Label(  1,  'lesion',    (*self.red, 1.),    (60,71,72,80,90,91,92))]

        if any(['regression' in task for task in self.prediction_tasks]):
            self.bin_labels = [binLabel(0, 'bg', (*self.gray, 0.), (0,), (0,))]
            self.bin_labels += [binLabel(cl, cl_dic["name"], cl_dic["color"], tuple(cl_dic["gleasons"]),
                                         tuple([self.rg_map[gs] for gs in cl_dic["gleasons"]])) for cl, cl_dic in
                                sorted(load_obj(os.path.join(self.data_sourcedir, "pp_class_labels.pkl")).items())]
            self.bin_id2label = {label.id: label for label in self.bin_labels}
            self.gs2bin_label = {gs: label for label in self.bin_labels for gs in label.gleasons}
            bins = [(min(label.bin_vals), max(label.bin_vals)) for label in self.bin_labels]
            self.bin_id2rg_val = {ix: [np.mean(bin)] for ix, bin in enumerate(bins)}
            self.bin_edges = [(bins[i][1] + bins[i+1][0]) / 2 for i in range(len(bins)-1)]
            self.bin_dict = {label.id: label.name for label in self.bin_labels if label.id != 0}


        if self.class_specific_seg:
            self.seg_labels = self.class_labels
        else:
            self.seg_labels = [  # id      #name           #color
                Label(0, 'bg', (*self.white, 0.)),
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
        self.plot_class_ids = True

        self.num_classes = len(self.class_dict)  # for instance classification (excl background)
        self.num_seg_classes = len(self.seg_labels)  # incl background

        #########################
        #   Data Augmentation   #
        #########################
        #the angle rotations are implemented incorrectly in batchgenerators! in 2D,
        #the x-axis angle controls the z-axis angle.
        if self.dim == 2:
            angle_x = (-np.pi / 3., np.pi / 3.)
            angle_z = (0.,0.)
            rcd = (self.patch_size[0] / 2., self.patch_size[1] / 2.)
        else:
            angle_x = (0.,0.)
            angle_z = (-np.pi / 2., np.pi / 2.)
            rcd = (self.patch_size[0] / 2., self.patch_size[1] / 2.,
                   self.patch_size[2] / 2.)
        
        self.do_aug = True
        # DA settings for DWI
        self.da_kwargs = {
            'mirror': True,
            'mirror_axes': tuple(np.arange(0, self.dim, 1)),
            'random_crop': True,
            'rand_crop_dist': rcd,
            'do_elastic_deform': self.dim==2,
            'alpha': (0., 1500.),
            'sigma': (25., 50.),
            'do_rotation': True,
            'angle_x': angle_x,
            'angle_y': (0., 0.),
            'angle_z': angle_z,
            'do_scale': True,
            'scale': (0.7, 1.3),
            'border_mode_data': 'constant',
            'gamma_transform': True,
            'gamma_range': (0.5, 2.)
        }
        # for T2
        # self.da_kwargs = {
        #     'mirror': True,
        #     'mirror_axes': tuple(np.arange(0, self.dim, 1)),
        #     'random_crop': False,
        #     'rand_crop_dist': rcd,
        #     'do_elastic_deform': False,
        #     'alpha': (0., 1500.),
        #     'sigma': (25., 50.),
        #     'do_rotation': True,
        #     'angle_x': angle_x,
        #     'angle_y': (0., 0.),
        #     'angle_z': angle_z,
        #     'do_scale': False,
        #     'scale': (0.7, 1.3),
        #     'border_mode_data': 'constant',
        #     'gamma_transform': False,
        #     'gamma_range': (0.5, 2.)
        # }


        #################################
        #  Schedule / Selection / Optim #
        #################################

        # good guess: train for n_samples = 1.1m = epochs*n_train_bs*b_size
        self.num_epochs = 270
        self.num_train_batches = 120 if self.dim == 2 else 140
        
        self.val_mode = 'val_patient' # one of 'val_sampling', 'val_patient'
        # decide whether to validate on entire patient volumes (like testing) or sampled patches (like training)
        # the former is more accurate, while the latter is faster (depending on volume size)
        self.num_val_batches = 200 if self.dim==2 else 40 # for val_sampling, number or "all"
        self.max_val_patients = "all"  #for val_patient, "all" takes whole split

        self.save_n_models = 6
        self.min_save_thresh = 3 if self.dim == 2 else 4 #=wait time in epochs
        if "class" in self.prediction_tasks:
            # 'criterion': weight
            self.model_selection_criteria = {"benign_ap": 0.2, "malignant_ap": 0.8}
        elif any("regression" in task for task in self.prediction_tasks):
            self.model_selection_criteria = {"lesion_ap": 0.2, "lesion_avp": 0.8}
        #self.model_selection_criteria = {"GS71-92_ap": 0.9, "GS60_ap": 0.1}  # 'criterion':weight
        #self.model_selection_criteria = {"lesion_ap": 0.2, "lesion_avp": 0.8}
        #self.model_selection_criteria = {label.name+"_ap": 1. for label in self.class_labels if label.id!=0}

        self.scan_det_thresh = False
        self.warm_up = 0

        self.optimizer = "ADAM"
        self.weight_decay = 1e-5
        self.clip_norm = None #number or None

        self.learning_rate = [1e-4] * self.num_epochs
        self.dynamic_lr_scheduling = True
        self.lr_decay_factor = 0.5
        self.scheduling_patience = int(self.num_epochs / 6)

        #########################
        #   Testing             #
        #########################

        self.test_aug_axes = (0,1,(0,1))  # None or list: choices are 0,1,(0,1) (0==spatial y, 1== spatial x).
        self.hold_out_test_set = False
        self.max_test_patients = "all"  # "all" or number
        self.report_score_level = ['rois', 'patient']  # 'patient' or 'rois' (incl)
        self.patient_class_of_interest = 2 if 'class' in self.prediction_tasks else 1


        self.eval_bins_separately = "additionally" if not 'class' in self.prediction_tasks else False
        self.patient_bin_of_interest = 2
        self.metrics = ['ap', 'auc', 'dice']
        if any(['regression' in task for task in self.prediction_tasks]):
            self.metrics += ['avp', 'rg_MAE_weighted', 'rg_MAE_weighted_tp',
                             'rg_bin_accuracy_weighted', 'rg_bin_accuracy_weighted_tp']
        if 'aleatoric' in self.model:
            self.metrics += ['rg_uncertainty', 'rg_uncertainty_tp', 'rg_uncertainty_tp_weighted']
        self.evaluate_fold_means = True

        self.min_det_thresh = 0.02

        self.ap_match_ious = [0.1]  # threshold(s) for considering a prediction as true positive
        # aggregation method for test and val_patient predictions.
        # wbc = weighted box clustering as in https://arxiv.org/pdf/1811.08661.pdf,
        # nms = standard non-maximum suppression, or None = no clustering
        self.clustering = 'wbc'
        # iou thresh (exclusive!) for regarding two preds as concerning the same ROI
        self.clustering_iou = 0.1  # has to be larger than desired possible overlap iou of model predictions
        # 2D-3D merging is applied independently from clustering setting.
        self.merge_2D_to_3D_preds = True if self.dim == 2 else False
        self.merge_3D_iou = 0.1
        self.n_test_plots = 1  # per fold and rank
        self.test_n_epochs = self.save_n_models  # should be called n_test_ens, since is number of models to ensemble over during testing
        # is multiplied by n_test_augs if test_aug

        #########################
        # shared model settings #
        #########################

        # max number of roi candidates to identify per image and class (slice in 2D, volume in 3D)
        self.n_roi_candidates = 10 if self.dim == 2 else 15

        #########################
        #      assertions       #
        #########################
        if not 'class' in self.prediction_tasks:
            assert self.num_classes == 1
        for mod in self.prepro['modalities2concat']:
            assert mod in self.prepro['images'].keys(), "need to adapt mods2concat to chosen images"

        #########################
        #   Add model specifics #
        #########################
        
        {'mrcnn': self.add_mrcnn_configs, 'mrcnn_aleatoric': self.add_mrcnn_configs,
         'mrcnn_gan': self.add_mrcnn_configs,
         'retina_net': self.add_mrcnn_configs, 'retina_unet': self.add_mrcnn_configs,
         'detection_unet': self.add_det_unet_configs, 'detection_fpn': self.add_det_fpn_configs
         }[self.model]()

    def gleason_map(self, GS):
        """gleason to class id
        :param GS: gleason score as in histo file
        """
        if "gleason_thresh" in self.prepro.keys():
            assert "gleason_mapping" not in self.prepro.keys(), "cant define both, thresh and map, for GS to classes"
            # -1 == bg, 0 == benign, 1 == malignant
            # before shifting, i.e., 0!=bg, but 0==first class
            remapping = 0 if GS >= self.prepro["gleason_thresh"] else -1
            return remapping
        elif "gleason_mapping" in self.prepro.keys():
            return self.prepro["gleason_mapping"][GS]
        else:
            raise Exception("Need to define some remapping, at least GS 0 -> background (class -1)")

    def rg_val_to_bin_id(self, rg_val):
        return float(np.digitize(rg_val, self.bin_edges))

    def add_det_fpn_configs(self):
        self.scheduling_criterion = 'torch_loss'
        self.scheduling_mode = 'min' if "loss" in self.scheduling_criterion else 'max'

        # loss mode: either weighted cross entropy ('wce'), batch-wise dice loss ('dice), or the sum of both ('dice_wce')
        self.seg_loss_mode = 'wce'
        self.wce_weights = [1]*self.num_seg_classes if 'dice' in self.seg_loss_mode else [0.1, 1, 1]
        # if <1, false positive predictions in foreground are penalized less.
        self.fp_dice_weight = 1 if self.dim == 2 else 1


        self.detection_min_confidence = 0.05
        #how to determine score of roi: 'max' or 'median'
        self.score_det = 'max'

        self.cuda_benchmark = self.dim==3

    def add_det_unet_configs(self):
        self.scheduling_criterion = "torch_loss"
        self.scheduling_mode = 'min' if "loss" in self.scheduling_criterion else 'max'

        # loss mode: either weighted cross entropy ('wce'), batch-wise dice loss ('dice), or the sum of both ('dice_wce')
        self.seg_loss_mode = 'wce'
        self.wce_weights = [1] * self.num_seg_classes if 'dice' in self.seg_loss_mode else [0.1, 1, 1]
        # if <1, false positive predictions in foreground are penalized less.
        self.fp_dice_weight = 1 if self.dim == 2 else 1

        self.detection_min_confidence = 0.05
        #how to determine score of roi: 'max' or 'median'
        self.score_det = 'max'

        self.init_filts = 32
        self.kernel_size = 3 #ks for horizontal, normal convs
        self.kernel_size_m = 2 #ks for max pool
        self.pad = "same" # "same" or integer, padding of horizontal convs

        self.cuda_benchmark = True

    def add_mrcnn_configs(self):

        self.scheduling_criterion = max(self.model_selection_criteria, key=self.model_selection_criteria.get)
        self.scheduling_mode = 'min' if "loss" in self.scheduling_criterion else 'max'

        # number of classes for network heads: n_foreground_classes + 1 (background)
        self.head_classes = self.num_classes + 1
        #
        # feed +/- n neighbouring slices into channel dimension. set to None for no context.
        self.n_3D_context = None
        if self.n_3D_context is not None and self.dim == 2:
            self.n_channels *= (self.n_3D_context * 2 + 1)

        self.frcnn_mode = False
        # disable the re-sampling of mask proposals to original size for speed-up.
        # since evaluation is detection-driven (box-matching) and not instance segmentation-driven (iou-matching),
        # mask outputs are optional.
        self.return_masks_in_train = True
        self.return_masks_in_val = True
        self.return_masks_in_test = True

        # feature map strides per pyramid level are inferred from architecture. anchor scales are set accordingly.
        self.backbone_strides =  {'xy': [4, 8, 16, 32], 'z': [1, 2, 4, 8]}
        # anchor scales are chosen according to expected object sizes in data set. Default uses only one anchor scale
        # per pyramid level. (outer list are pyramid levels (corresponding to BACKBONE_STRIDES), inner list are scales per level.)
        self.rpn_anchor_scales = {'xy': [[4], [8], [16], [32]], 'z': [[1], [2], [4], [8]]}
        # choose which pyramid levels to extract features from: P2: 0, P3: 1, P4: 2, P5: 3.
        self.pyramid_levels = [0, 1, 2, 3]
        # number of feature maps in rpn. typically lowered in 3D to save gpu-memory.
        self.n_rpn_features = 512 if self.dim == 2 else 128

        # anchor ratios and strides per position in feature maps.
        self.rpn_anchor_ratios = [0.5,1.,2.]
        self.rpn_anchor_stride = 1
        # Threshold for first stage (RPN) non-maximum suppression (NMS):  LOWER == HARDER SELECTION
        self.rpn_nms_threshold = 0.7 if self.dim == 2 else 0.7

        # loss sampling settings.
        self.rpn_train_anchors_per_image = 6
        self.train_rois_per_image = 6 #per batch_instance
        self.roi_positive_ratio = 0.5
        self.anchor_matching_iou = 0.7

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
                               self.patch_size_3D[2], self.patch_size_3D[2]]) #y1,x1,y2,x2,z1,z2

        if self.dim == 2:
            self.rpn_bbox_std_dev = self.rpn_bbox_std_dev[:4]
            self.bbox_std_dev = self.bbox_std_dev[:4]
            self.window = self.window[:4]
            self.scale = self.scale[:4]

        self.plot_y_max = 1.5
        self.n_plot_rpn_props = 5 if self.dim == 2 else 30 #per batch_instance (slice in 2D / patient in 3D)

        # pre-selection in proposal-layer (stage 1) for NMS-speedup. applied per batch element.
        self.pre_nms_limit = 3000 if self.dim == 2 else 6000

        # n_proposals to be selected after NMS per batch element. too high numbers blow up memory if "detect_while_training" is True,
        # since proposals of the entire batch are forwarded through second stage in as one "batch".
        self.roi_chunk_size = 2000 if self.dim == 2 else 400
        self.post_nms_rois_training = 250 * (self.head_classes-1) if self.dim == 2 else 500
        self.post_nms_rois_inference = 250 * (self.head_classes-1)

        # Final selection of detections (refine_detections)
        self.model_max_instances_per_batch_element = self.n_roi_candidates  # per batch element and class.
        # iou for nms in box refining (directly after heads), should be >0 since ths>=x in mrcnn.py, otherwise all predictions are one cluster.
        self.detection_nms_threshold = 1e-5
        # detection score threshold in refine_detections()
        self.model_min_confidence = 0.05 #self.min_det_thresh/2

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

        self.operate_stride1 = False

        if self.model == 'retina_net' or self.model == 'retina_unet':
            self.cuda_benchmark = self.dim == 3
            #implement extra anchor-scales according to https://arxiv.org/abs/1708.02002
            self.rpn_anchor_scales['xy'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                            self.rpn_anchor_scales['xy']]
            self.rpn_anchor_scales['z'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                            self.rpn_anchor_scales['z']]
            self.n_anchors_per_pos = len(self.rpn_anchor_ratios) * 3

            self.n_rpn_features = 256 if self.dim == 2 else 64

            # pre-selection of detections for NMS-speedup. per entire batch.
            self.pre_nms_limit = (1000 if self.dim == 2 else 6250) * self.batch_size

            # anchor matching iou is lower than in Mask R-CNN according to https://arxiv.org/abs/1708.02002
            self.anchor_matching_iou = 0.5

            if self.model == 'retina_unet':
                self.operate_stride1 = True