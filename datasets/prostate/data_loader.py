__author__ = ''
#credit derives from Paul Jaeger, Simon Kohl

import os
import time
import warnings

from collections import OrderedDict
import pickle

import numpy as np
import pandas as pd

# batch generator tools from https://github.com/MIC-DKFZ/batchgenerators
from batchgenerators.augmentations.utils import resize_image_by_padding, center_crop_2D_image, center_crop_3D_image
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.spatial_transforms import MirrorTransform as Mirror
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading import SingleThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform
#from batchgenerators.transforms.utility_transforms import ConvertSegToBoundingBoxCoordinates
from batchgenerators.transforms import AbstractTransform
from batchgenerators.transforms.color_transforms import GammaTransform

#sys.path.append(os.path.dirname(os.path.realpath(__file__)))

#import utils.exp_utils as utils
import utils.dataloader_utils as dutils
from utils.dataloader_utils import ConvertSegToBoundingBoxCoordinates
import data_manager as dmanager


def load_obj(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

def id_to_spec(id, base_spec):
    """Construct subject specifier from base string and an integer subject number."""
    num_zeros = 5 - len(str(id))
    assert num_zeros>=0, "id_to_spec: patient id too long to fit into 5 figures"
    return base_spec + '_' + ('').join(['0'] * num_zeros) + str(id)

def convert_3d_to_2d_generator(data_dict, shape="bcxyz"):
    """Fold/Shape z-dimension into color-channel.
    :param shape: bcxyz or bczyx
    :return: shape b(c*z)xy or b(c*z)yx
    """
    if shape=="bcxyz":
        data_dict['data'] = np.transpose(data_dict['data'], axes=(0,1,4,3,2))
        data_dict['seg'] = np.transpose(data_dict['seg'], axes=(0,1,4,3,2))
    elif shape=="bczyx":
        pass
    else:
        raise Exception("unknown datashape {} in 3d_to_2d transform converter".format(shape))
 
    shp = data_dict['data'].shape
    data_dict['orig_shape_data'] = shp
    seg_shp = data_dict['seg'].shape
    data_dict['orig_shape_seg'] = seg_shp
    
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['seg'] = data_dict['seg'].reshape((seg_shp[0], seg_shp[1] * seg_shp[2], seg_shp[3], seg_shp[4]))

    return data_dict

def convert_2d_to_3d_generator(data_dict, shape="bcxyz"):
    """Unfold z-dimension from color-channel.
    data needs to be in shape bcxy or bcyx, x,y dims won't be swapped relative to each other.
    :param shape: target shape, bcxyz or bczyx
    """
    shp = data_dict['orig_shape_data']
    cur_shape = data_dict['data'].shape
    seg_shp = data_dict['orig_shape_seg']
    cur_shape_seg = data_dict['seg'].shape
    
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1], shp[2], cur_shape[-2], cur_shape[-1]))
    data_dict['seg'] = data_dict['seg'].reshape((seg_shp[0], seg_shp[1], seg_shp[2], cur_shape_seg[-2], cur_shape_seg[-1]))
    
    if shape=="bcxyz":
        data_dict['data'] = np.transpose(data_dict['data'], axes=(0,1,4,3,2))
        data_dict['seg'] = np.transpose(data_dict['seg'], axes=(0,1,4,3,2)) 
    return data_dict

class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(data_dict)

class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(data_dict)

def vector(item):
    """ensure item is vector-like (list or array or tuple)
    :param item: anything
    """
    if not isinstance(item, (list, tuple, np.ndarray)):
        item = [item]
    return item

class Dataset(dutils.Dataset):
    r"""Load a dict holding memmapped arrays and clinical parameters for each patient,
    evtly subset of those.
        If server_env: copy and evtly unpack (npz->npy) data in cf.data_rootdir to
        cf.data_dest.
    :param cf: config file
    :param data_dir: directory in which to find data, defaults to cf.data_dir if None.
    :return: dict with imgs, segs, pids, class_labels, observables
    """

    def __init__(self, cf, logger=None, subset_ids=None, data_sourcedir=None):
        super(Dataset,self).__init__(cf, data_sourcedir=data_sourcedir)

        info_dict = load_obj(cf.info_dict_path)

        if subset_ids is not None:
            pids = subset_ids
            if logger is None:
                print('subset: selected {} instances from df'.format(len(pids)))
            else:
                logger.info('subset: selected {} instances from df'.format(len(pids)))
        else:
            pids = list(info_dict.keys())

        #evtly copy data from data_rootdir to data_dir
        if cf.server_env and not hasattr(cf, "data_dir"):
            file_subset = [info_dict[pid]['img'][:-3]+"*" for pid in pids]
            file_subset+= [info_dict[pid]['seg'][:-3]+"*" for pid in pids]
            file_subset += [cf.info_dict_path]
            self.copy_data(cf, file_subset=file_subset)
            cf.data_dir = self.data_dir

        img_paths = [os.path.join(self.data_dir, info_dict[pid]['img']) for pid in pids]
        seg_paths = [os.path.join(self.data_dir, info_dict[pid]['seg']) for pid in pids]

        # load all subject files
        self.data = OrderedDict()
        for i, pid in enumerate(pids):
            subj_spec = id_to_spec(pid, cf.prepro['dir_spec'])
            subj_data = {'pid':pid, "spec":subj_spec}
            subj_data['img'] = img_paths[i]
            subj_data['seg'] = seg_paths[i]
            #read, add per-roi labels
            for obs in cf.observables_patient+cf.observables_rois:
                subj_data[obs] = np.array(info_dict[pid][obs])
            if 'class' in self.cf.prediction_tasks:
                subj_data['class_targets'] = np.array(info_dict[pid]['roi_classes'], dtype='uint8') + 1
            else:
                subj_data['class_targets'] = np.ones_like(np.array(info_dict[pid]['roi_classes']), dtype='uint8')
            if any(['regression' in task for task in self.cf.prediction_tasks]):
                if hasattr(cf, "rg_map"):
                    subj_data["regression_targets"] = np.array([vector(cf.rg_map[v]) for v in info_dict[pid][cf.regression_target]], dtype='float16')
                else:
                    subj_data["regression_targets"] = np.array([vector(v) for v in info_dict[pid][cf.regression_target]], dtype='float16')
                subj_data["rg_bin_targets"] = np.array([cf.rg_val_to_bin_id(v) for v in subj_data["regression_targets"]], dtype='uint8')
            subj_data['fg_slices'] = info_dict[pid]['fg_slices']

            self.data[pid] = subj_data

        cf.roi_items = cf.observables_rois[:]
        cf.roi_items += ['class_targets']
        if any(['regression' in task for task in self.cf.prediction_tasks]):
            cf.roi_items += ['regression_targets']
            cf.roi_items += ['rg_bin_targets']
        #cf.patient_items = cf.observables_patient[:]
        #patient-wise items not used currently
        self.set_ids = np.array(list(self.data.keys()))

        self.df = None

class BatchGenerator(dutils.BatchGenerator):
    """
    create the training/validation batch generator. Randomly sample batch_size patients
    from the data set, (draw a random slice if 2D), pad-crop them to equal sizes and merge to an array.
    :param data: data dictionary as provided by 'load_dataset'
    :param img_modalities: list of strings ['adc', 'b1500'] from config
    :param batch_size: number of patients to sample for the batch
    :param pre_crop_size: equal size for merging the patients to a single array (before the final random-crop in data aug.)
    :param sample_pids_w_replace: whether to randomly draw pids from dataset for batch generation. if False, step through whole dataset
        before repition.
    :return dictionary containing the batch data / seg / pids as lists; the augmenter will later concatenate them into an array.
    """
    def __init__(self, cf, data, n_batches=None, sample_pids_w_replace=True):
        super(BatchGenerator, self).__init__(cf, data,  n_batches)
        self.dataset_length = len(self._data)
        self.cf = cf

        self.sample_pids_w_replace = sample_pids_w_replace
        self.eligible_pids = list(self._data.keys())

        self.chans = cf.channels if cf.channels is not None else np.index_exp[:]
        assert hasattr(self.chans, "__iter__"), "self.chans has to be list-like to maintain dims when slicing"

        self.p_fg = 0.5
        self.empty_samples_max_ratio = 0.6
        self.random_count = int(cf.batch_random_ratio * cf.batch_size)

        self.balance_target_distribution(plot=sample_pids_w_replace)
        self.stats = {"roi_counts" : np.zeros((len(self.unique_ts),), dtype='uint32'), "empty_samples_count" : 0}
        
    def generate_train_batch(self):
        #everything done in here is per batch
        #print statements in here get confusing due to multithreading
        if self.sample_pids_w_replace:
            # fully random patients
            batch_patient_ids = list(np.random.choice(self.dataset_pids, size=self.random_count, replace=False))
            # target-balanced patients
            batch_patient_ids += list(np.random.choice(
                self.dataset_pids, size=self.batch_size - self.random_count, replace=False, p=self.p_probs))
        else:
            batch_patient_ids = np.random.choice(self.eligible_pids, size=self.batch_size,
                                                 replace=False)
        if self.sample_pids_w_replace == False:
            self.eligible_pids = [pid for pid in self.eligible_pids if pid not in batch_patient_ids]
            if len(self.eligible_pids) < self.batch_size:
                self.eligible_pids = self.dataset_pids
        
        batch_data, batch_segs, batch_patient_specs = [], [], []
        batch_roi_items = {name: [] for name in self.cf.roi_items}
        #record roi count of classes in batch
        batch_roi_counts, empty_samples_count = np.zeros((len(self.unique_ts),), dtype='uint32'), 0
        #empty count for full bg samples (empty slices in 2D/patients in 3D)

        for sample in range(self.batch_size):

            patient = self._data[batch_patient_ids[sample]]
            
            #swap dimensions from (c,)z,y,x to (c,)y,x,z or h,w,d to ease 2D/3D-case handling
            data = np.transpose(np.load(patient['img'], mmap_mode='r'), axes=(0, 2, 3, 1))[self.chans]
            seg = np.transpose(np.load(patient['seg'], mmap_mode='r'), axes=(1, 2, 0))
            (c,y,x,z) = data.shape 

            #original data is 3D MRIs, so need to pick (e.g. randomly) single slice to make it 2D,
            #consider batch roi-class balance
            if self.cf.dim == 2:
                elig_slices, choose_fg = [], False
                if self.sample_pids_w_replace and len(patient['fg_slices']) > 0:
                    if empty_samples_count / self.batch_size >= self.empty_samples_max_ratio or np.random.rand(
                            1) <= self.p_fg:
                        # fg is to be picked
                        for tix in np.argsort(batch_roi_counts):
                            # pick slices of patient that have roi of sought-for target
                            # np.unique(seg[...,sl_ix][seg[...,sl_ix]>0]) gives roi_ids (numbering) of rois in slice sl_ix
                            elig_slices = [sl_ix for sl_ix in np.arange(z) if np.count_nonzero(
                                patient[self.balance_target][np.unique(seg[..., sl_ix][seg[..., sl_ix] > 0]) - 1] ==
                                self.unique_ts[tix]) > 0]
                            if len(elig_slices) > 0:
                                choose_fg = True
                                break
                    else:
                        # pick bg
                        elig_slices = np.setdiff1d(np.arange(z), patient['fg_slices'])
                if len(elig_slices) == 0:
                    elig_slices = z
                sl_pick_ix = np.random.choice(elig_slices, size=None)
                data = data[..., sl_pick_ix]
                seg = seg[..., sl_pick_ix]

            spatial_shp = data[0].shape
            assert spatial_shp==seg.shape, "spatial shape incongruence betw. data and seg"

            if np.any([spatial_shp[ix] < self.cf.pre_crop_size[ix] for ix in range(len(spatial_shp))]):
                new_shape = [np.max([spatial_shp[ix], self.cf.pre_crop_size[ix]]) for ix in range(len(spatial_shp))]
                data = dutils.pad_nd_image(data, (len(data), *new_shape))
                seg = dutils.pad_nd_image(seg, new_shape)
            
            #eventual cropping to pre_crop_size: with prob self.p_fg sample pixel from random ROI and shift center,
            #if possible, to that pixel, so that img still contains ROI after pre-cropping
            dim_cropflags = [spatial_shp[i] > self.cf.pre_crop_size[i] for i in range(len(spatial_shp))]
            if np.any(dim_cropflags):
                print("dim crop applied")
                # sample pixel from random ROI and shift center, if possible, to that pixel
                if self.cf.dim==3:
                    choose_fg = (empty_samples_count/self.batch_size>=self.empty_samples_max_ratio) or np.random.rand(1) <= self.p_fg
                if self.sample_pids_w_replace and choose_fg and np.any(seg):
                    available_roi_ids = np.unique(seg)[1:]
                    for tix in np.argsort(batch_roi_counts):
                        elig_roi_ids = available_roi_ids[
                            patient[self.balance_target][available_roi_ids - 1] == self.unique_ts[tix]]
                        if len(elig_roi_ids) > 0:
                            seg_ics = np.argwhere(seg == np.random.choice(elig_roi_ids, size=None))
                            break
                    roi_anchor_pixel = seg_ics[np.random.choice(seg_ics.shape[0], size=None)]
                    assert seg[tuple(roi_anchor_pixel)] > 0

                    # sample the patch center coords. constrained by edges of image - pre_crop_size /2 and 
                    # distance to the selected ROI < patch_size /2
                    def get_cropped_centercoords(dim):     
                        low = np.max((self.cf.pre_crop_size[dim]//2,
                                      roi_anchor_pixel[dim] - (self.cf.patch_size[dim]//2 - self.cf.crop_margin[dim])))
                        high = np.min((spatial_shp[dim] - self.cf.pre_crop_size[dim]//2,
                                       roi_anchor_pixel[dim] + (self.cf.patch_size[dim]//2 - self.cf.crop_margin[dim])))
                        if low >= high: #happens if lesion on the edge of the image.
                            #print('correcting low/high:', low, high, spatial_shp, roi_anchor_pixel, dim)
                            low = self.cf.pre_crop_size[dim] // 2
                            high = spatial_shp[dim] - self.cf.pre_crop_size[dim]//2
                        
                        assert low<high, 'low greater equal high, data dimension {} too small, shp {}, patient {}, low {}, high {}'.format(dim, 
                                                                         spatial_shp, patient['pid'], low, high)
                        return np.random.randint(low=low, high=high)
                else:
                    #sample crop center regardless of ROIs, not guaranteed to be empty
                    def get_cropped_centercoords(dim):                        
                        return np.random.randint(low=self.cf.pre_crop_size[dim]//2,
                                                 high=spatial_shp[dim] - self.cf.pre_crop_size[dim]//2)
                    
                sample_seg_center = {}
                for dim in np.where(dim_cropflags)[0]:
                    sample_seg_center[dim] = get_cropped_centercoords(dim)
                    min_ = int(sample_seg_center[dim] - self.cf.pre_crop_size[dim]//2)
                    max_ = int(sample_seg_center[dim] + self.cf.pre_crop_size[dim]//2)
                    data = np.take(data, indices=range(min_, max_), axis=dim+1) #+1 for channeldim
                    seg = np.take(seg, indices=range(min_, max_), axis=dim)
                    
            batch_data.append(data)
            batch_segs.append(seg[np.newaxis])
                
            for o in batch_roi_items: #after loop, holds every entry of every batchpatient per roi-item
                    batch_roi_items[o].append(patient[o])
            batch_patient_specs.append(patient['spec'])

            if self.cf.dim == 3:
                for tix in range(len(self.unique_ts)):
                    batch_roi_counts[tix] += np.count_nonzero(patient[self.balance_target] == self.unique_ts[tix])
            elif self.cf.dim == 2:
                for tix in range(len(self.unique_ts)):
                    batch_roi_counts[tix] += np.count_nonzero(patient[self.balance_target][np.unique(seg[seg>0]) - 1] == self.unique_ts[tix])
            if not np.any(seg):
                empty_samples_count += 1

        #self.stats['roi_counts'] += batch_roi_counts #DOESNT WORK WITH MULTITHREADING! do outside
        #self.stats['empty_samples_count'] += empty_samples_count

        batch = {'data': np.array(batch_data), 'seg': np.array(batch_segs).astype('uint8'),
                 'pid': batch_patient_ids, 'spec': batch_patient_specs,
                 'roi_counts':batch_roi_counts, 'empty_samples_count': empty_samples_count}
        for key,val in batch_roi_items.items(): #extend batch dic by roi-wise items (obs, class ids, regression vectors...)
            batch[key] = np.array(val)

        return batch

class PatientBatchIterator(dutils.PatientBatchIterator):
    """
    creates a val/test generator. Step through the dataset and return dictionaries per patient.
    2D is a special case of 3D patching with patch_size[2] == 1 (slices)
    Creates whole Patient batch and targets, and - if necessary - patchwise batch and targets.
    Appends patient targets anyway for evaluation.
    For Patching, shifts all patches into batch dimension. batch_tiling_forward will take care of exceeding batch dimensions.
    
    This iterator/these batches are not intended to go through MTaugmenter afterwards
    """

    def __init__(self, cf, data):
        super(PatientBatchIterator, self).__init__(cf, data)

        self.patient_ix = 0 #running index over all patients in set
        
        self.patch_size = cf.patch_size+[1] if cf.dim==2 else cf.patch_size
        self.chans = cf.channels if cf.channels is not None else np.index_exp[:]
        assert hasattr(self.chans, "__iter__"), "self.chans has to be list-like to maintain dims when slicing"

    def generate_train_batch(self, pid=None):
        
        if self.patient_ix == len(self.dataset_pids):
            self.patient_ix = 0          
        if pid is None:
            pid = self.dataset_pids[self.patient_ix] # + self.thread_id
        patient = self._data[pid]

        #swap dimensions from (c,)z,y,x to c,y,x,z or h,w,d to ease 2D/3D-case handling
        data  = np.transpose(np.load(patient['img'], mmap_mode='r'), axes=(0, 2, 3, 1))
        seg = np.transpose(np.load(patient['seg'], mmap_mode='r'), axes=(1, 2, 0))[np.newaxis]
        data_shp_raw = data.shape
        plot_bg = data[self.cf.plot_bg_chan] if self.cf.plot_bg_chan not in self.chans else None
        data = data[self.chans]
        discarded_chans = len(
            [c for c in np.setdiff1d(np.arange(data_shp_raw[0]), self.chans) if c < self.cf.plot_bg_chan])
        spatial_shp = data[0].shape # spatial dims need to be in order x,y,z
        assert spatial_shp==seg[0].shape, "spatial shape incongruence betw. data and seg"
                
        if np.any([spatial_shp[i] < ps for i, ps in enumerate(self.patch_size)]):
            new_shape = [np.max([spatial_shp[i], self.patch_size[i]]) for i in range(len(self.patch_size))]
            data = dutils.pad_nd_image(data, new_shape) # use 'return_slicer' to crop image back to original shape.
            seg = dutils.pad_nd_image(seg, new_shape)
            if plot_bg is not None:
                plot_bg = dutils.pad_nd_image(plot_bg, new_shape)

        if self.cf.dim == 3 or self.cf.merge_2D_to_3D_preds:
            #adds the batch dim here bc won't go through MTaugmenter 
            out_data = data[np.newaxis]
            out_seg = seg[np.newaxis]
            if plot_bg is not None:
                out_plot_bg = plot_bg[np.newaxis]
            #data and seg shape: (1,c,x,y,z), where c=1 for seg
            batch_3D = {'data': out_data, 'seg': out_seg}
            for o in self.cf.roi_items:
                batch_3D[o] = np.array([patient[o]])
            converter = ConvertSegToBoundingBoxCoordinates(3, self.cf.roi_items, False, self.cf.class_specific_seg)
            batch_3D = converter(**batch_3D)
            batch_3D.update({'patient_bb_target': batch_3D['bb_target'], 'original_img_shape': out_data.shape})
            for o in self.cf.roi_items:
                batch_3D["patient_" + o] = batch_3D[o]

        if self.cf.dim == 2:
            out_data = np.transpose(data, axes=(3,0,1,2)) #(c,y,x,z) to (b=z,c,x,y), use z=b as batchdim
            out_seg = np.transpose(seg, axes=(3,0,1,2)).astype('uint8')   #(c,y,x,z) to (b=z,c,x,y)

            batch_2D = {'data': out_data, 'seg': out_seg}
            for o in self.cf.roi_items:
                batch_2D[o] = np.repeat(np.array([patient[o]]), len(out_data), axis=0)

            converter = ConvertSegToBoundingBoxCoordinates(2, self.cf.roi_items, False, self.cf.class_specific_seg)
            batch_2D = converter(**batch_2D)

            if plot_bg is not None:
                out_plot_bg = np.transpose(plot_bg, axes=(2,0,1)).astype('float32')

            if self.cf.merge_2D_to_3D_preds:
                batch_2D.update({'patient_bb_target': batch_3D['patient_bb_target'],
                                      'original_img_shape': out_data.shape})
                for o in self.cf.roi_items:
                    batch_2D["patient_" + o] = batch_3D['patient_'+o]
            else:
                batch_2D.update({'patient_bb_target': batch_2D['bb_target'],
                                 'original_img_shape': out_data.shape})
                for o in self.cf.roi_items:
                    batch_2D["patient_" + o] = batch_2D[o]

        out_batch = batch_3D if self.cf.dim == 3 else batch_2D
        out_batch.update({'pid': np.array([patient['pid']] * len(out_data)),
                         'spec':np.array([patient['spec']] * len(out_data))})

        if self.cf.plot_bg_chan in self.chans and discarded_chans>0:
            assert plot_bg is None
            plot_bg = int(self.cf.plot_bg_chan - discarded_chans)
            out_plot_bg = plot_bg
        if plot_bg is not None:
            out_batch['plot_bg'] = out_plot_bg

        #eventual tiling into patches
        spatial_shp = out_batch["data"].shape[2:]
        if np.any([spatial_shp[ix] > self.patch_size[ix] for ix in range(len(spatial_shp))]):
            patient_batch = out_batch
            #print("patientiterator produced patched batch!")
            patch_crop_coords_list = dutils.get_patch_crop_coords(data[0], self.patch_size)
            new_img_batch, new_seg_batch = [], []

            for c in patch_crop_coords_list:
                new_img_batch.append(data[:, c[0]:c[1], c[2]:c[3], c[4]:c[5]])
                seg_patch = seg[:, c[0]:c[1], c[2]: c[3], c[4]:c[5]]
                new_seg_batch.append(seg_patch)
            shps = []
            for arr in new_img_batch:
                shps.append(arr.shape)
            
            data = np.array(new_img_batch) # (patches, c, x, y, z)
            seg = np.array(new_seg_batch)
            if self.cf.dim == 2:
                # all patches have z dimension 1 (slices). discard dimension
                data = data[..., 0]
                seg = seg[..., 0]
            patch_batch = {'data': data, 'seg': seg.astype('uint8'),
                                'pid': np.array([patient['pid']] * data.shape[0]),
                                'spec':np.array([patient['spec']] * data.shape[0])}
            for o in self.cf.roi_items:
                patch_batch[o] = np.repeat(np.array([patient[o]]), len(patch_crop_coords_list), axis=0)
            # patient-wise (orig) batch info for putting the patches back together after prediction
            for o in self.cf.roi_items:
                patch_batch["patient_"+o] = patient_batch['patient_'+o]
            patch_batch['patch_crop_coords'] = np.array(patch_crop_coords_list)
            patch_batch['patient_bb_target'] = patient_batch['patient_bb_target']
            #patch_batch['patient_roi_labels'] = patient_batch['patient_roi_labels']
            patch_batch['patient_data'] = patient_batch['data']
            patch_batch['patient_seg'] = patient_batch['seg']
            patch_batch['original_img_shape'] = patient_batch['original_img_shape']
            if plot_bg is not None:
                patch_batch['patient_plot_bg'] = patient_batch['plot_bg']

            converter = ConvertSegToBoundingBoxCoordinates(self.cf.dim, self.cf.roi_items, False, self.cf.class_specific_seg)
            
            patch_batch = converter(**patch_batch)
            out_batch = patch_batch
        
        self.patient_ix += 1
        # todo raise stopiteration when in test mode
        if self.patient_ix == len(self.dataset_pids):
            self.patient_ix = 0

        return out_batch


def create_data_gen_pipeline(cf, patient_data, do_aug=True, sample_pids_w_replace=True):
    """
    create mutli-threaded train/val/test batch generation and augmentation pipeline.
    :param patient_data: dictionary containing one dictionary per patient in the train/test subset
    :param test_pids: (optional) list of test patient ids, calls the test generator.
    :param do_aug: (optional) whether to perform data augmentation (training) or not (validation/testing)
    :return: multithreaded_generator
    """
    data_gen = BatchGenerator(cf, patient_data, sample_pids_w_replace=sample_pids_w_replace)
    
    my_transforms = []
    if do_aug:
        if cf.da_kwargs["mirror"]:
            mirror_transform = Mirror(axes=cf.da_kwargs['mirror_axes'])
            my_transforms.append(mirror_transform)
        if cf.da_kwargs["gamma_transform"]:
            gamma_transform = GammaTransform(gamma_range=cf.da_kwargs["gamma_range"], invert_image=False,
                                             per_channel=False, retain_stats=True)
            my_transforms.append(gamma_transform)
        if cf.dim == 3:
            # augmentations with desired effect on z-dimension
            spatial_transform = SpatialTransform(patch_size=cf.patch_size,
                                             patch_center_dist_from_border=cf.da_kwargs['rand_crop_dist'],
                                             do_elastic_deform=False,
                                             do_rotation=cf.da_kwargs['do_rotation'], angle_x=cf.da_kwargs['angle_x'],
                                             angle_y=cf.da_kwargs['angle_y'], angle_z=cf.da_kwargs['angle_z'],
                                             do_scale=cf.da_kwargs['do_scale'], scale=cf.da_kwargs['scale'],
                                             random_crop=cf.da_kwargs['random_crop'],
                                             border_mode_data=cf.da_kwargs['border_mode_data'])
            my_transforms.append(spatial_transform)
            # augmentations that are only meant to affect x-y
            my_transforms.append(Convert3DTo2DTransform())
            spatial_transform = SpatialTransform(patch_size=cf.patch_size[:2],
                                             patch_center_dist_from_border=cf.da_kwargs['rand_crop_dist'][:2],
                                             do_elastic_deform=cf.da_kwargs['do_elastic_deform'],
                                             alpha=cf.da_kwargs['alpha'], sigma=cf.da_kwargs['sigma'],
                                             do_rotation=False,
                                             do_scale=False,
                                             random_crop=False,
                                             border_mode_data=cf.da_kwargs['border_mode_data'])
            my_transforms.append(spatial_transform)
            my_transforms.append(Convert2DTo3DTransform())

        else:
            spatial_transform = SpatialTransform(patch_size=cf.patch_size[:cf.dim],
                                             patch_center_dist_from_border=cf.da_kwargs['rand_crop_dist'][:2],
                                             do_elastic_deform=cf.da_kwargs['do_elastic_deform'],
                                             alpha=cf.da_kwargs['alpha'], sigma=cf.da_kwargs['sigma'],
                                             do_rotation=cf.da_kwargs['do_rotation'], angle_x=cf.da_kwargs['angle_x'],
                                             angle_y=cf.da_kwargs['angle_y'], angle_z=cf.da_kwargs['angle_z'],
                                             do_scale=cf.da_kwargs['do_scale'], scale=cf.da_kwargs['scale'],
                                             random_crop=cf.da_kwargs['random_crop'],
                                             border_mode_data=cf.da_kwargs['border_mode_data'])
            my_transforms.append(spatial_transform)
    else:
        my_transforms.append(CenterCropTransform(crop_size=cf.patch_size[:cf.dim]))

    if cf.create_bounding_box_targets:
        my_transforms.append(ConvertSegToBoundingBoxCoordinates(cf.dim, cf.roi_items, False, cf.class_specific_seg))
        #batch receives entry 'bb_target' w bbox coordinates as [y1,x1,y2,x2,z1,z2].
    #my_transforms.append(ConvertSegToOnehotTransform(classes=range(cf.num_seg_classes)))
    all_transforms = Compose(my_transforms)
    #MTAugmenter creates iterator from data iterator data_gen after applying the composed transform all_transforms
    multithreaded_generator = MultiThreadedAugmenter(data_gen, all_transforms, num_processes=data_gen.n_filled_threads,
                                                     seeds=range(data_gen.n_filled_threads))
    return multithreaded_generator

def get_train_generators(cf, logger, data_statistics=True):
    """
    wrapper function for creating the training batch generator pipeline. returns the train/val generators
    need to select cv folds on patient level, but be able to include both breasts of each patient.
    """
    dataset = Dataset(cf, logger)

    dataset.init_FoldGenerator(cf.seed, cf.n_cv_splits)
    dataset.generate_splits(check_file=os.path.join(cf.exp_dir, 'fold_ids.pickle'))
    set_splits = dataset.fg.splits

    test_ids, val_ids = set_splits.pop(cf.fold), set_splits.pop(cf.fold-1)
    train_ids = np.concatenate(set_splits, axis=0)

    if cf.held_out_test_set:
        train_ids = np.concatenate((train_ids, test_ids), axis=0)
        test_ids = []

    train_data = {k: v for (k, v) in dataset.data.items() if k in train_ids}
    val_data = {k: v for (k, v) in dataset.data.items() if k in val_ids}
    
    logger.info("data set loaded with: {} train / {} val / {} test patients".format(len(train_ids), len(val_ids), len(test_ids)))
    if data_statistics:
        dataset.calc_statistics(subsets={"train":train_ids, "val":val_ids, "test":test_ids},
                                plot_dir=os.path.join(cf.plot_dir,"dataset"))
        
    batch_gen = {}
    batch_gen['train'] = create_data_gen_pipeline(cf, train_data, do_aug=cf.do_aug)
    batch_gen['val_sampling'] = create_data_gen_pipeline(cf, val_data, do_aug=False, sample_pids_w_replace=False)

    if cf.val_mode == 'val_patient':
        batch_gen['val_patient'] = PatientBatchIterator(cf, val_data)
        batch_gen['n_val'] = len(val_ids) if cf.max_val_patients=="all" else cf.max_val_patients
    elif cf.val_mode == 'val_sampling':
        batch_gen['n_val'] = cf.num_val_batches if cf.num_val_batches!="all" else len(val_ids)
    
    return batch_gen

def get_test_generator(cf, logger):
    """
    if get_test_generators is called multiple times in server env, every time of 
    Dataset initiation rsync will check for copying the data; this should be okay
    since rsync will not copy if files already exist in destination.
    """

    if cf.held_out_test_set:
        sourcedir = cf.test_data_sourcedir
        test_ids = None
    else:
        sourcedir = None
        with open(os.path.join(cf.exp_dir, 'fold_ids.pickle'), 'rb') as handle:
            set_splits = pickle.load(handle)
        test_ids = set_splits[cf.fold]

    test_set = Dataset(cf, logger, test_ids, data_sourcedir=sourcedir)
    logger.info("data set loaded with: {} test patients".format(len(test_set.set_ids)))
    batch_gen = {}
    batch_gen['test'] = PatientBatchIterator(cf, test_set.data)
    batch_gen['n_test'] = len(test_set.set_ids) if cf.max_test_patients=="all" else min(cf.max_test_patients, len(test_set.set_ids))
    
    return batch_gen


if __name__=="__main__":
    import sys
    sys.path.append('../')  # works on cluster indep from where sbatch job is started
    import plotting as plg
    import utils.exp_utils as utils
    from configs import Configs
    cf = configs()
    
    total_stime = time.time()
    times = {}

    #cf.server_env = True
    #cf.data_dir = "experiments/dev_data"
    
    #dataset = Dataset(cf)
    #patient = dataset['Master_00018'] 
    cf.exp_dir = "experiments/dev/"
    cf.plot_dir = cf.exp_dir+"plots"
    os.makedirs(cf.exp_dir, exist_ok=True)
    cf.fold = 0
    logger = utils.get_logger(cf.exp_dir)
    gens = get_train_generators(cf, logger)
    train_loader = gens['train']
    
    #for i in range(train_loader.dataset_length):
    #    print("batch", i)
    stime = time.time()
    ex_batch = next(train_loader)
    #ex_batch = next(train_loader)
    times["train_batch"] = time.time()-stime
    plg.view_batch(cf, ex_batch, out_file="experiments/dev/dev_exbatch.png", show_gt_labels=True)

    #with open(os.path.join(cf.exp_dir, "fold_"+str(cf.fold), "BatchGenerator_stats.txt"), mode="w") as file:
    #    train_loader.generator.print_stats(logger, file)


    val_loader = gens['val_sampling']
    stime = time.time()
    ex_batch = next(val_loader)
    times["val_batch"] = time.time()-stime
    stime = time.time()
    plg.view_batch(cf, ex_batch, out_file="experiments/dev/dev_exvalbatch.png", show_gt_labels=True, plot_mods=False, show_info=False)
    times["val_plot"] = time.time()-stime
    
    test_loader = get_test_generator(cf, logger)["test"]
    stime = time.time()
    ex_batch = test_loader.generate_train_batch()
    print(ex_batch["data"].shape)
    times["test_batch"] = time.time()-stime
    stime = time.time()
    plg.view_batch(cf, ex_batch, show_gt_labels=True, out_file="experiments/dev/ex_patchbatch.png", show_gt_boxes=False, show_info=False, dpi=400, sample_picks=[2,5], plot_mods=False)
    times["test_patchbatch_plot"] = time.time()-stime

    #stime = time.time()
    #ex_batch['data'] = ex_batch['patient_data']
    #ex_batch['seg'] = ex_batch['patient_seg']
    #if 'patient_plot_bg' in ex_batch.keys():
    #    ex_batch['plot_bg'] = ex_batch['patient_plot_bg']
    #plg.view_batch(cf, ex_batch, show_gt_labels=True, out_file="experiments/dev/dev_expatchbatch.png")
    #times["test_patientbatch_plot"] = time.time() - stime
    
    
    #print("patch batch keys", ex_batch.keys())
    #print("patch batch les gle", ex_batch["lesion_gleasons"].shape)
    #print("patch batch gsbx", ex_batch["GSBx"].shape)
    #print("patch batch class_targ", ex_batch["class_targets"].shape)
    #print("patient b roi labels", ex_batch["patient_roi_labels"].shape)
    #print("patient les gleas", ex_batch["patient_lesion_gleasons"].shape)
    #print("patch&patient batch pid", ex_batch["pid"], len(ex_batch["pid"]))
    #print("unique patient_seg", np.unique(ex_batch["patient_seg"]))
    #print("pb patient roi labels", len(ex_batch["patient_roi_labels"]), ex_batch["patient_roi_labels"])
    #print("pid", ex_batch["pid"])
    
    #patient_batch = {k[len("patient_"):]:v for (k,v) in ex_batch.items() if k.lower().startswith("patient")}
    #patient_batch["pid"] = ex_batch["pid"]
    #stime = time.time()
    #plg.view_batch(cf, patient_batch, out_file="experiments/dev_expatientbatch")
    #times["test_plot"] = time.time()-stime
    
    
    print("Times recorded throughout:")
    for (k,v) in times.items():
        print(k, "{:.2f}".format(v))
    
    mins, secs = divmod((time.time() - total_stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs)) 
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))