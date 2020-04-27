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
sys.path.append('../') # works on cluster indep from where sbatch job is started
import plotting as plg

import numpy as np
import os
from multiprocessing import Lock
from collections import OrderedDict
import pandas as pd
import pickle
import time

# batch generator tools from https://github.com/MIC-DKFZ/batchgenerators
from batchgenerators.transforms.spatial_transforms import MirrorTransform as Mirror
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import utils.dataloader_utils as dutils
from utils.dataloader_utils import ConvertSegToBoundingBoxCoordinates


def load_obj(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

class Dataset(dutils.Dataset):
    r""" Load a dict holding memmapped arrays and clinical parameters for each patient,
    evtly subset of those.
        If server_env: copy and evtly unpack (npz->npy) data in cf.data_rootdir to
        cf.data_dir.
    :param cf: config file
    :param folds: number of folds out of @params n_cv folds to include
    :param n_cv: number of total folds
    :return: dict with imgs, segs, pids, class_labels, observables
    """

    def __init__(self, cf, logger, subset_ids=None, data_sourcedir=None, mode='train'):
        super(Dataset,self).__init__(cf, data_sourcedir=data_sourcedir)

        load_exact_gts = (mode=='test' or cf.val_mode=="val_patient") and self.cf.test_against_exact_gt

        p_df = pd.read_pickle(os.path.join(self.data_dir, cf.info_df_name))

        if subset_ids is not None:
            p_df = p_df[p_df.pid.isin(subset_ids)]
            logger.info('subset: selected {} instances from df'.format(len(p_df)))

        pids = p_df.pid.tolist()
        #evtly copy data from data_sourcedir to data_dest
        if cf.server_env and not hasattr(cf, "data_dir"):
            file_subset = [os.path.join(self.data_dir, '{}.*'.format(pid)) for pid in pids]
            file_subset += [os.path.join(self.data_dir, '{}_seg.*'.format(pid)) for pid in pids]
            file_subset += [cf.info_df_name]
            if load_exact_gts:
                file_subset += [os.path.join(self.data_dir, '{}_exact_seg.*'.format(pid)) for pid in pids]
            self.copy_data(cf, file_subset=file_subset)

        img_paths = [os.path.join(self.data_dir, '{}.npy'.format(pid)) for pid in pids]
        seg_paths = [os.path.join(self.data_dir, '{}_seg.npy'.format(pid)) for pid in pids]
        if load_exact_gts:
            exact_seg_paths = [os.path.join(self.data_dir, '{}_exact_seg.npy'.format(pid)) for pid in pids]

        class_targets = p_df['class_ids'].tolist()
        rg_targets = p_df['regression_vectors'].tolist()
        if load_exact_gts:
            exact_rg_targets = p_df['undistorted_rg_vectors'].tolist()
        fg_slices = p_df['fg_slices'].tolist()

        self.data = OrderedDict()
        for ix, pid in enumerate(pids):
            self.data[pid] = {'data': img_paths[ix], 'seg': seg_paths[ix], 'pid': pid,
                              'fg_slices': np.array(fg_slices[ix])}
            if load_exact_gts:
                self.data[pid]['exact_seg'] = exact_seg_paths[ix]
            if 'class' in self.cf.prediction_tasks:
                self.data[pid]['class_targets'] = np.array(class_targets[ix], dtype='uint8')
            else:
                self.data[pid]['class_targets'] = np.ones_like(np.array(class_targets[ix]), dtype='uint8')
            if load_exact_gts:
                self.data[pid]['exact_class_targets'] = self.data[pid]['class_targets']
            if any(['regression' in task for task in self.cf.prediction_tasks]):
                self.data[pid]['regression_targets'] = np.array(rg_targets[ix], dtype='float16')
                self.data[pid]["rg_bin_targets"] = np.array([cf.rg_val_to_bin_id(v) for v in rg_targets[ix]], dtype='uint8')
                if load_exact_gts:
                    self.data[pid]['exact_regression_targets'] = np.array(exact_rg_targets[ix], dtype='float16')
                    self.data[pid]["exact_rg_bin_targets"] = np.array([cf.rg_val_to_bin_id(v) for v in exact_rg_targets[ix]],
                                                                dtype='uint8')


        cf.roi_items = cf.observables_rois[:]
        cf.roi_items += ['class_targets']
        if any(['regression' in task for task in self.cf.prediction_tasks]):
            cf.roi_items += ['regression_targets']
            cf.roi_items += ['rg_bin_targets']

        self.set_ids = np.array(list(self.data.keys()))
        self.df = None

class BatchGenerator(dutils.BatchGenerator):
    """
    creates the training/validation batch generator. Samples n_batch_size patients (draws a slice from each patient if 2D)
    from the data set while maintaining foreground-class balance. Returned patches are cropped/padded to pre_crop_size.
    Actual patch_size is obtained after data augmentation.
    :param data: data dictionary as provided by 'load_dataset'.
    :param batch_size: number of patients to sample for the batch
    :return dictionary containing the batch data (b, c, x, y, (z)) / seg (b, 1, x, y, (z)) / pids / class_target
    """
    def __init__(self, cf, data, sample_pids_w_replace=True, max_batches=None, raise_stop_iteration=False, seed=0):
        super(BatchGenerator, self).__init__(cf, data, sample_pids_w_replace=sample_pids_w_replace,
                                             max_batches=max_batches, raise_stop_iteration=raise_stop_iteration,
                                             seed=seed)

        self.chans = cf.channels if cf.channels is not None else np.index_exp[:]
        assert hasattr(self.chans, "__iter__"), "self.chans has to be list-like to maintain dims when slicing"

        self.crop_margin = np.array(self.cf.patch_size) / 8.  # min distance of ROI center to edge of cropped_patch.
        self.p_fg = 0.5
        self.empty_samples_max_ratio = 0.6

        self.balance_target_distribution(plot=sample_pids_w_replace)

    def generate_train_batch(self):
        # everything done in here is per batch
        # print statements in here get confusing due to multithreading

        batch_pids = self.get_batch_pids()

        batch_data, batch_segs, batch_patient_targets = [], [], []
        batch_roi_items = {name: [] for name in self.cf.roi_items}
        # record roi count and empty count of classes in batch
        # empty count for no presence of resp. class in whole sample (empty slices in 2D/patients in 3D)
        batch_roi_counts = np.zeros((len(self.unique_ts),), dtype='uint32')
        batch_empty_counts = np.zeros((len(self.unique_ts),), dtype='uint32')

        for b in range(len(batch_pids)):
            patient = self._data[batch_pids[b]]

            data = np.load(patient['data'], mmap_mode='r').astype('float16')[np.newaxis]
            seg =  np.load(patient['seg'], mmap_mode='r').astype('uint8')

            (c, y, x, z) = data.shape
            if self.cf.dim == 2:
                elig_slices, choose_fg = [], False
                if len(patient['fg_slices']) > 0:
                    if np.all(batch_empty_counts / self.batch_size >= self.empty_samples_max_ratio) or np.random.rand(
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
                if len(elig_slices) > 0:
                    sl_pick_ix = np.random.choice(elig_slices, size=None)
                else:
                    sl_pick_ix = np.random.choice(z, size=None)
                data = data[..., sl_pick_ix]
                seg = seg[..., sl_pick_ix]

            spatial_shp = data[0].shape
            assert spatial_shp == seg.shape, "spatial shape incongruence betw. data and seg"
            if np.any([spatial_shp[ix] < self.cf.pre_crop_size[ix] for ix in range(len(spatial_shp))]):
                new_shape = [np.max([spatial_shp[ix], self.cf.pre_crop_size[ix]]) for ix in range(len(spatial_shp))]
                data = dutils.pad_nd_image(data, (len(data), *new_shape))
                seg = dutils.pad_nd_image(seg, new_shape)

            # eventual cropping to pre_crop_size: sample pixel from random ROI and shift center,
            # if possible, to that pixel, so that img still contains ROI after pre-cropping
            dim_cropflags = [spatial_shp[i] > self.cf.pre_crop_size[i] for i in range(len(spatial_shp))]
            if np.any(dim_cropflags):
                # sample pixel from random ROI and shift center, if possible, to that pixel
                if self.cf.dim==3:
                    choose_fg = np.any(batch_empty_counts/self.batch_size>=self.empty_samples_max_ratio) or \
                                np.random.rand(1) <= self.p_fg
                if choose_fg and np.any(seg):
                    available_roi_ids = np.unique(seg)[1:]
                    for tix in np.argsort(batch_roi_counts):
                        elig_roi_ids = available_roi_ids[patient[self.balance_target][available_roi_ids-1] == self.unique_ts[tix]]
                        if len(elig_roi_ids)>0:
                            seg_ics = np.argwhere(seg == np.random.choice(elig_roi_ids, size=None))
                            break
                    roi_anchor_pixel = seg_ics[np.random.choice(seg_ics.shape[0], size=None)]
                    assert seg[tuple(roi_anchor_pixel)] > 0

                    # sample the patch center coords. constrained by edges of image - pre_crop_size /2 and
                    # distance to the selected ROI < patch_size /2
                    def get_cropped_centercoords(dim):
                        low = np.max((self.cf.pre_crop_size[dim] // 2,
                                      roi_anchor_pixel[dim] - (
                                                  self.cf.patch_size[dim] // 2 - self.cf.crop_margin[dim])))
                        high = np.min((spatial_shp[dim] - self.cf.pre_crop_size[dim] // 2,
                                       roi_anchor_pixel[dim] + (
                                                   self.cf.patch_size[dim] // 2 - self.cf.crop_margin[dim])))
                        if low >= high:  # happens if lesion on the edge of the image.
                            low = self.cf.pre_crop_size[dim] // 2
                            high = spatial_shp[dim] - self.cf.pre_crop_size[dim] // 2

                        assert low < high, 'low greater equal high, data dimension {} too small, shp {}, patient {}, low {}, high {}'.format(
                            dim,
                            spatial_shp, patient['pid'], low, high)
                        return np.random.randint(low=low, high=high)
                else:
                    # sample crop center regardless of ROIs, not guaranteed to be empty
                    def get_cropped_centercoords(dim):
                        return np.random.randint(low=self.cf.pre_crop_size[dim] // 2,
                                                 high=spatial_shp[dim] - self.cf.pre_crop_size[dim] // 2)

                sample_seg_center = {}
                for dim in np.where(dim_cropflags)[0]:
                    sample_seg_center[dim] = get_cropped_centercoords(dim)
                    min_ = int(sample_seg_center[dim] - self.cf.pre_crop_size[dim] // 2)
                    max_ = int(sample_seg_center[dim] + self.cf.pre_crop_size[dim] // 2)
                    data = np.take(data, indices=range(min_, max_), axis=dim + 1)  # +1 for channeldim
                    seg = np.take(seg, indices=range(min_, max_), axis=dim)

            batch_data.append(data)
            batch_segs.append(seg[np.newaxis])

            for o in batch_roi_items: #after loop, holds every entry of every batchpatient per observable
                    batch_roi_items[o].append(patient[o])

            if self.cf.dim == 3:
                for tix in range(len(self.unique_ts)):
                    non_zero = np.count_nonzero(patient[self.balance_target] == self.unique_ts[tix])
                    batch_roi_counts[tix] += non_zero
                    batch_empty_counts[tix] += int(non_zero==0)
                    # todo remove assert when checked
                    if not np.any(seg):
                        assert non_zero==0
            elif self.cf.dim == 2:
                for tix in range(len(self.unique_ts)):
                    non_zero = np.count_nonzero(patient[self.balance_target][np.unique(seg[seg>0]) - 1] == self.unique_ts[tix])
                    batch_roi_counts[tix] += non_zero
                    batch_empty_counts[tix] += int(non_zero == 0)
                    # todo remove assert when checked
                    if not np.any(seg):
                        assert non_zero==0

        batch = {'data': np.array(batch_data), 'seg': np.array(batch_segs).astype('uint8'),
                 'pid': batch_pids,
                 'roi_counts': batch_roi_counts, 'empty_counts': batch_empty_counts}
        for key,val in batch_roi_items.items(): #extend batch dic by entries of observables dic
            batch[key] = np.array(val)

        return batch

class PatientBatchIterator(dutils.PatientBatchIterator):
    """
    creates a test generator that iterates over entire given dataset returning 1 patient per batch.
    Can be used for monitoring if cf.val_mode = 'patient_val' for a monitoring closer to actually evaluation (done in 3D),
    if willing to accept speed-loss during training.
    Specific properties of toy data set: toy data may be created with added ground-truth noise. thus, there are
    exact ground truths (GTs) and noisy ground truths available. the normal or noisy GTs are used in training by
    the BatchGenerator. The PatientIterator, however, may use the exact GTs if set in configs.

    :return: out_batch: dictionary containing one patient with batch_size = n_3D_patches in 3D or
    batch_size = n_2D_patches in 2D .
    """

    def __init__(self, cf, data, mode='test'):
        super(PatientBatchIterator, self).__init__(cf, data)

        self.patch_size = cf.patch_size_2D + [1] if cf.dim == 2 else cf.patch_size_3D
        self.chans = cf.channels if cf.channels is not None else np.index_exp[:]
        assert hasattr(self.chans, "__iter__"), "self.chans has to be list-like to maintain dims when slicing"

        if (mode=="validation" and hasattr(self.cf, 'val_against_exact_gt') and self.cf.val_against_exact_gt) or \
                (mode == 'test' and self.cf.test_against_exact_gt):
            self.gt_prefix = 'exact_'
            print("PatientIterator: Loading exact Ground Truths.")
        else:
            self.gt_prefix = ''

        self.patient_ix = 0  # running index over all patients in set

    def generate_train_batch(self, pid=None):

        if pid is None:
            pid = self.dataset_pids[self.patient_ix]
        patient = self._data[pid]

        # already swapped dimensions in pp from (c,)z,y,x to c,y,x,z or h,w,d to ease 2D/3D-case handling
        data = np.load(patient['data'], mmap_mode='r').astype('float16')[np.newaxis]
        seg =  np.load(patient[self.gt_prefix+'seg']).astype('uint8')[np.newaxis]

        data_shp_raw = data.shape
        plot_bg = data[self.cf.plot_bg_chan] if self.cf.plot_bg_chan not in self.chans else None
        data = data[self.chans]
        discarded_chans = len(
            [c for c in np.setdiff1d(np.arange(data_shp_raw[0]), self.chans) if c < self.cf.plot_bg_chan])
        spatial_shp = data[0].shape  # spatial dims need to be in order x,y,z
        assert spatial_shp == seg[0].shape, "spatial shape incongruence betw. data and seg"

        if np.any([spatial_shp[i] < ps for i, ps in enumerate(self.patch_size)]):
            new_shape = [np.max([spatial_shp[i], self.patch_size[i]]) for i in range(len(self.patch_size))]
            data = dutils.pad_nd_image(data, new_shape)  # use 'return_slicer' to crop image back to original shape.
            seg = dutils.pad_nd_image(seg, new_shape)
            if plot_bg is not None:
                plot_bg = dutils.pad_nd_image(plot_bg, new_shape)

        if self.cf.dim == 3 or self.cf.merge_2D_to_3D_preds:
            # adds the batch dim here bc won't go through MTaugmenter
            out_data = data[np.newaxis]
            out_seg = seg[np.newaxis]
            if plot_bg is not None:
               out_plot_bg = plot_bg[np.newaxis]
            # data and seg shape: (1,c,x,y,z), where c=1 for seg

            batch_3D = {'data': out_data, 'seg': out_seg}
            for o in self.cf.roi_items:
                batch_3D[o] = np.array([patient[self.gt_prefix+o]])
            converter = ConvertSegToBoundingBoxCoordinates(3, self.cf.roi_items, False, self.cf.class_specific_seg)
            batch_3D = converter(**batch_3D)
            batch_3D.update({'patient_bb_target': batch_3D['bb_target'], 'original_img_shape': out_data.shape})
            for o in self.cf.roi_items:
                batch_3D["patient_" + o] = batch_3D[o]

        if self.cf.dim == 2:
            out_data = np.transpose(data, axes=(3, 0, 1, 2)).astype('float32')  # (c,y,x,z) to (b=z,c,x,y), use z=b as batchdim
            out_seg = np.transpose(seg, axes=(3, 0, 1, 2)).astype('uint8')  # (c,y,x,z) to (b=z,c,x,y)

            batch_2D = {'data': out_data, 'seg': out_seg}
            for o in self.cf.roi_items:
                batch_2D[o] = np.repeat(np.array([patient[self.gt_prefix+o]]), len(out_data), axis=0)
            converter = ConvertSegToBoundingBoxCoordinates(2, self.cf.roi_items, False, self.cf.class_specific_seg)
            batch_2D = converter(**batch_2D)

            if plot_bg is not None:
                out_plot_bg = np.transpose(plot_bg, axes=(2, 0, 1)).astype('float32')

            if self.cf.merge_2D_to_3D_preds:
                batch_2D.update({'patient_bb_target': batch_3D['patient_bb_target'],
                                 'original_img_shape': out_data.shape})
                for o in self.cf.roi_items:
                    batch_2D["patient_" + o] = batch_3D[o]
            else:
                batch_2D.update({'patient_bb_target': batch_2D['bb_target'],
                                 'original_img_shape': out_data.shape})
                for o in self.cf.roi_items:
                    batch_2D["patient_" + o] = batch_2D[o]

        out_batch = batch_3D if self.cf.dim == 3 else batch_2D
        out_batch.update({'pid': np.array([patient['pid']] * len(out_data))})

        if self.cf.plot_bg_chan in self.chans and discarded_chans > 0:  # len(self.chans[:self.cf.plot_bg_chan])<data_shp_raw[0]:
            assert plot_bg is None
            plot_bg = int(self.cf.plot_bg_chan - discarded_chans)
            out_plot_bg = plot_bg
        if plot_bg is not None:
            out_batch['plot_bg'] = out_plot_bg

        # eventual tiling into patches
        spatial_shp = out_batch["data"].shape[2:]
        if np.any([spatial_shp[ix] > self.patch_size[ix] for ix in range(len(spatial_shp))]):
            patient_batch = out_batch
            print("patientiterator produced patched batch!")
            patch_crop_coords_list = dutils.get_patch_crop_coords(data[0], self.patch_size)
            new_img_batch, new_seg_batch = [], []

            for c in patch_crop_coords_list:
                new_img_batch.append(data[:, c[0]:c[1], c[2]:c[3], c[4]:c[5]])
                seg_patch = seg[:, c[0]:c[1], c[2]: c[3], c[4]:c[5]]
                new_seg_batch.append(seg_patch)
            shps = []
            for arr in new_img_batch:
                shps.append(arr.shape)

            data = np.array(new_img_batch)  # (patches, c, x, y, z)
            seg = np.array(new_seg_batch)
            if self.cf.dim == 2:
                # all patches have z dimension 1 (slices). discard dimension
                data = data[..., 0]
                seg = seg[..., 0]
            patch_batch = {'data': data.astype('float32'), 'seg': seg.astype('uint8'),
                           'pid': np.array([patient['pid']] * data.shape[0])}
            for o in self.cf.roi_items:
                patch_batch[o] = np.repeat(np.array([patient[self.gt_prefix+o]]), len(patch_crop_coords_list), axis=0)
            #patient-wise (orig) batch info for putting the patches back together after prediction
            for o in self.cf.roi_items:
                patch_batch["patient_"+o] = patient_batch["patient_"+o]
                if self.cf.dim == 2:
                    # this could also be named "unpatched_2d_roi_items"
                    patch_batch["patient_" + o + "_2d"] = patient_batch[o]
            patch_batch['patch_crop_coords'] = np.array(patch_crop_coords_list)
            patch_batch['patient_bb_target'] = patient_batch['patient_bb_target']
            if self.cf.dim == 2:
                patch_batch['patient_bb_target_2d'] = patient_batch['bb_target']
            patch_batch['patient_data'] = patient_batch['data']
            patch_batch['patient_seg'] = patient_batch['seg']
            patch_batch['original_img_shape'] = patient_batch['original_img_shape']
            if plot_bg is not None:
                patch_batch['patient_plot_bg'] = patient_batch['plot_bg']

            converter = ConvertSegToBoundingBoxCoordinates(self.cf.dim, self.cf.roi_items, get_rois_from_seg=False,
                                                           class_specific_seg=self.cf.class_specific_seg)

            patch_batch = converter(**patch_batch)
            out_batch = patch_batch

        self.patient_ix += 1
        if self.patient_ix == len(self.dataset_pids):
            self.patient_ix = 0

        return out_batch


def create_data_gen_pipeline(cf, patient_data, do_aug=True, **kwargs):
    """
    create mutli-threaded train/val/test batch generation and augmentation pipeline.
    :param patient_data: dictionary containing one dictionary per patient in the train/test subset.
    :param is_training: (optional) whether to perform data augmentation (training) or not (validation/testing)
    :return: multithreaded_generator
    """

    # create instance of batch generator as first element in pipeline.
    data_gen = BatchGenerator(cf, patient_data, **kwargs)

    my_transforms = []
    if do_aug:
        if cf.da_kwargs["mirror"]:
            mirror_transform = Mirror(axes=cf.da_kwargs['mirror_axes'])
            my_transforms.append(mirror_transform)

        spatial_transform = SpatialTransform(patch_size=cf.patch_size[:cf.dim],
                                             patch_center_dist_from_border=cf.da_kwargs['rand_crop_dist'],
                                             do_elastic_deform=cf.da_kwargs['do_elastic_deform'],
                                             alpha=cf.da_kwargs['alpha'], sigma=cf.da_kwargs['sigma'],
                                             do_rotation=cf.da_kwargs['do_rotation'], angle_x=cf.da_kwargs['angle_x'],
                                             angle_y=cf.da_kwargs['angle_y'], angle_z=cf.da_kwargs['angle_z'],
                                             do_scale=cf.da_kwargs['do_scale'], scale=cf.da_kwargs['scale'],
                                             random_crop=cf.da_kwargs['random_crop'])

        my_transforms.append(spatial_transform)
    else:
        my_transforms.append(CenterCropTransform(crop_size=cf.patch_size[:cf.dim]))

    my_transforms.append(ConvertSegToBoundingBoxCoordinates(cf.dim, cf.roi_items, False, cf.class_specific_seg))
    all_transforms = Compose(my_transforms)
    # multithreaded_generator = SingleThreadedAugmenter(data_gen, all_transforms)
    multithreaded_generator = MultiThreadedAugmenter(data_gen, all_transforms, num_processes=data_gen.n_filled_threads,
                                                     seeds=range(data_gen.n_filled_threads))
    return multithreaded_generator

def get_train_generators(cf, logger, data_statistics=False):
    """
    wrapper function for creating the training batch generator pipeline. returns the train/val generators.
    selects patients according to cv folds (generated by first run/fold of experiment):
    splits the data into n-folds, where 1 split is used for val, 1 split for testing and the rest for training. (inner loop test set)
    If cf.hold_out_test_set is True, adds the test split to the training data.
    """
    dataset = Dataset(cf, logger)
    dataset.init_FoldGenerator(cf.seed, cf.n_cv_splits)
    dataset.generate_splits(check_file=os.path.join(cf.exp_dir, 'fold_ids.pickle'))
    set_splits = dataset.fg.splits

    test_ids, val_ids = set_splits.pop(cf.fold), set_splits.pop(cf.fold - 1)
    train_ids = np.concatenate(set_splits, axis=0)

    if cf.hold_out_test_set:
        train_ids = np.concatenate((train_ids, test_ids), axis=0)
        test_ids = []

    train_data = {k: v for (k, v) in dataset.data.items() if str(k) in train_ids}
    val_data = {k: v for (k, v) in dataset.data.items() if str(k) in val_ids}

    logger.info("data set loaded with: {} train / {} val / {} test patients".format(len(train_ids), len(val_ids),
                                                                                    len(test_ids)))
    if data_statistics:
        dataset.calc_statistics(subsets={"train": train_ids, "val": val_ids, "test": test_ids}, plot_dir=
        os.path.join(cf.plot_dir,"dataset"))



    batch_gen = {}
    batch_gen['train'] = create_data_gen_pipeline(cf, train_data, do_aug=cf.do_aug, sample_pids_w_replace=True)
    if cf.val_mode == 'val_patient':
        batch_gen['val_patient'] = PatientBatchIterator(cf, val_data, mode='validation')
        batch_gen['n_val'] = len(val_ids) if cf.max_val_patients=="all" else min(len(val_ids), cf.max_val_patients)
    elif cf.val_mode == 'val_sampling':
        batch_gen['n_val'] = int(np.ceil(len(val_data)/cf.batch_size)) if cf.num_val_batches == "all" else cf.num_val_batches
        # in current setup, val loader is used like generator. with max_batches being applied in train routine.
        batch_gen['val_sampling'] = create_data_gen_pipeline(cf, val_data, do_aug=False, sample_pids_w_replace=False,
                                                             max_batches=None, raise_stop_iteration=False)

    return batch_gen

def get_test_generator(cf, logger):
    """
    if get_test_generators is possibly called multiple times in server env, every time of
    Dataset initiation rsync will check for copying the data; this should be okay
    since rsync will not copy if files already exist in destination.
    """

    if cf.hold_out_test_set:
        sourcedir = cf.test_data_sourcedir
        test_ids = None
    else:
        sourcedir = None
        with open(os.path.join(cf.exp_dir, 'fold_ids.pickle'), 'rb') as handle:
            set_splits = pickle.load(handle)
        test_ids = set_splits[cf.fold]

    test_set = Dataset(cf, logger, subset_ids=test_ids, data_sourcedir=sourcedir, mode='test')
    logger.info("data set loaded with: {} test patients".format(len(test_set.set_ids)))
    batch_gen = {}
    batch_gen['test'] = PatientBatchIterator(cf, test_set.data)
    batch_gen['n_test'] = len(test_set.set_ids) if cf.max_test_patients=="all" else \
        min(cf.max_test_patients, len(test_set.set_ids))

    return batch_gen


if __name__=="__main__":

    import utils.exp_utils as utils
    from datasets.toy.configs import Configs

    cf = Configs()

    total_stime = time.time()
    times = {}

    # cf.server_env = True
    # cf.data_dir = "experiments/dev_data"

    cf.exp_dir = "experiments/dev/"
    cf.plot_dir = cf.exp_dir + "plots"
    os.makedirs(cf.exp_dir, exist_ok=True)
    cf.fold = 0
    logger = utils.get_logger(cf.exp_dir)
    gens = get_train_generators(cf, logger)
    train_loader = gens['train']
    for i in range(0):
        stime = time.time()
        print("producing training batch nr ", i)
        ex_batch = next(train_loader)
        times["train_batch"] = time.time() - stime
        #experiments/dev/dev_exbatch_{}.png".format(i)
        plg.view_batch(cf, ex_batch, out_file="experiments/dev/dev_exbatch_{}.png".format(i), show_gt_labels=True, vmin=0, show_info=False)


    val_loader = gens['val_sampling']
    stime = time.time()
    for i in range(1):
        ex_batch = next(val_loader)
        times["val_batch"] = time.time() - stime
        stime = time.time()
        #"experiments/dev/dev_exvalbatch_{}.png"
        plg.view_batch(cf, ex_batch, out_file="experiments/dev/dev_exvalbatch_{}.png".format(i), show_gt_labels=True, vmin=0, show_info=True)
        times["val_plot"] = time.time() - stime
    #
    test_loader = get_test_generator(cf, logger)["test"]
    stime = time.time()
    ex_batch = test_loader.generate_train_batch(pid=None)
    times["test_batch"] = time.time() - stime
    stime = time.time()
    plg.view_batch(cf, ex_batch, show_gt_labels=True, out_file="experiments/dev/dev_expatchbatch.png", vmin=0)
    times["test_patchbatch_plot"] = time.time() - stime



    print("Times recorded throughout:")
    for (k, v) in times.items():
        print(k, "{:.2f}".format(v))

    mins, secs = divmod((time.time() - total_stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))