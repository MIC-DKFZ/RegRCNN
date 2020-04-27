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

'''
Data Loader for the LIDC data set. This dataloader expects preprocessed data in .npy or .npz files per patient and
a pandas dataframe containing the meta info e.g. file paths, and some ground-truth info like labels, foreground slice ids.

LIDC 4-fold annotations storage capacity problem: keep segmentation gts compressed (npz), unpack at each batch generation.

'''

import plotting as plg

import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from collections import OrderedDict

# batch generator tools from https://github.com/MIC-DKFZ/batchgenerators
from batchgenerators.transforms.spatial_transforms import MirrorTransform as Mirror
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform


import utils.dataloader_utils as dutils
from utils.dataloader_utils import ConvertSegToBoundingBoxCoordinates
from utils.dataloader_utils import BatchGenerator as BatchGeneratorParent

def save_obj(obj, name):
    """Pickle a python object."""
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

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
    :param cf: config object.
    :param logger: logger.
    :param subset_ids: subset of patient/sample identifiers to load from whole set.
    :param data_sourcedir: directory in which to find data, defaults to cf.data_sourcedir if None.
    :return: dict with imgs, segs, pids, class_labels, observables
    """

    def __init__(self, cf, logger=None, subset_ids=None, data_sourcedir=None, mode='train'):
        super(Dataset,self).__init__(cf, data_sourcedir)
        if mode == 'train' and not cf.training_gts == "merged":
            self.gt_dir = "patient_gts_sa"
            self.gt_kind = cf.training_gts
        else:
            self.gt_dir = "patient_gts_merged"
            self.gt_kind = "merged"
        if logger is not None:
            logger.info("loading {} ground truths for {}".format(self.gt_kind, 'training and validation' if mode=='train'
        else 'testing'))

        p_df = pd.read_pickle(os.path.join(self.data_sourcedir, self.gt_dir, cf.input_df_name))
        #exclude_pids = ["0305a", "0447a"]  # due to non-bg segmentation but bg mal label in nodules 5728, 8840
        #p_df = p_df[~p_df.pid.isin(exclude_pids)]

        if subset_ids is not None:
            p_df = p_df[p_df.pid.isin(subset_ids)]
            if logger is not None:
                logger.info('subset: selected {} instances from df'.format(len(p_df)))
        if cf.select_prototype_subset is not None:
            prototype_pids = p_df.pid.tolist()[:cf.select_prototype_subset]
            p_df = p_df[p_df.pid.isin(prototype_pids)]
            if logger is not None:
                logger.warning('WARNING: using prototyping data subset of length {}!!!'.format(len(p_df)))

        pids = p_df.pid.tolist()

        # evtly copy data from data_sourcedir to data_dest
        if cf.server_env and not hasattr(cf, 'data_dir') and hasattr(cf, "data_dest"):
                # copy and unpack images
                file_subset = ["{}_img.npz".format(pid) for pid in pids if not
                os.path.isfile(os.path.join(cf.data_dest,'{}_img.npy'.format(pid)))]
                file_subset += [os.path.join(self.data_sourcedir, self.gt_dir, cf.input_df_name)]
                self.copy_data(cf, file_subset=file_subset, keep_packed=False, del_after_unpack=True)
                # copy and do not unpack segmentations
                file_subset = [os.path.join(self.gt_dir, "{}_rois.np*".format(pid)) for pid in pids]
                keep_packed = not cf.training_gts == "merged"
                self.copy_data(cf, file_subset=file_subset, keep_packed=keep_packed, del_after_unpack=(not keep_packed))
        else:
            cf.data_dir = self.data_sourcedir

        ext = 'npy' if self.gt_kind == "merged" else 'npz'
        imgs = [os.path.join(self.data_dir, '{}_img.npy'.format(pid)) for pid in pids]
        segs = [os.path.join(self.data_dir, self.gt_dir, '{}_rois.{}'.format(pid, ext)) for pid in pids]
        orig_class_targets = p_df['class_target'].tolist()

        data = OrderedDict()

        if self.gt_kind == 'merged':
            for ix, pid in enumerate(pids):
                data[pid] = {'data': imgs[ix], 'seg': segs[ix], 'pid': pid}
                data[pid]['fg_slices'] = np.array(p_df['fg_slices'].tolist()[ix])
                if 'class' in cf.prediction_tasks:
                    if len(cf.class_labels)==3:
                        # malignancy scores are binarized: (benign: 1-2 --> cl 1, malignant: 3-5 --> cl 2)
                        data[pid]['class_targets'] = np.array([2 if ii >= 3 else 1 for ii in orig_class_targets[ix]],
                                                              dtype='uint8')
                    elif len(cf.class_labels)==6:
                        # classify each malignancy score
                        data[pid]['class_targets'] = np.array([1 if ii==0.5 else np.round(ii) for ii in orig_class_targets[ix]], dtype='uint8')
                    else:
                        raise Exception("mismatch class labels and data-loading implementations.")
                else:
                    data[pid]['class_targets'] = np.ones_like(np.array(orig_class_targets[ix]), dtype='uint8')
                if any(['regression' in task for task in cf.prediction_tasks]):
                    data[pid]["regression_targets"] = np.array([vector(v) for v in orig_class_targets[ix]],
                                                               dtype='float16')
                    data[pid]["rg_bin_targets"] = np.array(
                        [cf.rg_val_to_bin_id(v) for v in data[pid]["regression_targets"]], dtype='uint8')
        else:
            for ix, pid in enumerate(pids):
                data[pid] = {'data': imgs[ix], 'seg': segs[ix], 'pid': pid}
                data[pid]['fg_slices'] = np.array(p_df['fg_slices'].values[ix])
                if 'class' in cf.prediction_tasks:
                    # malignancy scores are binarized: (benign: 1-2 --> cl 1, malignant: 3-5 --> cl 2)
                    raise NotImplementedError
                    # todo need to consider bg
                    # data[pid]['class_targets'] = np.array(
                    #     [[2 if ii >= 3 else 1 for ii in four_fold_targs] for four_fold_targs in orig_class_targets[ix]])
                else:
                    data[pid]['class_targets'] = np.array(
                        [[1 if ii > 0 else 0 for ii in four_fold_targs] for four_fold_targs in orig_class_targets[ix]], dtype='uint8')
                if any(['regression' in task for task in cf.prediction_tasks]):
                    data[pid]["regression_targets"] = np.array(
                        [[vector(v) for v in four_fold_targs] for four_fold_targs in orig_class_targets[ix]], dtype='float16')
                    data[pid]["rg_bin_targets"] = np.array(
                        [[cf.rg_val_to_bin_id(v) for v in four_fold_targs] for four_fold_targs in data[pid]["regression_targets"]], dtype='uint8')

        cf.roi_items = cf.observables_rois[:]
        cf.roi_items += ['class_targets']
        if any(['regression' in task for task in cf.prediction_tasks]):
            cf.roi_items += ['regression_targets']
            cf.roi_items += ['rg_bin_targets']

        self.data = data
        self.set_ids = np.array(list(self.data.keys()))
        self.df = None

# merged GTs
class BatchGenerator_merged(dutils.BatchGenerator):
    """
    creates the training/validation batch generator. Samples n_batch_size patients (draws a slice from each patient if 2D)
    from the data set while maintaining foreground-class balance. Returned patches are cropped/padded to pre_crop_size.
    Actual patch_size is obtained after data augmentation.
    :param data: data dictionary as provided by 'load_dataset'.
    :param batch_size: number of patients to sample for the batch
    :return dictionary containing the batch data (b, c, x, y, (z)) / seg (b, 1, x, y, (z)) / pids / class_target
    """
    def __init__(self, cf, data, name="train"):
        super(BatchGenerator_merged, self).__init__(cf, data)

        self.crop_margin = np.array(self.cf.patch_size)/8. #min distance of ROI center to edge of cropped_patch.
        self.p_fg = 0.5
        self.empty_samples_max_ratio = 0.6

        self.random_count = int(cf.batch_random_ratio * cf.batch_size)
        self.class_targets = {k: v["class_targets"] for (k, v) in self._data.items()}


        self.balance_target_distribution(plot=name=="train")

    def generate_train_batch(self):

        # samples patients towards equilibrium of foreground classes on a roi-level after sampling a random ratio
        # fully random patients
        batch_patient_ids = list(np.random.choice(self.dataset_pids, size=self.random_count, replace=False))
        # target-balanced patients
        batch_patient_ids += list(np.random.choice(self.dataset_pids, size=self.batch_size-self.random_count,
                                                   replace=False, p=self.p_probs))

        batch_data, batch_segs, batch_pids, batch_patient_labels = [], [], [], []
        batch_roi_items = {name: [] for name in self.cf.roi_items}
        # record roi count of classes in batch
        batch_roi_counts = np.zeros((len(self.unique_ts),), dtype='uint32')
        batch_empty_counts = np.zeros((len(self.unique_ts),), dtype='uint32')
        # empty count for full bg samples (empty slices in 2D/patients in 3D) per class


        for sample in range(self.batch_size):
            patient = self._data[batch_patient_ids[sample]]

            data = np.transpose(np.load(patient['data'], mmap_mode='r'), axes=(1, 2, 0))[np.newaxis]
            seg = np.transpose(np.load(patient['seg'], mmap_mode='r'), axes=(1, 2, 0))
            batch_pids.append(patient['pid'])
            (c, y, x, z) = data.shape

            if self.cf.dim == 2:

                elig_slices, choose_fg = [], False
                if len(patient['fg_slices']) > 0:
                    if np.all(batch_empty_counts / self.batch_size >= self.empty_samples_max_ratio) or \
                            np.random.rand(1)<=self.p_fg:
                        # fg is to be picked
                        for tix in np.argsort(batch_roi_counts):
                            # pick slices of patient that have roi of sought-for target
                            # np.unique(seg[...,sl_ix][seg[...,sl_ix]>0]) gives roi_ids (numbering) of rois in slice sl_ix
                            elig_slices = [sl_ix for sl_ix in np.arange(z) if np.count_nonzero(
                                patient[self.balance_target][np.unique(seg[..., sl_ix][seg[..., sl_ix] > 0])-1] ==
                                self.unique_ts[tix]) > 0]
                            if len(elig_slices) > 0:
                                choose_fg = True
                                break
                    else:
                        # pick bg
                        elig_slices = np.setdiff1d(np.arange(z), patient['fg_slices'])

                if len(elig_slices)>0:
                    sl_pick_ix = np.random.choice(elig_slices, size=None)
                else:
                    sl_pick_ix = np.random.choice(z, size=None)

                data = data[..., sl_pick_ix]
                seg = seg[..., sl_pick_ix]

            # pad data if smaller than pre_crop_size.
            if np.any([data.shape[dim + 1] < ps for dim, ps in enumerate(self.cf.pre_crop_size)]):
                new_shape = [np.max([data.shape[dim + 1], ps]) for dim, ps in enumerate(self.cf.pre_crop_size)]
                data = dutils.pad_nd_image(data, new_shape, mode='constant')
                seg = dutils.pad_nd_image(seg, new_shape, mode='constant')

            # crop patches of size pre_crop_size, while sampling patches containing foreground with p_fg.
            crop_dims = [dim for dim, ps in enumerate(self.cf.pre_crop_size) if data.shape[dim + 1] > ps]
            if len(crop_dims) > 0:
                if self.cf.dim == 3:
                    choose_fg = np.all(batch_empty_counts / self.batch_size >= self.empty_samples_max_ratio)\
                                or np.random.rand(1) <= self.p_fg
                if choose_fg and np.any(seg):
                    available_roi_ids = np.unique(seg)[1:]
                    for tix in np.argsort(batch_roi_counts):
                        elig_roi_ids = available_roi_ids[patient[self.balance_target][available_roi_ids-1] == self.unique_ts[tix]]
                        if len(elig_roi_ids)>0:
                            seg_ics = np.argwhere(seg == np.random.choice(elig_roi_ids, size=None))
                            break
                    roi_anchor_pixel = seg_ics[np.random.choice(seg_ics.shape[0], size=None)]
                    assert seg[tuple(roi_anchor_pixel)] > 0
                    # sample the patch center coords. constrained by edges of images - pre_crop_size /2. And by
                    # distance to the desired ROI < patch_size /2.
                    # (here final patch size to account for center_crop after data augmentation).
                    sample_seg_center = {}
                    for ii in crop_dims:
                        low = np.max((self.cf.pre_crop_size[ii]//2, roi_anchor_pixel[ii] - (self.cf.patch_size[ii]//2 - self.crop_margin[ii])))
                        high = np.min((data.shape[ii + 1] - self.cf.pre_crop_size[ii]//2,
                                       roi_anchor_pixel[ii] + (self.cf.patch_size[ii]//2 - self.crop_margin[ii])))
                        # happens if lesion on the edge of the image. dont care about roi anymore,
                        # just make sure pre-crop is inside image.
                        if low >= high:
                            low = data.shape[ii + 1] // 2 - (data.shape[ii + 1] // 2 - self.cf.pre_crop_size[ii] // 2)
                            high = data.shape[ii + 1] // 2 + (data.shape[ii + 1] // 2 - self.cf.pre_crop_size[ii] // 2)
                        sample_seg_center[ii] = np.random.randint(low=low, high=high)

                else:
                    # not guaranteed to be empty. probability of emptiness depends on the data.
                    sample_seg_center = {ii: np.random.randint(low=self.cf.pre_crop_size[ii]//2,
                                                           high=data.shape[ii + 1] - self.cf.pre_crop_size[ii]//2) for ii in crop_dims}

                for ii in crop_dims:
                    min_crop = int(sample_seg_center[ii] - self.cf.pre_crop_size[ii] // 2)
                    max_crop = int(sample_seg_center[ii] + self.cf.pre_crop_size[ii] // 2)
                    data = np.take(data, indices=range(min_crop, max_crop), axis=ii + 1)
                    seg = np.take(seg, indices=range(min_crop, max_crop), axis=ii)

            batch_data.append(data)
            batch_segs.append(seg[np.newaxis])
            for o in batch_roi_items: #after loop, holds every entry of every batchpatient per roi-item
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


        data = np.array(batch_data).astype(np.float16)
        seg = np.array(batch_segs).astype(np.uint8)
        batch = {'data': data, 'seg': seg, 'pid': batch_pids,
                'roi_counts':batch_roi_counts, 'empty_counts': batch_empty_counts}
        for key,val in batch_roi_items.items(): #extend batch dic by roi-wise items (obs, class ids, regression vectors...)
            batch[key] = np.array(val)

        return batch

class PatientBatchIterator_merged(dutils.PatientBatchIterator):
    """
    creates a test generator that iterates over entire given dataset returning 1 patient per batch.
    Can be used for monitoring if cf.val_mode = 'patient_val' for a monitoring closer to actualy evaluation (done in 3D),
    if willing to accept speed-loss during training.
    :return: out_batch: dictionary containing one patient with batch_size = n_3D_patches in 3D or
    batch_size = n_2D_patches in 2D .
    """

    def __init__(self, cf, data):  # threads in augmenter
        super(PatientBatchIterator_merged, self).__init__(cf, data)
        self.patient_ix = 0
        self.patch_size = cf.patch_size + [1] if cf.dim == 2 else cf.patch_size

    def generate_train_batch(self, pid=None):

        if pid is None:
            pid = self.dataset_pids[self.patient_ix]
        patient = self._data[pid]

        data = np.transpose(np.load(patient['data'], mmap_mode='r'), axes=(1, 2, 0))
        seg = np.transpose(np.load(patient['seg'], mmap_mode='r'), axes=(1, 2, 0))

        # pad data if smaller than patch_size seen during training.
        if np.any([data.shape[dim] < ps for dim, ps in enumerate(self.patch_size)]):
            new_shape = [np.max([data.shape[dim], self.patch_size[dim]]) for dim, ps in enumerate(self.patch_size)]
            data = dutils.pad_nd_image(data, new_shape)  # use 'return_slicer' to crop image back to original shape.
            seg = dutils.pad_nd_image(seg, new_shape)

        # get 3D targets for evaluation, even if network operates in 2D. 2D predictions will be merged to 3D in predictor.
        if self.cf.dim == 3 or self.cf.merge_2D_to_3D_preds:
            out_data = data[np.newaxis, np.newaxis]
            out_seg = seg[np.newaxis, np.newaxis]
            batch_3D = {'data': out_data, 'seg': out_seg}
            for o in self.cf.roi_items:
                batch_3D[o] = np.array([patient[o]])
            converter = ConvertSegToBoundingBoxCoordinates(3, self.cf.roi_items, False, self.cf.class_specific_seg)
            batch_3D = converter(**batch_3D)
            batch_3D.update({'patient_bb_target': batch_3D['bb_target'], 'original_img_shape': out_data.shape})
            for o in self.cf.roi_items:
                batch_3D["patient_" + o] = batch_3D[o]

        if self.cf.dim == 2:
            out_data = np.transpose(data, axes=(2, 0, 1))[:, np.newaxis]  # (z, c, x, y )
            out_seg = np.transpose(seg, axes=(2, 0, 1))[:, np.newaxis]

            batch_2D = {'data': out_data, 'seg': out_seg}
            for o in self.cf.roi_items:
                batch_2D[o] = np.repeat(np.array([patient[o]]), out_data.shape[0], axis=0)

            converter = ConvertSegToBoundingBoxCoordinates(2, self.cf.roi_items, False, self.cf.class_specific_seg)
            batch_2D = converter(**batch_2D)

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

        # crop patient-volume to patches of patch_size used during training. stack patches up in batch dimension.
        # in this case, 2D is treated as a special case of 3D with patch_size[z] = 1.
        if np.any([data.shape[dim] > self.patch_size[dim] for dim in range(3)]):
            patient_batch = out_batch
            patch_crop_coords_list = dutils.get_patch_crop_coords(data, self.patch_size)
            new_img_batch, new_seg_batch = [], []

            for cix, c in enumerate(patch_crop_coords_list):

                seg_patch = seg[c[0]:c[1], c[2]: c[3], c[4]:c[5]]
                new_seg_batch.append(seg_patch)

                tmp_c_5 = c[5]

                new_img_batch.append(data[c[0]:c[1], c[2]:c[3], c[4]:tmp_c_5])

            data = np.array(new_img_batch)[:, np.newaxis]  # (n_patches, c, x, y, z)
            seg = np.array(new_seg_batch)[:, np.newaxis]  # (n_patches, 1, x, y, z)
            if self.cf.dim == 2:
                # all patches have z dimension 1 (slices). discard dimension
                data = data[..., 0]
                seg = seg[..., 0]

            patch_batch = {'data': data.astype('float32'), 'seg': seg.astype('uint8'),
                           'pid': np.array([patient['pid']] * data.shape[0])}
            for o in self.cf.roi_items:
                patch_batch[o] = np.repeat(np.array([patient[o]]), len(patch_crop_coords_list), axis=0)
            # patient-wise (orig) batch info for putting the patches back together after prediction
            for o in self.cf.roi_items:
                patch_batch["patient_" + o] = patient_batch['patient_' + o]
                if self.cf.dim == 2:
                    # this could also be named "unpatched_2d_roi_items"
                    patch_batch["patient_" + o + "_2d"] = patient_batch[o]
            # adding patient-wise data and seg adds about 2 GB of additional RAM consumption to a batch 20x288x288
            # and enables calculating test-dice/viewing patient-wise results in test
            # remove, but also remove dice from metrics, when like to save memory
            patch_batch['patient_data'] = patient_batch['data']
            patch_batch['patient_seg'] = patient_batch['seg']
            patch_batch['patch_crop_coords'] = np.array(patch_crop_coords_list)
            patch_batch['patient_bb_target'] = patient_batch['patient_bb_target']
            if self.cf.dim == 2:
                patch_batch['patient_bb_target_2d'] = patient_batch['bb_target']
            patch_batch['original_img_shape'] = patient_batch['original_img_shape']

            converter = ConvertSegToBoundingBoxCoordinates(self.cf.dim, self.cf.roi_items, False,
                                                           self.cf.class_specific_seg)
            patch_batch = converter(**patch_batch)
            out_batch = patch_batch

        self.patient_ix += 1
        if self.patient_ix == len(self.dataset_pids):
            self.patient_ix = 0

        return out_batch

# single-annotator GTs
class BatchGenerator_sa(BatchGeneratorParent):
    """
    creates the training/validation batch generator. Samples n_batch_size patients (draws a slice from each patient if 2D)
    from the data set while maintaining foreground-class balance. Returned patches are cropped/padded to pre_crop_size.
    Actual patch_size is obtained after data augmentation.
    :param data: data dictionary as provided by 'load_dataset'.
    :param batch_size: number of patients to sample for the batch
    :return dictionary containing the batch data (b, c, x, y, (z)) / seg (b, 1, x, y, (z)) / pids / class_target
    """

    # noinspection PyMethodOverriding
    def balance_target_distribution(self, rater, plot=False):
        """
        :param rater: for which rater slot to generate the distribution
        :param self.targets:  dic holding {patient_specifier : patient-wise-unique ROI targets}
        :param plot: whether to plot the generated patient distributions
        :return: probability distribution over all pids. draw without replace from this.
        """
        # todo limit bg weights
        unique_ts = np.unique([v[rater] for pat in self.targets.values() for v in pat])
        sample_stats = pd.DataFrame(columns=[str(ix) + suffix for ix in unique_ts for suffix in ["", "_bg"]],
                                         index=list(self.targets.keys()))
        for pid in sample_stats.index:
            for targ in unique_ts:
                fg_count = 0 if len(self.targets[pid]) == 0 else np.count_nonzero(self.targets[pid][:, rater] == targ)
                sample_stats.loc[pid, str(targ)] = int(fg_count > 0)
                sample_stats.loc[pid, str(targ) + "_bg"] = int(fg_count == 0)

        target_stats = sample_stats.agg(
            ("sum", lambda col: col.sum() / len(self._data)), axis=0, sort=False).rename({"<lambda>": "relative"})

        anchor = 1. - target_stats.loc["relative"].iloc[0]
        fg_bg_weights = anchor / target_stats.loc["relative"]
        cum_weights = anchor * len(fg_bg_weights)
        fg_bg_weights /= cum_weights

        p_probs = sample_stats.apply(self.sample_targets_to_weights, args=(fg_bg_weights,), axis=1).sum(axis=1)
        p_probs = p_probs / p_probs.sum()
        if plot:
            print("Rater: {}. Applying class-weights:\n {}".format(rater, fg_bg_weights))
        if len(sample_stats.columns) == 2:
            # assert that probs are calc'd correctly:
            # (p_probs * sample_stats["1"]).sum() == (p_probs * sample_stats["1_bg"]).sum()
            # only works if one label per patient (multi-label expectations depend on multi-label occurences).
            for rater in range(self.rater_bsize):
                expectations = []
                for targ in sample_stats.columns:
                    expectations.append((p_probs[rater] * sample_stats[targ]).sum())
                assert np.allclose(expectations, expectations[0], atol=1e-4), "expectation values for fgs/bgs: {}".format(
                    expectations)

        if plot:
            plg.plot_batchgen_distribution(self.cf, self.dataset_pids, p_probs, self.balance_target,
                                           out_file=os.path.join(self.plot_dir,
                                                                 "train_gen_distr_"+str(self.cf.fold)+"_rater"+str(rater)+".png"))
        return p_probs, unique_ts, sample_stats



    def __init__(self, cf, data, name="train"):
        super(BatchGenerator_sa, self).__init__(cf, data)
        self.name = name
        self.crop_margin = np.array(self.cf.patch_size) / 8.  # min distance of ROI center to edge of cropped_patch.
        self.p_fg = 0.5
        self.empty_samples_max_ratio = 0.6

        self.random_count = int(cf.batch_random_ratio * cf.batch_size)

        self.rater_bsize = 4
        unique_ts_total = set()
        self.p_probs = []
        self.sample_stats = []

        # todo resolve pickling error
        # p = Pool(processes=min(self.rater_bsize, cf.n_workers))
        # mp_res = p.starmap(self.balance_target_distribution, [(r, name=="train") for r in range(self.rater_bsize)])
        # p.close()
        # p.join()
        # for r, res in enumerate(mp_res):
        #     p_probs, unique_ts, sample_stats = res
        #     self.p_probs.append(p_probs)
        #     self.sample_stats.append(sample_stats)
        #     unique_ts_total.update(unique_ts)

        for r in range(self.rater_bsize):
            # todo multiprocess. takes forever
            p_probs, unique_ts, sample_stats = self.balance_target_distribution(r, plot=name == "train")
            self.p_probs.append(p_probs)
            self.sample_stats.append(sample_stats)
            unique_ts_total.update(unique_ts)

        self.unique_ts = sorted(list(unique_ts_total))
        self.stats = {"roi_counts": np.zeros(len(self.unique_ts,), dtype='uint32'),
                      "empty_counts": np.zeros(len(self.unique_ts,), dtype='uint32')}

    def generate_train_batch(self):

        rater = np.random.randint(self.rater_bsize)

        # samples patients towards equilibrium of foreground classes on a roi-level (after randomly sampling the ratio batch_random_ratio).
        # random patients
        batch_patient_ids = list(np.random.choice(self.dataset_pids, size=self.random_count, replace=False))
        # target-balanced patients
        batch_patient_ids += list(np.random.choice(self.dataset_pids, size=self.batch_size-self.random_count, replace=False,
                                             p=self.p_probs[rater]))

        batch_data, batch_segs, batch_pids, batch_patient_labels = [], [], [], []
        batch_roi_items = {name: [] for name in self.cf.roi_items}
        # record roi count of classes in batch
        batch_roi_counts = np.zeros((len(self.unique_ts),), dtype='uint32')
        batch_empty_counts = np.zeros((len(self.unique_ts),), dtype='uint32')
        # empty count for full bg samples (empty slices in 2D/patients in 3D)


        for sample in range(self.batch_size):

            patient = self._data[batch_patient_ids[sample]]

            patient_balance_ts = np.array([roi[rater] for roi in patient[self.balance_target]])
            data = np.transpose(np.load(patient['data'], mmap_mode='r'), axes=(1, 2, 0))[np.newaxis]
            seg = np.load(patient['seg'], mmap_mode='r')
            seg = np.transpose(seg[list(seg.keys())[0]][rater], axes=(1, 2, 0))
            batch_pids.append(patient['pid'])
            (c, y, x, z) = data.shape

            if self.cf.dim == 2:

                elig_slices, choose_fg = [], False
                if len(patient['fg_slices']) > 0:
                    if np.all(batch_empty_counts / self.batch_size >= self.empty_samples_max_ratio) or \
                            np.random.rand(1) <= self.p_fg:
                        # fg is to be picked
                        for tix in np.argsort(batch_roi_counts):
                            # pick slices of patient that have roi of sought-for target
                            # np.unique(seg[...,sl_ix][seg[...,sl_ix]>0]) gives roi_ids (numbering) of rois in slice sl_ix
                            elig_slices = [sl_ix for sl_ix in np.arange(z) if np.count_nonzero(
                                patient_balance_ts[np.unique(seg[..., sl_ix][seg[..., sl_ix] > 0]) - 1] ==
                                self.unique_ts[tix]) > 0]
                            if len(elig_slices) > 0:
                                choose_fg = True
                                break
                    else:
                        # pick bg
                        elig_slices = np.setdiff1d(np.arange(z), patient['fg_slices'][rater])

                if len(elig_slices) > 0:
                    sl_pick_ix = np.random.choice(elig_slices, size=None)
                else:
                    sl_pick_ix = np.random.choice(z, size=None)

                data = data[..., sl_pick_ix]
                seg = seg[..., sl_pick_ix]

            # pad data if smaller than pre_crop_size.
            if np.any([data.shape[dim + 1] < ps for dim, ps in enumerate(self.cf.pre_crop_size)]):
                new_shape = [np.max([data.shape[dim + 1], ps]) for dim, ps in enumerate(self.cf.pre_crop_size)]
                data = dutils.pad_nd_image(data, new_shape, mode='constant')
                seg = dutils.pad_nd_image(seg, new_shape, mode='constant')

            # crop patches of size pre_crop_size, while sampling patches containing foreground with p_fg.
            crop_dims = [dim for dim, ps in enumerate(self.cf.pre_crop_size) if data.shape[dim + 1] > ps]
            if len(crop_dims) > 0:
                if self.cf.dim == 3:
                    choose_fg = np.all(batch_empty_counts / self.batch_size >= self.empty_samples_max_ratio) or \
                                np.random.rand(1) <= self.p_fg
                if choose_fg and np.any(seg):
                    available_roi_ids = np.unique(seg[seg>0])
                    assert np.all(patient_balance_ts[available_roi_ids-1]>0), "trying to choose roi with rating 0"
                    for tix in np.argsort(batch_roi_counts):
                        elig_roi_ids = available_roi_ids[ patient_balance_ts[available_roi_ids-1] == self.unique_ts[tix] ]
                        if len(elig_roi_ids)>0:
                            seg_ics = np.argwhere(seg == np.random.choice(elig_roi_ids, size=None))
                            roi_anchor_pixel = seg_ics[np.random.choice(seg_ics.shape[0], size=None)]
                            break

                    assert seg[tuple(roi_anchor_pixel)] > 0, "roi_anchor_pixel not inside roi: {}, pb_ts {}, elig ids {}".format(tuple(roi_anchor_pixel), patient_balance_ts, elig_roi_ids)
                    # sample the patch center coords. constrained by edges of images - pre_crop_size /2. And by
                    # distance to the desired ROI < patch_size /2.
                    # (here final patch size to account for center_crop after data augmentation).
                    sample_seg_center = {}
                    for ii in crop_dims:
                        low = np.max((self.cf.pre_crop_size[ii]//2, roi_anchor_pixel[ii] - (self.cf.patch_size[ii]//2 - self.crop_margin[ii])))
                        high = np.min((data.shape[ii + 1] - self.cf.pre_crop_size[ii]//2,
                                       roi_anchor_pixel[ii] + (self.cf.patch_size[ii]//2 - self.crop_margin[ii])))
                        # happens if lesion on the edge of the image. dont care about roi anymore,
                        # just make sure pre-crop is inside image.
                        if low >= high:
                            low = data.shape[ii + 1] // 2 - (data.shape[ii + 1] // 2 - self.cf.pre_crop_size[ii] // 2)
                            high = data.shape[ii + 1] // 2 + (data.shape[ii + 1] // 2 - self.cf.pre_crop_size[ii] // 2)
                        sample_seg_center[ii] = np.random.randint(low=low, high=high)
                else:
                    # not guaranteed to be empty. probability of emptiness depends on the data.
                    sample_seg_center = {ii: np.random.randint(low=self.cf.pre_crop_size[ii]//2,
                                                           high=data.shape[ii + 1] - self.cf.pre_crop_size[ii]//2) for ii in crop_dims}

                for ii in crop_dims:
                    min_crop = int(sample_seg_center[ii] - self.cf.pre_crop_size[ii] // 2)
                    max_crop = int(sample_seg_center[ii] + self.cf.pre_crop_size[ii] // 2)
                    data = np.take(data, indices=range(min_crop, max_crop), axis=ii + 1)
                    seg = np.take(seg, indices=range(min_crop, max_crop), axis=ii)

            batch_data.append(data)
            batch_segs.append(seg[np.newaxis])
            for o in batch_roi_items: #after loop, holds every entry of every batchpatient per roi-item
                batch_roi_items[o].append([roi[rater] for roi in patient[o]])

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


        data = np.array(batch_data).astype('float16')
        seg = np.array(batch_segs).astype('uint8')
        batch = {'data': data, 'seg': seg, 'pid': batch_pids, 'rater_id': rater,
                'roi_counts': batch_roi_counts, 'empty_counts': batch_empty_counts}
        for key,val in batch_roi_items.items(): #extend batch dic by roi-wise items (obs, class ids, regression vectors...)
            batch[key] = np.array(val)

        return batch

class PatientBatchIterator_sa(dutils.PatientBatchIterator):
    """
    creates a test generator that iterates over entire given dataset returning 1 patient per batch.
    Can be used for monitoring if cf.val_mode = 'patient_val' for a monitoring closer to actual evaluation (done in 3D),
    if willing to accept speed loss during training.
    :return: out_batch: dictionary containing one patient with batch_size = n_3D_patches in 3D or
    batch_size = n_2D_patches in 2D .

    This is the data & gt loader for the 4-fold single-annotator GTs: each data input has separate annotations of 4 annotators.
    the way the pipeline is currently setup, the single-annotator GTs are only used if training with validation mode
    val_patient; during testing the Iterator with the merged GTs is used.
    # todo mode val_patient not implemented yet (since very slow). would need to sample from all available rater GTs.
    """
    def __init__(self, cf, data): #threads in augmenter
        super(PatientBatchIterator_sa, self).__init__(cf, data)
        self.cf = cf
        self.patient_ix = 0
        self.dataset_pids = list(self._data.keys())
        self.patch_size =  cf.patch_size+[1] if cf.dim==2 else cf.patch_size

        self.rater_bsize = 4


    def generate_train_batch(self, pid=None):

        if pid is None:
            pid = self.dataset_pids[self.patient_ix]
        patient = self._data[pid]

        data = np.transpose(np.load(patient['data'], mmap_mode='r'), axes=(1, 2, 0))
        # all gts are 4-fold and npz!
        seg = np.load(patient['seg'], mmap_mode='r')
        seg = np.transpose(seg[list(seg.keys())[0]], axes=(0, 2, 3, 1))

        # pad data if smaller than patch_size seen during training.
        if np.any([data.shape[dim] < ps for dim, ps in enumerate(self.patch_size)]):
            new_shape = [np.max([data.shape[dim], self.patch_size[dim]]) for dim, ps in enumerate(self.patch_size)]
            data = dutils.pad_nd_image(data, new_shape) # use 'return_slicer' to crop image back to original shape.
            seg = dutils.pad_nd_image(seg, new_shape)

        # get 3D targets for evaluation, even if network operates in 2D. 2D predictions will be merged to 3D in predictor.
        if self.cf.dim == 3 or self.cf.merge_2D_to_3D_preds:
            out_data = data[np.newaxis, np.newaxis]
            out_seg = seg[:, np.newaxis]
            batch_3D = {'data': out_data, 'seg': out_seg}

            for item in self.cf.roi_items:
                batch_3D[item] = []
            for r in range(self.rater_bsize):
                for item in self.cf.roi_items:
                    batch_3D[item].append(np.array([roi[r] for roi in patient[item]]))

            converter = ConvertSegToBoundingBoxCoordinates(3, self.cf.roi_items, False, self.cf.class_specific_seg)
            batch_3D = converter(**batch_3D)
            batch_3D.update({'patient_bb_target': batch_3D['bb_target'], 'original_img_shape': out_data.shape})
            for o in self.cf.roi_items:
                batch_3D["patient_" + o] = batch_3D[o]

        if self.cf.dim == 2:
            out_data = np.transpose(data, axes=(2, 0, 1))[:, np.newaxis]  # (z, c, y, x )
            out_seg = np.transpose(seg, axes=(0, 3, 1, 2))[:, :, np.newaxis] # (n_raters, z, 1, y,x)

            batch_2D = {'data': out_data}

            for item in ["seg", "bb_target"]+self.cf.roi_items:
                batch_2D[item] = []

            converter = ConvertSegToBoundingBoxCoordinates(2, self.cf.roi_items, False, self.cf.class_specific_seg)
            for r in range(self.rater_bsize):
                tmp_batch = {"seg": out_seg[r]}
                for item in self.cf.roi_items:
                    tmp_batch[item] = np.repeat(np.array([[roi[r] for roi in patient[item]]]), out_data.shape[0], axis=0)
                tmp_batch = converter(**tmp_batch)
                for item in ["seg", "bb_target"]+self.cf.roi_items:
                    batch_2D[item].append(tmp_batch[item])
            # for item in ["seg", "bb_target"]+self.cf.roi_items:
            #     batch_2D[item] = np.array(batch_2D[item])

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
        out_batch.update({'pid': np.array([patient['pid']] * out_data.shape[0])})

        # crop patient-volume to patches of patch_size used during training. stack patches up in batch dimension.
        # in this case, 2D is treated as a special case of 3D with patch_size[z] = 1.
        if np.any([data.shape[dim] > self.patch_size[dim] for dim in range(3)]):
            patient_batch = out_batch
            patch_crop_coords_list = dutils.get_patch_crop_coords(data, self.patch_size)
            new_img_batch  = []
            new_seg_batch = []

            for cix, c in enumerate(patch_crop_coords_list):
                seg_patch = seg[:, c[0]:c[1], c[2]: c[3], c[4]:c[5]]
                new_seg_batch.append(seg_patch)
                tmp_c_5 = c[5]

                new_img_batch.append(data[c[0]:c[1], c[2]:c[3], c[4]:tmp_c_5])

            data = np.array(new_img_batch)[:, np.newaxis] # (n_patches, c, x, y, z)
            seg = np.transpose(np.array(new_seg_batch), axes=(1,0,2,3,4))[:,:,np.newaxis] # (n_raters, n_patches, x, y, z)

            if self.cf.dim == 2:
                # all patches have z dimension 1 (slices). discard dimension
                data = data[..., 0]
                seg = seg[..., 0]

            patch_batch = {'data': data.astype('float32'),
                           'pid': np.array([patient['pid']] * data.shape[0])}
            # for o in self.cf.roi_items:
            #     patch_batch[o] = np.repeat(np.array([patient[o]]), len(patch_crop_coords_list), axis=0)

            converter = ConvertSegToBoundingBoxCoordinates(self.cf.dim, self.cf.roi_items, False,
                                                           self.cf.class_specific_seg)

            for item in ["seg", "bb_target"]+self.cf.roi_items:
                patch_batch[item] = []
            # coord_list = [np.min(seg_ixs[:, 1]) - 1, np.min(seg_ixs[:, 2]) - 1, np.max(seg_ixs[:, 1]) + 1,
            # IndexError: index 2 is out of bounds for axis 1 with size 2
            for r in range(self.rater_bsize):
                tmp_batch = {"seg": seg[r]}
                for item in self.cf.roi_items:
                    tmp_batch[item] = np.repeat(np.array([[roi[r] for roi in patient[item]]]), len(patch_crop_coords_list), axis=0)
                tmp_batch = converter(**tmp_batch)
                for item in ["seg", "bb_target"]+self.cf.roi_items:
                    patch_batch[item].append(tmp_batch[item])

            # patient-wise (orig) batch info for putting the patches back together after prediction
            for o in self.cf.roi_items:
                patch_batch["patient_" + o] = patient_batch['patient_'+o]
                if self.cf.dim==2:
                    # this could also be named "unpatched_2d_roi_items"
                    patch_batch["patient_"+o+"_2d"] = patient_batch[o]
            # adding patient-wise data and seg adds about 2 GB of additional RAM consumption to a batch 20x288x288
            # and enables calculating test-dice/viewing patient-wise results in test
            # remove, but also remove dice from metrics, if you like to save memory
            patch_batch['patient_data'] =  patient_batch['data']
            patch_batch['patient_seg'] = patient_batch['seg']
            patch_batch['patch_crop_coords'] = np.array(patch_crop_coords_list)
            patch_batch['patient_bb_target'] = patient_batch['patient_bb_target']
            if self.cf.dim==2:
                patch_batch['patient_bb_target_2d'] = patient_batch['bb_target']
            patch_batch['original_img_shape'] = patient_batch['original_img_shape']

            out_batch = patch_batch

        self.patient_ix += 1
        if self.patient_ix == len(self.dataset_pids):
            self.patient_ix = 0

        return out_batch


def create_data_gen_pipeline(cf, patient_data, is_training=True):
    """ create multi-threaded train/val/test batch generation and augmentation pipeline.
    :param cf: configs object.
    :param patient_data: dictionary containing one dictionary per patient in the train/test subset.
    :param is_training: (optional) whether to perform data augmentation (training) or not (validation/testing)
    :return: multithreaded_generator
    """
    BG_name = "train" if is_training else "val"
    data_gen = BatchGenerator_merged(cf, patient_data, name=BG_name) if cf.training_gts=='merged' else \
        BatchGenerator_sa(cf, patient_data, name=BG_name)

    # add transformations to pipeline.
    my_transforms = []
    if is_training:
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

    if cf.create_bounding_box_targets:
        my_transforms.append(ConvertSegToBoundingBoxCoordinates(cf.dim, cf.roi_items, False, cf.class_specific_seg))
    all_transforms = Compose(my_transforms)

    multithreaded_generator = MultiThreadedAugmenter(data_gen, all_transforms, num_processes=data_gen.n_filled_threads,
                                                     seeds=range(data_gen.n_filled_threads))
    return multithreaded_generator

def get_train_generators(cf, logger,  data_statistics=True):
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

    train_data = {k: v for (k, v) in dataset.data.items() if k in train_ids}
    val_data = {k: v for (k, v) in dataset.data.items() if k in val_ids}

    logger.info("data set loaded with: {} train / {} val / {} test patients".format(len(train_ids), len(val_ids),
                                                                                    len(test_ids)))
    if data_statistics:
        dataset.calc_statistics(subsets={"train": train_ids, "val": val_ids, "test": test_ids},
                                plot_dir=os.path.join(cf.plot_dir,"dataset"))

    batch_gen = {}
    batch_gen['train'] = create_data_gen_pipeline(cf, train_data, is_training=True)
    batch_gen['val_sampling'] = create_data_gen_pipeline(cf, val_data, is_training=False)
    if cf.val_mode == 'val_patient':
        assert cf.training_gts == 'merged', 'val_patient not yet implemented for sa gts'
        batch_gen['val_patient'] = PatientBatchIterator_merged(cf, val_data) if cf.training_gts=='merged' \
            else PatientBatchIterator_sa(cf, val_data)
        batch_gen['n_val'] = len(val_data) if cf.max_val_patients=="all" else min(len(val_data), cf.max_val_patients)
    else:
        batch_gen['n_val'] = cf.num_val_batches

    return batch_gen

def get_test_generator(cf, logger):
    """
    wrapper function for creating the test batch generator pipeline.
    selects patients according to cv folds (generated by first run/fold of experiment)
    If cf.hold_out_test_set is True, gets the data from an external folder instead.
    """
    if cf.hold_out_test_set:
        sourcedir = cf.test_data_sourcedir
        test_ids = None
    else:
        sourcedir = None
        with open(os.path.join(cf.exp_dir, 'fold_ids.pickle'), 'rb') as handle:
            set_splits = pickle.load(handle)
        test_ids = set_splits[cf.fold]

    test_data = Dataset(cf, logger, subset_ids=test_ids, data_sourcedir=sourcedir, mode="test").data
    logger.info("data set loaded with: {} test patients".format(len(test_ids)))
    batch_gen = {}
    batch_gen['test'] = PatientBatchIterator_merged(cf, test_data)
    batch_gen['n_test'] = len(test_ids) if cf.max_test_patients == "all" else min(cf.max_test_patients, len(test_ids))
    return batch_gen


if __name__ == "__main__":
    import sys
    sys.path.append('../')
    import plotting as plg
    import utils.exp_utils as utils
    from configs import Configs

    cf = Configs()
    cf.batch_size = 3
    #dataset_path = os.path.dirname(os.path.realpath(__file__))
    #exp_path = os.path.join(dataset_path, "experiments/dev")
    #cf = utils.prep_exp(dataset_path, exp_path, server_env=False, use_stored_settings=False, is_training=True)
    cf.created_fold_id_pickle = False
    total_stime = time.time()
    times = {}

    # cf.server_env = True
    # cf.data_dir = "experiments/dev_data"

    # dataset = Dataset(cf)
    # patient = dataset['Master_00018']
    cf.exp_dir = "experiments/dev/"
    cf.plot_dir = cf.exp_dir + "plots"
    os.makedirs(cf.exp_dir, exist_ok=True)
    cf.fold = 0
    logger = utils.get_logger(cf.exp_dir)
    gens = get_train_generators(cf, logger)
    train_loader = gens['train']



    for i in range(1):
        stime = time.time()
        #ex_batch = next(train_loader)
        print("train batch", i)
        times["train_batch"] = time.time() - stime
        #plg.view_batch(cf, ex_batch, out_file="experiments/dev/dev_exbatch.png", show_gt_labels=True)
    #
    # # with open(os.path.join(cf.exp_dir, "fold_"+str(cf.fold), "BatchGenerator_stats.txt"), mode="w") as file:
    # #    train_loader.generator.print_stats(logger, file)
    #
    val_loader = gens['val_sampling']
    stime = time.time()
    ex_batch = next(val_loader)
    times["val_batch"] = time.time() - stime
    stime = time.time()
    #plg.view_batch(cf, ex_batch, out_file="experiments/dev/dev_exvalbatch.png", show_gt_labels=True, plot_mods=False,
    #               show_info=False)
    times["val_plot"] = time.time() - stime
    #
    test_loader = get_test_generator(cf, logger)["test"]
    stime = time.time()
    ex_batch = test_loader.generate_train_batch()
    times["test_batch"] = time.time() - stime
    stime = time.time()
    plg.view_batch(cf, ex_batch, show_gt_labels=True, out_file="experiments/dev/dev_expatchbatch.png", get_time=False)#, sample_picks=[0,1,2,3])
    times["test_patchbatch_plot"] = time.time() - stime

    # ex_batch['data'] = ex_batch['patient_data']
    # ex_batch['seg'] = ex_batch['patient_seg']
    # ex_batch['bb_target'] = ex_batch['patient_bb_target']
    # for item in cf.roi_items:
    #     ex_batch[]
    # stime = time.time()
    # #ex_batch = next(test_loader)
    # ex_batch = next(test_loader)
    # plg.view_batch(cf, ex_batch, show_gt_labels=False, show_gt_boxes=True, patient_items=True,# vol_slice_picks=[146,148, 218,220],
    #                 out_file="experiments/dev/dev_expatientbatch.png")  # , sample_picks=[0,1,2,3])
    # times["test_patient_batch_plot"] = time.time() - stime



    print("Times recorded throughout:")
    for (k, v) in times.items():
        print(k, "{:.2f}".format(v))

    mins, secs = divmod((time.time() - total_stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))
