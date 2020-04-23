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
            self.copy_data(cf, file_subset=file_subset)

        img_paths = [os.path.join(self.data_dir, '{}.npy'.format(pid)) for pid in pids]
        seg_paths = [os.path.join(self.data_dir, '{}.npy'.format(pid)) for pid in pids]

        class_targets = p_df['class_id'].tolist()

        self.data = OrderedDict()
        for ix, pid in enumerate(pids):
            self.data[pid] = {'data': img_paths[ix], 'seg': seg_paths[ix], 'pid': pid}
            self.data[pid]['class_targets'] = np.array([class_targets[ix]], dtype='uint8') + 1

        cf.roi_items = ['class_targets']

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

            all_data = np.load(patient['data'], mmap_mode='r')
            data = all_data[0].astype('float16')[np.newaxis]
            seg = all_data[1].astype('uint8')

            spatial_shp = data[0].shape
            assert spatial_shp == seg.shape, "spatial shape incongruence betw. data and seg"
            if np.any([spatial_shp[ix] < self.cf.pre_crop_size[ix] for ix in range(len(spatial_shp))]):
                new_shape = [np.max([spatial_shp[ix], self.cf.pre_crop_size[ix]]) for ix in range(len(spatial_shp))]
                data = dutils.pad_nd_image(data, (len(data), *new_shape))
                seg = dutils.pad_nd_image(seg, new_shape)

            batch_data.append(data)
            batch_segs.append(seg[np.newaxis])

            for o in batch_roi_items: #after loop, holds every entry of every batchpatient per observable
                    batch_roi_items[o].append(patient[o])

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

        self.patient_ix = 0  # running index over all patients in set

    def generate_train_batch(self, pid=None):

        if pid is None:
            pid = self.dataset_pids[self.patient_ix]
        patient = self._data[pid]

        # already swapped dimensions in pp from (c,)z,y,x to c,y,x,z or h,w,d to ease 2D/3D-case handling
        all_data = np.load(patient['data'], mmap_mode='r')
        data = all_data[0].astype('float16')[np.newaxis]
        seg = all_data[1].astype('uint8')[np.newaxis]

        data_shp_raw = data.shape
        data = data[self.chans]
        spatial_shp = data[0].shape  # spatial dims need to be in order x,y,z
        assert spatial_shp == seg[0].shape, "spatial shape incongruence betw. data and seg"

        out_data = data[None]
        out_seg = seg[None]

        batch_2D = {'data': out_data, 'seg': out_seg}
        for o in self.cf.roi_items:
            batch_2D[o] = np.repeat(np.array([patient[o]]), len(out_data), axis=0)
        converter = ConvertSegToBoundingBoxCoordinates(2, self.cf.roi_items, False, self.cf.class_specific_seg)
        batch_2D = converter(**batch_2D)

        batch_2D.update({'patient_bb_target': batch_2D['bb_target'],
                         'original_img_shape': out_data.shape})
        for o in self.cf.roi_items:
            batch_2D["patient_" + o] = batch_2D[o]

        out_batch = batch_2D
        out_batch.update({'pid': np.array([patient['pid']] * len(out_data))})

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

    assert cf.n_train_val_data <= len(dataset.set_ids), \
        "requested {} train val samples, but dataset only has {} train val samples.".format(
            cf.n_train_val_data, len(dataset.set_ids))
    train_ids = dataset.set_ids[:int(2*cf.n_train_val_data//3)]
    val_ids = dataset.set_ids[int(np.ceil(2*cf.n_train_val_data//3)):cf.n_train_val_data]

    train_data = {k: v for (k, v) in dataset.data.items() if str(k) in train_ids}
    val_data = {k: v for (k, v) in dataset.data.items() if str(k) in val_ids}

    logger.info("data set loaded with: {} train / {} val patients".format(len(train_ids), len(val_ids)))
    if data_statistics:
        dataset.calc_statistics(subsets={"train": train_ids, "val": val_ids}, plot_dir=
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