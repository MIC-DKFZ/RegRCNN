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
import plotting as plg

import os
from multiprocessing import Pool, Lock
import pickle
import warnings

import numpy as np
import pandas as pd
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from scipy.ndimage.measurements import label as lb
from torch.utils.data import Dataset as torchDataset
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

import utils.exp_utils as utils
import data_manager as dmanager


for msg in ["This figure includes Axes that are not compatible with tight_layout",
            "Data has no positive values, and therefore cannot be log-scaled."]:
    warnings.filterwarnings("ignore", msg)


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

##################################
#  data loading, organisation  #
##################################


class fold_generator:
    """
    generates splits of indices for a given length of a dataset to perform n-fold cross-validation.
    splits each fold into 3 subsets for training, validation and testing.
    This form of cross validation uses an inner loop test set, which is useful if test scores shall be reported on a
    statistically reliable amount of patients, despite limited size of a dataset.
    If hold out test set is provided and hence no inner loop test set needed, just add test_idxs to the training data in the dataloader.
    This creates straight-forward train-val splits.
    :returns names list: list of len n_splits. each element is a list of len 3 for train_ix, val_ix, test_ix.
    """
    def __init__(self, seed, n_splits, len_data):
        """
        :param seed: Random seed for splits.
        :param n_splits: number of splits, e.g. 5 splits for 5-fold cross-validation
        :param len_data: number of elements in the dataset.
        """
        self.tr_ix = []
        self.val_ix = []
        self.te_ix = []
        self.slicer = None
        self.missing = 0
        self.fold = 0
        self.len_data = len_data
        self.n_splits = n_splits
        self.myseed = seed
        self.boost_val = 0

    def init_indices(self):

        t = list(np.arange(self.l))
        # round up to next splittable data amount.
        split_length = int(np.ceil(len(t) / float(self.n_splits)))
        self.slicer = split_length
        self.mod = len(t) % self.n_splits
        if self.mod > 0:
            # missing is the number of folds, in which the new splits are reduced to account for missing data.
            self.missing = self.n_splits - self.mod

        self.te_ix = t[:self.slicer]
        self.tr_ix = t[self.slicer:]
        self.val_ix = self.tr_ix[:self.slicer]
        self.tr_ix = self.tr_ix[self.slicer:]

    def new_fold(self):

        slicer = self.slicer
        if self.fold < self.missing :
            slicer = self.slicer - 1

        temp = self.te_ix

        # catch exception mod == 1: test set collects 1+ data since walk through both roudned up splits.
        # account for by reducing last fold split by 1.
        if self.fold == self.n_splits-2 and self.mod ==1:
            temp += self.val_ix[-1:]
            self.val_ix = self.val_ix[:-1]

        self.te_ix = self.val_ix
        self.val_ix = self.tr_ix[:slicer]
        self.tr_ix = self.tr_ix[slicer:] + temp


    def get_fold_names(self):
        names_list = []
        rgen = np.random.RandomState(self.myseed)
        cv_names = np.arange(self.len_data)

        rgen.shuffle(cv_names)
        self.l = len(cv_names)
        self.init_indices()

        for split in range(self.n_splits):
            train_names, val_names, test_names = cv_names[self.tr_ix], cv_names[self.val_ix], cv_names[self.te_ix]
            names_list.append([train_names, val_names, test_names, self.fold])
            self.new_fold()
            self.fold += 1

        return names_list



class FoldGenerator():
    r"""takes a set of elements (identifiers) and randomly splits them into the specified amt of subsets.
    """

    def __init__(self, identifiers, seed, n_splits=5):
        self.ids = np.array(identifiers)
        self.n_splits = n_splits
        self.seed = seed

    def generate_splits(self, n_splits=None):
        if n_splits is None:
            n_splits = self.n_splits

        rgen = np.random.RandomState(self.seed)
        rgen.shuffle(self.ids)
        self.splits = list(np.array_split(self.ids, n_splits, axis=0))  # already returns list, but to be sure
        return self.splits


class Dataset(torchDataset):
    r"""Parent Class for actual Dataset classes to inherit from!
    """
    def __init__(self, cf, data_sourcedir=None):
        super(Dataset, self).__init__()
        self.cf = cf

        self.data_sourcedir = cf.data_sourcedir if data_sourcedir is None else data_sourcedir
        self.data_dir = cf.data_dir if hasattr(cf, 'data_dir') else self.data_sourcedir

        self.data_dest = cf.data_dest if hasattr(cf, "data_dest") else self.data_sourcedir

        self.data = {}
        self.set_ids = []

    def copy_data(self, cf, file_subset, keep_packed=False, del_after_unpack=False):
        if os.path.normpath(self.data_sourcedir) != os.path.normpath(self.data_dest):
            self.data_sourcedir = os.path.join(self.data_sourcedir, '')
            args = AttributeDict({
                    "source" :  self.data_sourcedir,
                    "destination" : self.data_dest,
                    "recursive" : True,
                    "cp_only_npz" : False,
                    "keep_packed" : keep_packed,
                    "del_after_unpack" : del_after_unpack,
                    "threads" : 16 if self.cf.server_env else os.cpu_count()
                    })
            dmanager.copy(args, file_subset=file_subset)
            self.data_dir = self.data_dest



    def __len__(self):
        return len(self.data)
    def __getitem__(self, id):
        """Return a sample of the dataset, i.e.,the dict of the id
        """
        return self.data[id]
    def __iter__(self):
        return self.data.__iter__()

    def init_FoldGenerator(self, seed, n_splits):
        self.fg = FoldGenerator(self.set_ids, seed=seed, n_splits=n_splits)

    def generate_splits(self, check_file):
        if not os.path.exists(check_file):
            self.fg.generate_splits()
            with open(check_file, 'wb') as handle:
                pickle.dump(self.fg.splits, handle)
        else:
            with open(check_file, 'rb') as handle:
                self.fg.splits = pickle.load(handle)

    def calc_statistics(self, subsets=None, plot_dir=None, overall_stats=True):

        if self.df is None:
            self.df = pd.DataFrame()
            balance_t = self.cf.balance_target if hasattr(self.cf, "balance_target") else "class_targets"
            self.df._metadata.append(balance_t)
            if balance_t=="class_targets":
                mapper = lambda cl_id: self.cf.class_id2label[cl_id]
                labels = self.cf.class_id2label.values()
            elif balance_t=="rg_bin_targets":
                mapper = lambda rg_bin: self.cf.bin_id2label[rg_bin]
                labels = self.cf.bin_id2label.values()
            # elif balance_t=="regression_targets":
            #     # todo this wont work
            #     mapper = lambda rg_val: AttributeDict({"name":rg_val}) #self.cf.bin_id2label[self.cf.rg_val_to_bin_id(rg_val)]
            #     labels = self.cf.bin_id2label.values()
            elif balance_t=="lesion_gleasons":
                mapper = lambda gs: self.cf.gs2label[gs]
                labels = self.cf.gs2label.values()
            else:
                mapper = lambda x: AttributeDict({"name":x})
                labels = None
            for pid, subj_data in self.data.items():
                unique_ts, counts = np.unique(subj_data[balance_t], return_counts=True)
                self.df = self.df.append(pd.DataFrame({"pid": [pid],
                                                       **{mapper(unique_ts[i]).name: [counts[i]] for i in
                                                          range(len(unique_ts))}}), ignore_index=True, sort=True)
            self.df = self.df.fillna(0)

        if overall_stats:
            df = self.df.drop("pid", axis=1)
            df = df.reindex(sorted(df.columns), axis=1).astype('uint32')
            print("Overall dataset roi counts per target kind:"); print(df.sum())
        if subsets is not None:
            self.df["subset"] = np.nan
            self.df["display_order"] = np.nan
            for ix, (subset, pids) in enumerate(subsets.items()):
                self.df.loc[self.df.pid.isin(pids), "subset"] = subset
                self.df.loc[self.df.pid.isin(pids), "display_order"] = ix
            df = self.df.groupby("subset").agg("sum").drop("pid", axis=1, errors='ignore').astype('int64')
            df = df.sort_values(by=['display_order']).drop('display_order', axis=1)
            df = df.reindex(sorted(df.columns), axis=1)

            print("Fold {} dataset roi counts per target kind:".format(self.cf.fold)); print(df)
        if plot_dir is not None:
            os.makedirs(plot_dir, exist_ok=True)
            if subsets is not None:
                plg.plot_fold_stats(self.cf, df, labels, os.path.join(plot_dir, "data_stats_fold_" + str(self.cf.fold))+".pdf")
            if overall_stats:
                plg.plot_data_stats(self.cf, df, labels, os.path.join(plot_dir, 'data_stats_overall.pdf'))

        return df, labels


def get_class_balanced_patients(all_pids, class_targets, batch_size, num_classes, random_ratio=0):
    '''
    samples towards equilibrium of classes (on basis of total RoI counts). for highly imbalanced dataset, this might be a too strong requirement.
    :param class_targets: dic holding {patient_specifier : ROI class targets}, list position of ROI target corresponds to respective seg label - 1
    :param batch_size:
    :param num_classes:
    :return:
    '''
    # assert len(all_pids)>=batch_size, "not enough eligible pids {} to form a single batch of size {}".format(len(all_pids), batch_size)
    class_counts = {k: 0 for k in range(1,num_classes+1)}
    not_picked = np.array(all_pids)
    batch_patients = np.empty((batch_size,), dtype=not_picked.dtype)
    rarest_class = np.random.randint(1,num_classes+1)

    for ix in range(batch_size):
        if len(not_picked) == 0:
            warnings.warn("Dataset too small to generate batch with unique samples; => recycling.")
            not_picked = np.array(all_pids)

        np.random.shuffle(not_picked) #this could actually go outside(above) the loop.
        pick = not_picked[0]
        for cand in not_picked:
            if np.count_nonzero(class_targets[cand] == rarest_class) > 0:
                pick = cand
                cand_rarest_class = np.argmin([np.count_nonzero(class_targets[cand] == cl) for cl in
                                               range(1,num_classes+1)])+1
                # if current batch already bigger than the batch random ratio, then
                # check that weakest class in this patient is not the weakest in current batch (since needs to be boosted)
                # also that at least one roi of this patient belongs to weakest class. If True, keep patient, else keep looking.
                if (cand_rarest_class != rarest_class and np.count_nonzero(class_targets[cand] == rarest_class) > 0) \
                        or ix < int(batch_size * random_ratio):
                    break

        for c in range(1,num_classes+1):
            class_counts[c] += np.count_nonzero(class_targets[pick] == c)
        if not ix < int(batch_size * random_ratio) and class_counts[rarest_class] == 0:  # means searched thru whole set without finding rarest class
            print("Class {} not represented in current dataset.".format(rarest_class))
        rarest_class = np.argmin(([class_counts[c] for c in range(1,num_classes+1)]))+1
        batch_patients[ix] = pick
        not_picked = not_picked[not_picked != pick]  # removes pick

    return batch_patients


class BatchGenerator(SlimDataLoaderBase):
    """
    create the training/validation batch generator. Randomly sample batch_size patients
    from the data set, (draw a random slice if 2D), pad-crop them to equal sizes and merge to an array.
    :param data: data dictionary as provided by 'load_dataset'
    :param img_modalities: list of strings ['adc', 'b1500'] from config
    :param batch_size: number of patients to sample for the batch
    :param pre_crop_size: equal size for merging the patients to a single array (before the final random-crop in data aug.)
    :return dictionary containing the batch data / seg / pids as lists; the augmenter will later concatenate them into an array.
    """

    def __init__(self, cf, data, sample_pids_w_replace=True, max_batches=None, raise_stop_iteration=False, n_threads=None, seed=0):
        if n_threads is None:
            n_threads = cf.n_workers
        super(BatchGenerator, self).__init__(data, cf.batch_size, number_of_threads_in_multithreaded=n_threads)
        self.cf = cf
        self.random_count = int(cf.batch_random_ratio * cf.batch_size)
        self.plot_dir = os.path.join(self.cf.plot_dir, 'train_generator')
        os.makedirs(self.plot_dir, exist_ok=True)
        self.max_batches = max_batches
        self.raise_stop = raise_stop_iteration
        self.thread_id = 0
        self.batches_produced = 0

        self.dataset_length = len(self._data)
        self.dataset_pids = list(self._data.keys())


        self.sample_pids_w_replace = sample_pids_w_replace
        self.n_filled_threads = min(int(self.dataset_length/self.batch_size), self.number_of_threads_in_multithreaded)
        if not self.sample_pids_w_replace:
            # if not sampling w replace --> iterator-like behaviour but multi-threaded. adjust threads
            # s.t. each thread has enough patients for at least one batch.
            assert len(self.dataset_pids) / self.n_filled_threads >= self.batch_size, \
                "at least one batch needed per thread. dataset size: {}, n_threads: {}, batch_size: {}.".format(
                    len(self.dataset_pids), self.n_filled_threads, self.batch_size)
            self.lock = Lock()

            if self.n_filled_threads != self.number_of_threads_in_multithreaded:
                print("adjusting nr of threads from {} to {}.".format(self.number_of_threads_in_multithreaded,
                                                                      self.n_filled_threads))
        self.rgen = np.random.RandomState(seed=seed)
        self.eligible_pids = self.rgen.permutation(self.dataset_pids.copy())
        self.eligible_pids = np.array_split(self.eligible_pids, self.n_filled_threads)
        self.eligible_pids = sorted(self.eligible_pids, key=len, reverse=True)


        if hasattr(cf, "balance_target"):
            # WARNING: "balance targets are only implemented for 1-d targets (or 1-component vectors)"
            self.balance_target = cf.balance_target
        else:
            self.balance_target = "class_targets"
        self.targets = {k:v[self.balance_target] for (k,v) in self._data.items()}

    def set_thread_id(self, thread_id):
        self.thread_ids = self.eligible_pids[thread_id]
        self.thread_id  = thread_id

    def reset(self):
        self.batches_produced = 0
        self.thread_ids = self.rgen.permutation(self.eligible_pids[self.thread_id])

    @staticmethod
    def sample_targets_to_weights(targets, fg_bg_weights):
        weights = targets * fg_bg_weights
        return weights

    def balance_target_distribution(self, plot=False):
        """Impose a drawing distribution over samples.
         Distribution should be designed so that classes' fg and bg examples are (as good as possible) shown in
         equal frequency. Since we are dealing with rois, fg/bg weights count a sample (e.g., a patient) with
         **at least** one occurrence as fg, otherwise bg. For fg weights among classes, each RoI counts.

        :param all_pids:
        :param self.targets:  dic holding {patient_specifier : patient-wise-unique ROI targets}
        :return: probability distribution over all pids. draw without replace from this.
        """
        # oversampling of fg: limit bg weights to anything <= fg weights by setting factor < 1 to overweight fg.
        bg_weight_factor = 0.1

        self.unique_ts = np.unique([v for pat in self.targets.values() for v in pat])
        self.sample_stats = pd.DataFrame(columns=[str(ix)+suffix for ix in self.unique_ts for suffix in ["", "_bg"]], index=list(self.targets.keys()))
        for pid in self.sample_stats.index:
            for targ in self.unique_ts:
                fg_count = np.count_nonzero(self.targets[pid] == targ)
                self.sample_stats.loc[pid, str(targ)] = int(fg_count > 0)
                self.sample_stats.loc[pid, str(targ)+"_bg"] = int(fg_count == 0)

        self.targ_stats = self.sample_stats.agg(
            ("sum", lambda col: col.sum() / len(self._data)), axis=0, sort=False).rename({"<lambda>": "relative"})

        anchor = 1. - self.targ_stats.loc["relative"].iloc[0]
        self.fg_bg_weights = anchor / self.targ_stats.loc["relative"]
        cum_weights = anchor * len(self.fg_bg_weights)
        self.fg_bg_weights /= cum_weights
        mask = ["_bg" in ix for ix in self.fg_bg_weights.index]
        self.fg_bg_weights.loc[mask] = self.fg_bg_weights.loc[mask].apply(lambda x: x * bg_weight_factor)

        self.p_probs = self.sample_stats.apply(self.sample_targets_to_weights, args=(self.fg_bg_weights,), axis=1).sum(axis=1)
        self.p_probs = self.p_probs / self.p_probs.sum()
        if plot:
            print("Applying class-weights:\n {}".format(self.fg_bg_weights))

        self.stats = {"roi_counts": np.zeros(len(self.unique_ts,), dtype='uint32'),
                      "empty_counts": np.zeros(len(self.unique_ts,), dtype='uint32')}

        if plot:
            os.makedirs(self.plot_dir, exist_ok=True)
            plg.plot_batchgen_distribution(self.cf, self.dataset_pids, self.p_probs, self.balance_target,
                                           out_file=os.path.join(self.plot_dir,
                                                                 "train_gen_distr_"+str(self.cf.fold)+".png"))
        return self.p_probs

    def get_batch_pids(self):
        if self.max_batches is not None and self.batches_produced * self.n_filled_threads \
                + self.thread_id >= self.max_batches:
            self.reset()
            raise StopIteration

        if self.sample_pids_w_replace:
            # fully random patients
            batch_pids = list(np.random.choice(self.dataset_pids, size=self.random_count, replace=False))
            # target-balanced patients
            batch_pids += list(np.random.choice(
                self.dataset_pids, size=self.batch_size - self.random_count, replace=False, p=self.p_probs))
        else:
            with self.lock:
                if len(self.thread_ids) == 0:
                    if self.raise_stop:
                        self.reset()
                        raise StopIteration
                    else:
                        self.thread_ids = self.rgen.permutation(self.eligible_pids[self.thread_id])
                batch_pids = self.thread_ids[:self.batch_size]
                # batch_pids = np.random.choice(self.thread_ids, size=self.batch_size, replace=False)
                self.thread_ids = [pid for pid in self.thread_ids if pid not in batch_pids]
        self.batches_produced += 1

        return batch_pids

    def generate_train_batch(self):
        # to be overriden by child
        # everything done in here is per batch
        # print statements in here get confusing due to multithreading
        raise NotImplementedError

    def print_stats(self, logger=None, file=None, plot_file=None, plot=True):
        print_f = utils.CombinedPrinter(logger, file)

        print_f('\n***Final Training Stats***')
        total_count = np.sum(self.stats['roi_counts'])
        for tix, count in enumerate(self.stats['roi_counts']):
            #name = self.cf.class_dict[tix] if self.balance_target=="class_targets" else str(self.unique_ts[tix])
            name=str(self.unique_ts[tix])
            print_f('{}: {} rois seen ({:.1f}%).'.format(name, count, count / total_count * 100))
        total_samples = self.cf.num_epochs*self.cf.num_train_batches*self.cf.batch_size
        empties = [
        '{}: {} ({:.1f}%)'.format(str(name), self.stats['empty_counts'][tix],
                                    self.stats['empty_counts'][tix]/total_samples*100)
            for tix, name in enumerate(self.unique_ts)
        ]
        empties = ", ".join(empties)
        print_f('empty samples seen: {}\n'.format(empties))
        if plot:
            if plot_file is None:
                plot_file = os.path.join(self.plot_dir, "train_gen_stats_{}.png".format(self.cf.fold))
                os.makedirs(self.plot_dir, exist_ok=True)
            plg.plot_batchgen_stats(self.cf, self.stats, empties, self.balance_target, self.unique_ts, plot_file)

class PatientBatchIterator(SlimDataLoaderBase):
    """
    creates a val/test generator. Step through the dataset and return dictionaries per patient.
    2D is a special case of 3D patching with patch_size[2] == 1 (slices)
    Creates whole Patient batch and targets, and - if necessary - patchwise batch and targets.
    Appends patient targets anyway for evaluation.
    For Patching, shifts all patches into batch dimension. batch_tiling_forward will take care of exceeding batch dimensions.

    This iterator/these batches are not intended to go through MTaugmenter afterwards
    """

    def __init__(self, cf, data):
        super(PatientBatchIterator, self).__init__(data, 0)
        self.cf = cf

        self.dataset_length = len(self._data)
        self.dataset_pids = list(self._data.keys())

    def generate_train_batch(self, pid=None):
        # to be overriden by child

        return

###################################
#  transforms, image manipulation #
###################################

def get_patch_crop_coords(img, patch_size, min_overlap=30):
    """
    _:param img (y, x, (z))
    _:param patch_size: list of len 2 (2D) or 3 (3D).
    _:param min_overlap: minimum required overlap of patches.
    If too small, some areas are poorly represented only at edges of single patches.
    _:return ndarray: shape (n_patches, 2*dim). crop coordinates for each patch.
    """
    crop_coords = []
    for dim in range(len(img.shape)):
        n_patches = int(np.ceil(img.shape[dim] / patch_size[dim]))

        # no crops required in this dimension, add image shape as coordinates.
        if n_patches == 1:
            crop_coords.append([(0, img.shape[dim])])
            continue

        # fix the two outside patches to coords patchsize/2 and interpolate.
        center_dists = (img.shape[dim] - patch_size[dim]) / (n_patches - 1)

        if (patch_size[dim] - center_dists) < min_overlap:
            n_patches += 1
            center_dists = (img.shape[dim] - patch_size[dim]) / (n_patches - 1)

        patch_centers = np.round([(patch_size[dim] / 2 + (center_dists * ii)) for ii in range(n_patches)])
        dim_crop_coords = [(center - patch_size[dim] / 2, center + patch_size[dim] / 2) for center in patch_centers]
        crop_coords.append(dim_crop_coords)

    coords_mesh_grid = []
    for ymin, ymax in crop_coords[0]:
        for xmin, xmax in crop_coords[1]:
            if len(crop_coords) == 3 and patch_size[2] > 1:
                for zmin, zmax in crop_coords[2]:
                    coords_mesh_grid.append([ymin, ymax, xmin, xmax, zmin, zmax])
            elif len(crop_coords) == 3 and patch_size[2] == 1:
                for zmin in range(img.shape[2]):
                    coords_mesh_grid.append([ymin, ymax, xmin, xmax, zmin, zmin + 1])
            else:
                coords_mesh_grid.append([ymin, ymax, xmin, xmax])
    return np.array(coords_mesh_grid).astype(int)

def pad_nd_image(image, new_shape=None, mode="edge", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    """
    one padder to pad them all. Documentation? Well okay. A little bit. by Fabian Isensee

    :param image: nd image. can be anything
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
    len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
    the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
    Example:
    image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
    image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).

    :param mode: see np.pad for documentation
    :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
    to original shape
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
    divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation
    """
    if kwargs is None:
        kwargs = {}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])
    res = np.pad(image, pad_list, mode, **kwargs)
    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer

def convert_seg_to_bounding_box_coordinates(data_dict, dim, roi_item_keys, get_rois_from_seg=False,
                                                class_specific_seg=False):
    '''adapted from batchgenerators

    :param data_dict: seg: segmentation with labels indicating roi_count (get_rois_from_seg=False) or classes (get_rois_from_seg=True),
        class_targets: list where list index corresponds to roi id (roi_count)
    :param dim:
    :param roi_item_keys: keys of the roi-wise items in data_dict to process
    :param n_rg_feats: nr of regression vector features
    :param get_rois_from_seg:
    :return: coords (y1,x1,y2,x2 (,z1,z2)) where the segmentation GT is framed by +1 voxel, i.e., for an object with
        z-extensions z1=0 through z2=5, bbox target coords will be z1=-1, z2=6. (analogically for x,y).
        data_dict['roi_masks']: (b, n(b), c, h(n), w(n) (z(n))) list like roi_labels but with arrays (masks) inplace of
        integers. c==1 if segmentation not one-hot encoded.
    '''

    bb_target = []
    roi_masks = []
    roi_items = {name:[] for name in roi_item_keys}
    out_seg = np.copy(data_dict['seg'])
    for b in range(data_dict['seg'].shape[0]):

        p_coords_list = [] #p for patient?
        p_roi_masks_list = []
        p_roi_items_lists = {name:[] for name in roi_item_keys}

        if np.sum(data_dict['seg'][b] != 0) > 0:
            if get_rois_from_seg:
                clusters, n_cands = lb(data_dict['seg'][b])
                data_dict['class_targets'][b] = [data_dict['class_targets'][b]] * n_cands
            else:
                n_cands = int(np.max(data_dict['seg'][b]))

            rois = np.array(
                [(data_dict['seg'][b] == ii) * 1 for ii in range(1, n_cands + 1)], dtype='uint8')  # separate clusters

            for rix, r in enumerate(rois):
                if np.sum(r != 0) > 0:  # check if the roi survived slicing (3D->2D) and data augmentation (cropping etc.)
                    seg_ixs = np.argwhere(r != 0)
                    coord_list = [np.min(seg_ixs[:, 1]) - 1, np.min(seg_ixs[:, 2]) - 1, np.max(seg_ixs[:, 1]) + 1,
                                  np.max(seg_ixs[:, 2]) + 1]
                    if dim == 3:
                        coord_list.extend([np.min(seg_ixs[:, 3]) - 1, np.max(seg_ixs[:, 3]) + 1])

                    p_coords_list.append(coord_list)
                    p_roi_masks_list.append(r)
                    # add background class = 0. rix is a patient wide index of lesions. since 'class_targets' is
                    # also patient wide, this assignment is not dependent on patch occurrences.
                    for name in roi_item_keys:
                        p_roi_items_lists[name].append(data_dict[name][b][rix])

                    assert data_dict["class_targets"][b][rix]>=1, "convertsegtobbox produced bg roi w cl targ {} and unique roi seg {}".format(data_dict["class_targets"][b][rix], np.unique(r))


                if class_specific_seg:
                    out_seg[b][data_dict['seg'][b] == rix + 1] = data_dict['class_targets'][b][rix]

            if not class_specific_seg:
                out_seg[b][data_dict['seg'][b] > 0] = 1

            bb_target.append(np.array(p_coords_list))
            roi_masks.append(np.array(p_roi_masks_list))
            for name in roi_item_keys:
                roi_items[name].append(np.array(p_roi_items_lists[name]))


        else:
            bb_target.append([])
            roi_masks.append(np.zeros_like(data_dict['seg'][b], dtype='uint8')[None])
            for name in roi_item_keys:
                roi_items[name].append(np.array([]))

    if get_rois_from_seg:
        data_dict.pop('class_targets', None)

    data_dict['bb_target'] = np.array(bb_target)
    data_dict['roi_masks'] = np.array(roi_masks)
    data_dict['seg'] = out_seg
    for name in roi_item_keys:
        data_dict[name] = np.array(roi_items[name])


    return data_dict

class ConvertSegToBoundingBoxCoordinates(AbstractTransform):
    """ Converts segmentation masks into bounding box coordinates.
    """

    def __init__(self, dim, roi_item_keys, get_rois_from_seg=False, class_specific_seg=False):
        self.dim = dim
        self.roi_item_keys = roi_item_keys
        self.get_rois_from_seg = get_rois_from_seg
        self.class_specific_seg = class_specific_seg

    def __call__(self, **data_dict):
        return convert_seg_to_bounding_box_coordinates(data_dict, self.dim, self.roi_item_keys, self.get_rois_from_seg,
                                                       self.class_specific_seg)





#############################
#  data packing / unpacking # not used, data_manager.py used instead
#############################

def get_case_identifiers(folder):
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz")]
    return case_identifiers


def convert_to_npy(npz_file):
    if not os.path.isfile(npz_file[:-3] + "npy"):
        a = np.load(npz_file)['data']
        np.save(npz_file[:-3] + "npy", a)


def unpack_dataset(folder, threads=8):
    case_identifiers = get_case_identifiers(folder)
    p = Pool(threads)
    npz_files = [os.path.join(folder, i + ".npz") for i in case_identifiers]
    p.map(convert_to_npy, npz_files)
    p.close()
    p.join()


def delete_npy(folder):
    case_identifiers = get_case_identifiers(folder)
    npy_files = [os.path.join(folder, i + ".npy") for i in case_identifiers]
    npy_files = [i for i in npy_files if os.path.isfile(i)]
    for n in npy_files:
        os.remove(n)