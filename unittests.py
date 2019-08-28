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

import unittest

import os
import pickle
import time
from multiprocessing import  Pool

import numpy as np
import pandas as pd

import utils.exp_utils as utils
import utils.model_utils as mutils

""" Note on unittests: run this file either in the way intended for unittests by starting the script with
    python -m unittest unittests.py or start it as a normal python file as python unittests.py.
    You can selective run single tests by calling python -m unittest unittests.TestClassOfYourChoice, where 
    TestClassOfYourChoice is the name of the test defined below, e.g., CompareFoldSplits.
"""



def inspect_info_df(pp_dir):
    """ use your debugger to look into the info df of a pp dir.
    :param pp_dir: preprocessed-data directory
    """

    info_df = pd.read_pickle(os.path.join(pp_dir, "info_df.pickle"))

    return

#------- perform integrity checks on data set(s) -----------
class VerifyLIDCSAIntegrity(unittest.TestCase):
    """ Perform integrity checks on preprocessed single-annotator GTs of LIDC data set.
    """
    @staticmethod
    def check_patient_sa_gt(pid, pp_dir, check_meta_files, check_info_df):

        faulty_cases = pd.DataFrame(columns=['pid', 'rater', 'cl_targets', 'roi_ids'])

        all_segs = np.load(os.path.join(pp_dir, pid + "_rois.npz"), mmap_mode='r')
        all_segs = all_segs[list(all_segs.keys())[0]]
        all_roi_ids = np.unique(all_segs[all_segs > 0])
        assert len(all_roi_ids) == np.max(all_segs), "roi ids not consecutive"
        if check_meta_files:
            meta_file = os.path.join(pp_dir, pid + "_meta_info.pickle")
            with open(meta_file, "rb") as handle:
                info = pickle.load(handle)
            assert info["pid"] == pid, "wrong pid in meta_file"
            all_cl_targets = info["class_target"]
        if check_info_df:
            info_df = pd.read_pickle(os.path.join(pp_dir, "info_df.pickle"))
            pid_info = info_df[info_df.pid == pid]
            assert len(pid_info) == 1, "found {} entries for pid {} in info df, expected exactly 1".format(len(pid_info),
                                                                                                           pid)
            if check_meta_files:
                assert pid_info[
                           "class_target"] == all_cl_targets, "meta_info and info_df class targets mismatch:\n{}\n{}".format(
                    pid_info["class_target"], all_cl_targets)
            all_cl_targets = pid_info["class_target"].iloc[0]
        assert len(all_roi_ids) == len(all_cl_targets)
        for rater in range(4):
            seg = all_segs[rater]
            roi_ids = np.unique(seg[seg > 0])
            cl_targs = np.array([roi[rater] for roi in all_cl_targets])
            assert np.count_nonzero(cl_targs) == len(roi_ids), "rater {} has targs {} but roi ids {}".format(rater, cl_targs, roi_ids)
            assert len(cl_targs) >= len(roi_ids), "not all marked rois have a label"
            for zeroix_roi_id, rating in enumerate(cl_targs):
                if not ((rating > 0) == (np.any(seg == zeroix_roi_id + 1))):
                    print("\n\nFAULTY CASE:", end=" ", )
                    print("pid {}, rater {}, cl_targs {}, ids {}\n".format(pid, rater, cl_targs, roi_ids))
                    faulty_cases = faulty_cases.append(
                        {'pid': pid, 'rater': rater, 'cl_targets': cl_targs, 'roi_ids': roi_ids}, ignore_index=True)
        print("finished checking pid {}, {} faulty cases".format(pid, len(faulty_cases)))
        return faulty_cases

    def check_sa_gts(self, pp_dir, pid_subset=None, check_meta_files=False, check_info_df=True, processes=os.cpu_count()):
        report_name = "verify_seg_label_pairings.csv"
        pids = {file_name.split("_")[0] for file_name in os.listdir(pp_dir) if file_name not in [report_name, "info_df.pickle"]}
        if pid_subset is not None:
            pids = [pid for pid in pids if pid in pid_subset]


        faulty_cases = pd.DataFrame(columns=['pid', 'rater', 'cl_targets', 'roi_ids'])

        p = Pool(processes=processes)
        mp_args = zip(pids, [pp_dir]*len(pids), [check_meta_files]*len(pids), [check_info_df]*len(pids))
        patient_cases = p.starmap(self.check_patient_sa_gt, mp_args)
        p.close(); p.join()
        faulty_cases = faulty_cases.append(patient_cases, sort=False)


        print("\n\nfaulty case count {}".format(len(faulty_cases)))
        print(faulty_cases)
        findings_file = os.path.join(pp_dir, "verify_seg_label_pairings.csv")
        faulty_cases.to_csv(findings_file)

        assert len(faulty_cases)==0, "there was a faulty case in data set {}.\ncheck {}".format(pp_dir, findings_file)

    def test(self):
        pp_root = "/mnt/HDD2TB/Documents/data/"
        pp_dir = "lidc/pp_20190805"
        gt_dir = os.path.join(pp_root, pp_dir, "patient_gts_sa")
        self.check_sa_gts(gt_dir, check_meta_files=True, check_info_df=False, pid_subset=None)  # ["0811a", "0812a"])

#------ compare segmentation gts of preprocessed data sets ------
class CompareSegGTs(unittest.TestCase):
    """ load and compare pre-processed gts by dice scores of segmentations.

    """
    @staticmethod
    def group_seg_paths(ref_path, comp_paths):
        # not working recursively
        ref_files = [fn for fn in os.listdir(ref_path) if
                     os.path.isfile(os.path.join(ref_path, fn)) and 'seg' in fn and fn.endswith('.npy')]

        comp_files = [[os.path.join(c_path, fn) for c_path in comp_paths] for fn in ref_files]

        ref_files = [os.path.join(ref_path, fn) for fn in ref_files]

        return zip(ref_files, comp_files)

    @staticmethod
    def load_calc_dice(paths):
        dices = []
        ref_seg = np.load(paths[0])[np.newaxis, np.newaxis]
        n_classes = len(np.unique(ref_seg))
        ref_seg = mutils.get_one_hot_encoding(ref_seg, n_classes)

        for c_file in paths[1]:
            c_seg = np.load(c_file)[np.newaxis, np.newaxis]
            assert n_classes == len(np.unique(c_seg)), "unequal nr of objects/classes betw segs {} {}".format(paths[0],
                                                                                                              c_file)
            c_seg = mutils.get_one_hot_encoding(c_seg, n_classes)

            dice = mutils.dice_per_batch_inst_and_class(c_seg, ref_seg, n_classes, convert_to_ohe=False)
            dices.append(dice)
        print("processed ref_path {}".format(paths[0]))
        return np.mean(dices), np.std(dices)

    def iterate_files(self, grouped_paths, processes=os.cpu_count()):
        p = Pool(processes)

        means_stds = np.array(p.map(self.load_calc_dice, grouped_paths))

        p.close(); p.join()
        min_dice = np.min(means_stds[:, 0])
        print("min mean dice {:.2f}, max std {:.4f}".format(min_dice, np.max(means_stds[:, 1])))
        assert min_dice > 1-1e5, "compared seg gts have insufficient minimum mean dice overlap of {}".format(min_dice)

    def test(self):
        ref_path = '/mnt/HDD2TB/Documents/data/prostate/data_t2_250519_ps384_gs6071'
        comp_paths = ['/mnt/HDD2TB/Documents/data/prostate/data_t2_190419_ps384_gs6071', ]
        paths = self.group_seg_paths(ref_path, comp_paths)
        self.iterate_files(paths)

#------- check if cross-validation fold splits of different experiments are identical ----------
class CompareFoldSplits(unittest.TestCase):
    """ Find evtl. differences in cross-val file splits across different experiments.
    """
    @staticmethod
    def group_id_paths(ref_exp_dir, comp_exp_dirs):

        f_name = 'fold_ids.pickle'

        ref_paths = os.path.join(ref_exp_dir, f_name)
        assert os.path.isfile(ref_paths), "ref file {} does not exist.".format(ref_paths)


        ref_paths = [ref_paths for comp_ed in comp_exp_dirs]
        comp_paths = [os.path.join(comp_ed, f_name) for comp_ed in comp_exp_dirs]

        return zip(ref_paths, comp_paths)

    @staticmethod
    def comp_fold_ids(mp_input):
        fold_ids1, fold_ids2 = mp_input
        with open(fold_ids1, 'rb') as f:
            fold_ids1 = pickle.load(f)
        try:
            with open(fold_ids2, 'rb') as f:
                fold_ids2 = pickle.load(f)
        except FileNotFoundError:
            print("comp file {} does not exist.".format(fold_ids2))
            return

        n_splits = len(fold_ids1)
        assert n_splits == len(fold_ids2), "mismatch n splits: ref has {}, comp {}".format(n_splits, len(fold_ids2))
        split_diffs = [np.setdiff1d(fold_ids1[s], fold_ids2[s]) for s in range(n_splits)]
        all_equal = np.any(split_diffs)
        return (split_diffs, all_equal)

    def iterate_exp_dirs(self, ref_exp, comp_exps, processes=os.cpu_count()):

        grouped_paths = list(self.group_id_paths(ref_exp, comp_exps))
        print("performing {} comparisons of cross-val file splits".format(len(grouped_paths)))
        p = Pool(processes)
        split_diffs = p.map(self.comp_fold_ids, grouped_paths)
        p.close(); p.join()

        df = pd.DataFrame(index=range(0,len(grouped_paths)), columns=["ref", "comp", "all_equal"])#, "diffs"])
        for ix, (ref, comp) in enumerate(grouped_paths):
            df.iloc[ix] = [ref, comp, split_diffs[ix][1]]#, split_diffs[ix][0]]

        print("Any splits not equal?", df.all_equal.any())
        assert not df.all_equal.any(), "a split set is different from reference split set, {}".format(df[~df.all_equal])

    def test(self):
        exp_parent_dir = '/home/gregor/networkdrives/E132-Cluster-Projects/prostate/experiments/'
        ref_exp = '/home/gregor/networkdrives/E132-Cluster-Projects/prostate/experiments/gs6071_detfpn2d_cl_bs10'
        comp_exps = [os.path.join(exp_parent_dir, p) for p in os.listdir(exp_parent_dir)]
        comp_exps = [p for p in comp_exps if os.path.isdir(p) and p != ref_exp]
        self.iterate_exp_dirs(ref_exp, comp_exps)


#------- check if cross-validation fold splits of a single experiment are actually incongruent (as required) ----------
class VerifyFoldSplits(unittest.TestCase):
    """ Check, for a single fold_ids file, i.e., for a single experiment, if the assigned folds (assignment of data
        identifiers) is actually incongruent. No overlaps between folds are required for a correct cross validation.
    """
    @staticmethod
    def verify_fold_ids(splits):
        for i, split1 in enumerate(splits):
            for j, split2 in enumerate(splits):
                if j > i:
                    inter = np.intersect1d(split1, split2)
                    if len(inter) > 0:
                        raise Exception("Split {} and {} intersect by pids {}".format(i, j, inter))
    def test(self):
        exp_dir = "/home/gregor/Documents/medicaldetectiontoolkit/datasets/lidc/experiments/dev"
        check_file = os.path.join(exp_dir, 'fold_ids.pickle')
        with open(check_file, 'rb') as handle:
            splits = pickle.load(handle)
        self.verify_fold_ids(splits)

if __name__=="__main__":
    stime = time.time()

    unittest.main()

    mins, secs = divmod((time.time() - stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))