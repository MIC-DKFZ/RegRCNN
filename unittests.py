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
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision as tv

import tqdm

import plotting as plg
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


def generate_boxes(count, dim=2, h=100, w=100, d=20, normalize=False, on_grid=False, seed=0):
    """ generate boxes of format [y1, x1, y2, x2, (z1, z2)].
    :param count: nr of boxes
    :param dim: dimension of boxes (2 or 3)
    :return: boxes in format (n_boxes, 4 or 6), scores
    """
    np.random.seed(seed)
    if on_grid:
        lower_y = np.random.randint(0, h // 2, (count,))
        lower_x = np.random.randint(0, w // 2, (count,))
        upper_y = np.random.randint(h // 2, h, (count,))
        upper_x = np.random.randint(w // 2, w, (count,))
        if dim == 3:
            lower_z = np.random.randint(0, d // 2, (count,))
            upper_z = np.random.randint(d // 2, d, (count,))
    else:
        lower_y = np.random.rand(count) * h / 2.
        lower_x = np.random.rand(count) * w / 2.
        upper_y = (np.random.rand(count) + 1.) * h / 2.
        upper_x = (np.random.rand(count) + 1.) * w / 2.
        if dim == 3:
            lower_z = np.random.rand(count) * d / 2.
            upper_z = (np.random.rand(count) + 1.) * d / 2.

    if dim == 3:
        boxes = np.array(list(zip(lower_y, lower_x, upper_y, upper_x, lower_z, upper_z)))
        # add an extreme box that tests the boundaries
        boxes = np.concatenate((boxes, np.array([[0., 0., h, w, 0, d]])))
    else:
        boxes = np.array(list(zip(lower_y, lower_x, upper_y, upper_x)))
        boxes = np.concatenate((boxes, np.array([[0., 0., h, w]])))

    scores = np.random.rand(count + 1)
    if normalize:
        divisor = np.array([h, w, h, w, d, d]) if dim == 3 else np.array([h, w, h, w])
        boxes = boxes / divisor
    return boxes, scores

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

    def check_sa_gts(cf, pp_dir, pid_subset=None, check_meta_files=False, check_info_df=True, processes=os.cpu_count()):
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
        pp_root = "/media/gregor/HDD2TB/Documents/data/"
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
        ref_path = '/media/gregor/HDD2TB/Documents/data/prostate/data_t2_250519_ps384_gs6071'
        comp_paths = ['/media/gregor/HDD2TB/Documents/data/prostate/data_t2_190419_ps384_gs6071', ]
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
        identifiers) is actually incongruent. No overlaps between folds are allowed for a correct cross validation.
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

# -------- check own nms CUDA implement against own numpy implement ------
class CheckNMSImplementation(unittest.TestCase):

    @staticmethod
    def assert_res_equality(keep_ics1, keep_ics2, boxes, scores, tolerance=0, names=("res1", "res2")):
        """
        :param keep_ics1: keep indices (results), torch.Tensor of shape (n_ics,)
        :param keep_ics2:
        :return:
        """
        keep_ics1, keep_ics2 = keep_ics1.cpu().numpy(), keep_ics2.cpu().numpy()
        discrepancies = np.setdiff1d(keep_ics1, keep_ics2)
        try:
            checks = np.array([
                len(discrepancies) <= tolerance
            ])
        except:
            checks = np.zeros((1,)).astype("bool")
        msgs = np.array([
            """{}: {} \n{}: {} \nboxes: {}\n {}\n""".format(names[0], keep_ics1, names[1], keep_ics2, boxes,
                                                            scores)
        ])

        assert np.all(checks), "NMS: results mismatch: " + "\n".join(msgs[~checks])

    def single_case(self, count=20, dim=3, threshold=0.2, seed=0):
        boxes, scores = generate_boxes(count, dim, seed=seed, h=320, w=280, d=30)

        keep_numpy = torch.tensor(mutils.nms_numpy(boxes, scores, threshold))

        # for some reason torchvision nms requires box coords as floats.
        boxes = torch.from_numpy(boxes).type(torch.float32)
        scores = torch.from_numpy(scores).type(torch.float32)
        if dim == 2:
            """need to wait until next pytorch release where they fixed nms on cpu (currently they have >= where it
            needs to be >.
            """
            keep_ops = tv.ops.nms(boxes, scores, threshold)
            # self.assert_res_equality(keep_numpy, keep_ops, boxes, scores, tolerance=0, names=["np", "ops"])
            pass

        boxes = boxes.cuda()
        scores = scores.cuda()
        keep = self.nms_ext.nms(boxes, scores, threshold)
        self.assert_res_equality(keep_numpy, keep, boxes, scores, tolerance=0, names=["np", "cuda"])

    def test(self, n_cases=200, box_count=30, threshold=0.5):
        # dynamically import module so that it doesn't affect other tests if import fails
        self.nms_ext = utils.import_module("nms_ext", 'custom_extensions/nms/nms.py')
        # change seed to something fix if you want exactly reproducible test
        seed0 = np.random.randint(50)
        print("NMS test progress (done/total box configurations) 2D:", end="\n")
        for i in tqdm.tqdm(range(n_cases)):
            self.single_case(count=box_count, dim=2, threshold=threshold, seed=seed0+i)
        print("NMS test progress (done/total box configurations) 3D:", end="\n")
        for i in tqdm.tqdm(range(n_cases)):
            self.single_case(count=box_count, dim=3, threshold=threshold, seed=seed0+i)

        return

class CheckRoIAlignImplementation(unittest.TestCase):

    def prepare(self, dim=2):

        b, c, h, w = 1, 3, 50, 50
        # feature map, (b, c, h, w(, z))
        if dim == 2:
            fmap = torch.rand(b, c, h, w).cuda()
            # rois = torch.tensor([[
            #     [0.1, 0.1, 0.3, 0.3],
            #     [0.2, 0.2, 0.4, 0.7],
            #     [0.5, 0.7, 0.7, 0.9],
            # ]]).cuda()
            pool_size = (7, 7)
            rois = generate_boxes(5, dim=dim, h=h, w=w, on_grid=True, seed=np.random.randint(50))[0]
        elif dim == 3:
            d = 20
            fmap = torch.rand(b, c, h, w, d).cuda()
            # rois = torch.tensor([[
            #     [0.1, 0.1, 0.3, 0.3, 0.1, 0.1],
            #     [0.2, 0.2, 0.4, 0.7, 0.2, 0.4],
            #     [0.5, 0.0, 0.7, 1.0, 0.4, 0.5],
            #     [0.0, 0.0, 0.9, 1.0, 0.0, 1.0],
            # ]]).cuda()
            pool_size = (7, 7, 3)
            rois = generate_boxes(5, dim=dim, h=h, w=w, d=d, on_grid=True, seed=np.random.randint(50),
                                  normalize=False)[0]
        else:
            raise ValueError("dim needs to be 2 or 3")

        rois = [torch.from_numpy(rois).type(dtype=torch.float32).cuda(), ]
        fmap.requires_grad_(True)
        return fmap, rois, pool_size

    def check_2d(self):
        """ check vs torchvision ops not possible as on purpose different approach.
        :return:
        """
        raise NotImplementedError
        # fmap, rois, pool_size = self.prepare(dim=2)
        # ra_object = self.ra_ext.RoIAlign(output_size=pool_size, spatial_scale=1., sampling_ratio=-1)
        # align_ext = ra_object(fmap, rois)
        # loss_ext = align_ext.sum()
        # loss_ext.backward()
        #
        # rois_swapped = [rois[0][:, [1,3,0,2]]]
        # align_ops = tv.ops.roi_align(fmap, rois_swapped, pool_size)
        # loss_ops = align_ops.sum()
        # loss_ops.backward()
        #
        # assert (loss_ops == loss_ext), "sum of roialign ops and extension 2D diverges"
        # assert (align_ops == align_ext).all(), "ROIAlign failed 2D test"

    def check_3d(self):
        fmap, rois, pool_size = self.prepare(dim=3)
        ra_object = self.ra_ext.RoIAlign(output_size=pool_size, spatial_scale=1., sampling_ratio=-1)
        align_ext = ra_object(fmap, rois)
        loss_ext = align_ext.sum()
        loss_ext.backward()

        align_np = mutils.roi_align_3d_numpy(fmap.cpu().detach().numpy(), [roi.cpu().numpy() for roi in rois],
                                             pool_size)
        align_np = np.squeeze(align_np)  # remove singleton batch dim

        align_ext = align_ext.cpu().detach().numpy()
        assert np.allclose(align_np, align_ext, rtol=1e-5,
                           atol=1e-8), "RoIAlign differences in numpy and CUDA implement"

    def specific_example_check(self):
        # dummy input
        self.ra_ext = utils.import_module("ra_ext", 'custom_extensions/roi_align/roi_align.py')
        exp = 6
        pool_size = (2,2)
        fmap = torch.arange(exp**2).view(exp,exp).unsqueeze(0).unsqueeze(0).cuda().type(dtype=torch.float32)

        boxes = torch.tensor([[1., 1., 5., 5.]]).cuda()/exp
        ind = torch.tensor([0.]*len(boxes)).cuda().type(torch.float32)
        y_exp, x_exp = fmap.shape[2:]  # exp = expansion
        boxes.mul_(torch.tensor([y_exp, x_exp, y_exp, x_exp], dtype=torch.float32).cuda())
        boxes = torch.cat((ind.unsqueeze(1), boxes), dim=1)
        aligned_tv = tv.ops.roi_align(fmap, boxes, output_size=pool_size, sampling_ratio=-1)
        aligned = self.ra_ext.roi_align_2d(fmap, boxes, output_size=pool_size, sampling_ratio=-1)

        boxes_3d = torch.cat((boxes, torch.tensor([[-1.,1.]]*len(boxes)).cuda()), dim=1)
        fmap_3d = fmap.unsqueeze(dim=-1)
        pool_size = (*pool_size,1)
        ra_object = self.ra_ext.RoIAlign(output_size=pool_size, spatial_scale=1.,)
        aligned_3d = ra_object(fmap_3d, boxes_3d)

        # expected_res = torch.tensor([[[[10.5000, 12.5000], # this would be with an alternative grid-point setting
        #                                [22.5000, 24.5000]]]]).cuda()
        expected_res = torch.tensor([[[[14., 16.],
                                       [26., 28.]]]]).cuda()
        expected_res_3d = torch.tensor([[[[[14.],[16.]],
                                          [[26.],[28.]]]]]).cuda()
        assert torch.all(aligned==expected_res), "2D RoIAlign check vs. specific example failed. res: {}\n expected: {}\n".format(aligned, expected_res)
        assert torch.all(aligned_3d==expected_res_3d), "3D RoIAlign check vs. specific example failed. res: {}\n expected: {}\n".format(aligned_3d, expected_res_3d)

    def manual_check(self):
        """ print examples from a toy batch to file.
        :return:
        """
        self.ra_ext = utils.import_module("ra_ext", 'custom_extensions/roi_align/roi_align.py')
        # actual mrcnn mask input
        from datasets.toy import configs
        cf = configs.Configs()
        cf.exp_dir = "datasets/toy/experiments/dev/"
        cf.plot_dir = cf.exp_dir + "plots"
        os.makedirs(cf.exp_dir, exist_ok=True)
        cf.fold = 0
        cf.n_workers = 1
        logger = utils.get_logger(cf.exp_dir)
        data_loader = utils.import_module('data_loader', os.path.join("datasets", "toy", 'data_loader.py'))
        batch_gen = data_loader.get_train_generators(cf, logger=logger)
        batch = next(batch_gen['train'])
        roi_mask = np.zeros((1, 320, 200))
        bb_target = (np.array([50, 40, 90, 120])).astype("int")
        roi_mask[:, bb_target[0]+1:bb_target[2]+1, bb_target[1]+1:bb_target[3]+1] = 1.
        #batch = {"roi_masks": np.array([np.array([roi_mask, roi_mask]), np.array([roi_mask])]), "bb_target": [[bb_target, bb_target + 25], [bb_target-20]]}
        #batch_boxes_cor = [torch.tensor(batch_el_boxes).cuda().float() for batch_el_boxes in batch_cor["bb_target"]]
        batch_boxes = [torch.tensor(batch_el_boxes).cuda().float() for batch_el_boxes in batch["bb_target"]]
        #import IPython; IPython.embed()
        for b in range(len(batch_boxes)):
            roi_masks = batch["roi_masks"][b]
            #roi_masks_cor = batch_cor["roi_masks"][b]
            if roi_masks.sum()>0:
                boxes = batch_boxes[b]
                roi_masks = torch.tensor(roi_masks).cuda().type(dtype=torch.float32)
                box_ids = torch.arange(roi_masks.shape[0]).cuda().unsqueeze(1).type(dtype=torch.float32)
                masks = tv.ops.roi_align(roi_masks, [boxes], cf.mask_shape)
                masks = masks.squeeze(1)
                masks = torch.round(masks)
                masks_own = self.ra_ext.roi_align_2d(roi_masks, torch.cat((box_ids, boxes), dim=1), cf.mask_shape)
                boxes = boxes.type(torch.int)
                #print("check roi mask", roi_masks[0, 0, boxes[0][0]:boxes[0][2], boxes[0][1]:boxes[0][3]].sum(), (boxes[0][2]-boxes[0][0]) * (boxes[0][3]-boxes[0][1]))
                #print("batch masks", batch["roi_masks"])
                masks_own = masks_own.squeeze(1)
                masks_own = torch.round(masks_own)
                #import IPython; IPython.embed()
                for mix, mask in enumerate(masks):
                    fig = plg.plt.figure()
                    ax = fig.add_subplot()
                    ax.imshow(roi_masks[mix][0].cpu().numpy(), cmap="gray", vmin=0.)
                    ax.axis("off")
                    y1, x1, y2, x2 = boxes[mix]
                    bbox = plg.mpatches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=0.9, edgecolor="c", facecolor='none')
                    ax.add_patch(bbox)
                    x1, y1, x2, y2 = boxes[mix]
                    bbox = plg.mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=0.9, edgecolor="r",
                                                  facecolor='none')
                    ax.add_patch(bbox)
                    debug_dir = Path("/home/gregor/Documents/regrcnn/datasets/toy/experiments/debugroial")
                    os.makedirs(debug_dir, exist_ok=True)
                    plg.plt.savefig(debug_dir/"mask_b{}_{}.png".format(b, mix))
                    plg.plt.imsave(debug_dir/"mask_b{}_{}_pooled_tv.png".format(b, mix), mask.cpu().numpy(), cmap="gray", vmin=0.)
                    plg.plt.imsave(debug_dir/"mask_b{}_{}_pooled_own.png".format(b, mix), masks_own[mix].cpu().numpy(), cmap="gray", vmin=0.)
        return

    def test(self):
        # dynamically import module so that it doesn't affect other tests if import fails
        self.ra_ext = utils.import_module("ra_ext", 'custom_extensions/roi_align/roi_align.py')

        self.specific_example_check()

        # 2d test
        #self.check_2d()

        # 3d test
        self.check_3d()

        return


class CheckRuntimeErrors(unittest.TestCase):
    """ Check if minimal examples of the exec.py module finish without runtime errors.
        This check requires a working path to data in the toy-dataset configs.
    """

    def test(self):
        cf = utils.import_module("toy_cf", 'datasets/toy/configs.py').Configs()
        exp_dir = "./unittesting/"
        #checks = {"retina_net": False, "mrcnn": False}
        #print("Testing for runtime errors with models {}".format(list(checks.keys())))
        #for model in tqdm.tqdm(list(checks.keys())):
            # cf.model = model
            # cf.model_path = 'models/{}.py'.format(cf.model if not 'retina' in cf.model else 'retina_net')
            # cf.model_path = os.path.join(cf.source_dir, cf.model_path)
            # {'mrcnn': cf.add_mrcnn_configs,
            #  'retina_net': cf.add_mrcnn_configs, 'retina_unet': cf.add_mrcnn_configs,
            #  'detection_unet': cf.add_det_unet_configs, 'detection_fpn': cf.add_det_fpn_configs
            #  }[model]()
        # todo change structure of configs-handling with exec.py so that its dynamically parseable instead of needing to
        # todo be changed in the file all the time.
        checks = {cf.model:False}
        completed_process = subprocess.run("python exec.py --dev --dataset_name toy -m train_test --exp_dir {}".format(exp_dir),
                                           shell=True, capture_output=True, text=True)
        if completed_process.returncode!=0:
            print("Runtime test of model {} failed due to\n{}".format(cf.model, completed_process.stderr))
        else:
            checks[cf.model] = True
        subprocess.call("rm -rf {}".format(exp_dir), shell=True)
        assert all(checks.values()), "A runtime test crashed."

class MulithreadedDataiterator(unittest.TestCase):

    def test(self):
        print("Testing multithreaded iterator.")


        dataset = "toy"
        exp_dir = Path("datasets/{}/experiments/dev".format(dataset))
        cf_file = utils.import_module("cf_file", exp_dir/"configs.py")
        cf = cf_file.Configs()
        dloader = utils.import_module('data_loader', 'datasets/{}/data_loader.py'.format(dataset))
        cf.exp_dir = Path(exp_dir)
        cf.n_workers = 5

        cf.batch_size = 3
        cf.fold = 0
        cf.plot_dir = cf.exp_dir / "plots"
        logger = utils.get_logger(cf.exp_dir, cf.server_env, cf.sysmetrics_interval)
        cf.num_val_batches = "all"
        cf.val_mode = "val_sampling"
        cf.n_workers = 8
        batch_gens = dloader.get_train_generators(cf, logger, data_statistics=False)
        val_loader = batch_gens["val_sampling"]

        for epoch in range(4):
            produced_ids = []
            for i in range(batch_gens['n_val']):
                batch = next(val_loader)
                produced_ids.append(batch["pid"])
            uni, cts = np.unique(np.concatenate(produced_ids), return_counts=True)
            assert np.all(cts < 3), "with batch size one: every item should occur exactly once.\n uni {}, cts {}".format(
                uni[cts>2], cts[cts>2])
            #assert len(np.setdiff1d(val_loader.generator.dataset_pids, uni))==0, "not all val pids were shown."
            assert len(np.setdiff1d(uni, val_loader.generator.dataset_pids))==0, "pids shown that are not val set. impossible?"

        cf.n_workers = os.cpu_count()
        cf.batch_size = int(val_loader.generator.dataset_length / cf.n_workers) + 2
        val_loader = dloader.create_data_gen_pipeline(cf, val_loader.generator._data, do_aug=False, sample_pids_w_replace=False,
                                                             max_batches=None, raise_stop_iteration=True)
        for epoch in range(2):
            produced_ids = []
            for b, batch in enumerate(val_loader):
                produced_ids.append(batch["pid"])
            uni, cts = np.unique(np.concatenate(produced_ids), return_counts=True)
            assert np.all(cts == 1), "with batch size one: every item should occur exactly once.\n uni {}, cts {}".format(
                uni[cts>1], cts[cts>1])
            assert len(np.setdiff1d(val_loader.generator.dataset_pids, uni))==0, "not all val pids were shown."
            assert len(np.setdiff1d(uni, val_loader.generator.dataset_pids))==0, "pids shown that are not val set. impossible?"




        pass


if __name__=="__main__":
    stime = time.time()

    t = CheckRoIAlignImplementation()
    t.manual_check()
    #unittest.main()

    mins, secs = divmod((time.time() - stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))