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

'''
This preprocessing script loads nrrd files obtained by the data conversion tool: https://github.com/MIC-DKFZ/LIDC-IDRI-processing/tree/v1.0.1
After applying preprocessing, images are saved as numpy arrays and the meta information for the corresponding patient is stored
as a line in the dataframe saved as info_df.pickle.
'''

import os
import sys
import argparse
import shutil
import subprocess
import pickle
import time

import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool
import pandas as pd
import numpy.testing as npt
from skimage.transform import resize

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append('../..')
import data_manager as dmanager

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def load_df(path):
    df = pd.read_pickle(path)
    print(df)

    return

def resample_array(src_imgs, src_spacing, target_spacing):
    """ Resample a numpy array.
    :param src_imgs: source image.
    :param src_spacing: source image's spacing.
    :param target_spacing: spacing to resample source image to.
    :return:
    """
    src_spacing = np.round(src_spacing, 3)
    target_shape = [int(src_imgs.shape[ix] * src_spacing[::-1][ix] / target_spacing[::-1][ix]) for ix in range(len(src_imgs.shape))]
    for i in range(len(target_shape)):
        try:
            assert target_shape[i] > 0
        except:
            raise AssertionError("AssertionError:", src_imgs.shape, src_spacing, target_spacing)

    img = src_imgs.astype('float64')
    resampled_img = resize(img, target_shape, order=1, clip=True, mode='edge').astype('float32')

    return resampled_img

class Preprocessor(object):
    """Preprocessor for LIDC raw data. Set in config: which ground truths to produce, choices are
        - "merged" for a single ground truth per input image, created by merging the given four rater annotations
            into one.
        - "single-annotator" for a four-fold ground truth per input image, created by leaving the each rater annotation
            separately.
    :param cf: config.
    :param exclude_inconsistents: bool or tuple, list, np.array, exclude patients that show technical inconsistencies
        in the raw files, likely due to file-naming mistakes. if bool and True: search for patients that have too many
        ratings per lesion or other inconstencies, exclude findings. if param is list of pids: exclude given pids.
    :param overwrite: look for patients that already exist in the pp dir. if overwrite is False, do not redo existing
        patients, otherwise ignore any existing files.
    :param max_count: maximum number of patients to preprocess.
    :param pids_subset: subset of pids to preprocess.
    """

    def __init__(self, cf, exclude_inconsistents=True, overwrite=False, max_count=None, pids_subset=None):

        self.cf = cf

        assert len(self.cf.gts_to_produce)>0, "need to specify which gts to produce, choices: 'merged', 'single_annotator'"

        self.paths = [os.path.join(cf.raw_data_dir, ii) for ii in os.listdir(cf.raw_data_dir)]
        if exclude_inconsistents:
            if isinstance(exclude_inconsistents, bool):
                exclude_paths = self.exclude_too_many_ratings()
                exclude_paths += self.verify_seg_label_pairings()
            else:
                assert isinstance(exclude_inconsistents, (tuple,list,np.ndarray))
                exclude_paths = exclude_inconsistents
            self.paths = [path for path in self.paths if path not in exclude_paths]


        if 'single_annotator' in self.cf.gts_to_produce or 'sa' in self.cf.gts_to_produce:
            self.pp_dir_sa = os.path.join(cf.pp_dir, "patient_gts_sa")
        if 'merged' in self.cf.gts_to_produce:
            self.pp_dir_merged = os.path.join(cf.pp_dir, "patient_gts_merged")
        orig_count = len(self.paths)
        # check if some patients already have ppd versions in destination dir
        if os.path.exists(cf.pp_dir) and not overwrite:
            fs_in_dir = os.listdir(cf.pp_dir)
            already_done =  [file.split("_")[0] for file in fs_in_dir if file.split("_")[1] == "img.npy"]
            if 'single_annotator' in self.cf.gts_to_produce or 'sa' in self.cf.gts_to_produce:
                ext = '.npy' if hasattr(self.cf, "save_sa_segs_as") and (
                            self.cf.save_sa_segs_as == "npy" or self.cf.save_sa_segs_as == ".npy") else '.npz'
                fs_in_dir = os.listdir(self.pp_dir_sa)
                already_done = [ pid for pid in already_done if pid+"_rois"+ext in fs_in_dir and pid+"_meta_info.pickle" in fs_in_dir]
            if 'merged' in self.cf.gts_to_produce:
                fs_in_dir = os.listdir(self.pp_dir_merged)
                already_done = [pid for pid in already_done if
                                pid + "_rois.npy" in fs_in_dir and pid+"_meta_info.pickle" in fs_in_dir]

            self.paths = [p for p in self.paths if not p.split(os.sep)[-1] in already_done]
            if len(self.paths)!=orig_count:
                print("Due to existing ppd files: Selected a subset of {} patients from originally {}".format(len(self.paths), orig_count))

        if pids_subset:
            self.paths = [p for p in self.paths if p.split(os.sep)[-1] in pids_subset]
        if max_count is not None:
            self.paths = self.paths[:max_count]

        if not os.path.exists(cf.pp_dir):
            os.mkdir(cf.pp_dir)
        if ('single_annotator' in self.cf.gts_to_produce or 'sa' in self.cf.gts_to_produce) and \
                not os.path.exists(self.pp_dir_sa):
            os.mkdir(self.pp_dir_sa)
        if 'merged' in self.cf.gts_to_produce and not os.path.exists(self.pp_dir_merged):
            os.mkdir(self.pp_dir_merged)


    def exclude_too_many_ratings(self):
        """exclude a patient's full path (the patient folder) from further processing if patient has nodules with
            ratings of more than four raters (which is inconsistent with what the raw data is supposed to comprise,
            also rater ids appear multiple times on the same nodule in these cases motivating the assumption that
            the same rater issued more than one rating / mixed up files or annotations for a nodule).
        :return: paths to be excluded.
        """
        exclude_paths = []
        for path in self.paths:
            roi_ids = set([ii.split('.')[0].split('_')[-1] for ii in os.listdir(path) if '.nii.gz' in ii])
            found = False
            for roi_id in roi_ids:
                n_raters = len([ii for ii in os.listdir(path) if '{}.nii'.format(roi_id) in ii])
                # assert n_raters<=4, "roi {} in path {} has {} raters".format(roi_id, path, n_raters)
                if n_raters > 4:
                    print("roi {} in path {} has {} raters".format(roi_id, path, n_raters))
                    found = True
            if found:
                exclude_paths.append(path)
        print("Patients excluded bc of too many raters:\n")
        for p in exclude_paths:
            print(p)
        print()

        return exclude_paths

    def analyze_lesion(self, pid, nodule_id):
        """print unique seg and counts of nodule nodule_id of patient pid.
        """
        nodule_id = nodule_id.lstrip("0")
        nodule_id_paths = [ii for ii in os.listdir(os.path.join(self.cf.raw_data_dir, pid)) if '.nii' in ii]
        nodule_id_paths = [ii for ii in nodule_id_paths if ii.split('_')[2].lstrip("0")==nodule_id]
        assert len(nodule_id_paths)==1
        nodule_path = nodule_id_paths[0]

        roi = sitk.ReadImage(os.path.join(self.cf.raw_data_dir, pid, nodule_path))
        roi_arr = sitk.GetArrayFromImage(roi).astype(np.uint8)

        print("pid {}, nodule {}, unique seg & counts: {}".format(pid, nodule_id, np.unique(roi_arr, return_counts=True)))
        return

    def verify_seg_label_pairing(self, path):
        """verifies that a nodule's segmentation has malignancy label > 0 if segmentation has foreground (>0 anywhere),
            and vice-versa that it has only background (==0 everywhere) if no malignancy label (==label 0) assigned.
        :param path: path to the patient folder.
        :return: df containing eventual inconsistency findings.
        """

        pid = path.split('/')[-1]

        df = pd.read_csv(os.path.join(self.cf.root_dir, 'characteristics.csv'), sep=';')
        df = df[df.PatientID == pid]

        findings_df = pd.DataFrame(columns=["problem", "pid", "roi_id", "nodule_id", "rater_ix", "seg_unique", "label"])

        print('verifying {}'.format(pid))

        roi_ids = set([ii.split('.')[0].split('_')[-1] for ii in os.listdir(path) if '.nii.gz' in ii])

        for roi_id in roi_ids:
            roi_id_paths = [ii for ii in os.listdir(path) if '{}.nii'.format(roi_id) in ii]
            nodule_ids = [rp.split('_')[2].lstrip("0") for rp in roi_id_paths]
            rater_ids = [rp.split('_')[1] for rp in roi_id_paths]
            rater_labels = [df[df.NoduleID == int(ii)].Malignancy.values[0] for ii in nodule_ids]

            # check double existence of nodule ids
            uniq, counts = np.unique(nodule_ids, return_counts=True)
            if np.any([count>1 for count in counts]):
                finding = ("same nodule id exists more than once", pid, roi_id, nodule_ids, "N/A", "N/A", "N/A")
                print("not unique nodule id", finding)
                findings_df.loc[findings_df.shape[0]] = finding

            # check double gradings of single rater for single roi
            uniq, counts = np.unique(rater_ids, return_counts=True)
            if np.any([count>1 for count in counts]):
                finding = ("same roi_id exists more than once for a single rater", pid, roi_id, nodule_ids, rater_ids, "N/A", rater_labels)
                print("more than one grading per roi per single rater", finding)
                findings_df.loc[findings_df.shape[0]] = finding


            rater_segs = []
            for rp in roi_id_paths:
                roi = sitk.ReadImage(os.path.join(self.cf.raw_data_dir, pid, rp))
                roi_arr = sitk.GetArrayFromImage(roi).astype(np.uint8)

                rater_segs.append(roi_arr)
            rater_segs = np.array(rater_segs)
            for r in range(rater_segs.shape[0]):
                if np.sum(rater_segs[r])>0:
                    if rater_labels[r]<=0:
                        finding =  ("non-empty seg w/ bg label", pid, roi_id, nodule_ids[r], rater_ids[r], np.unique(rater_segs[r]), rater_labels[r])
                        print("{}: pid {}, nodule {}, rater {}, seg unique {}, label {}".format(
                            *finding))
                        findings_df.loc[findings_df.shape[0]] = finding
                else:
                    if rater_labels[r]>0:
                        finding = ("empty seg w/ fg label", pid, roi_id, nodule_ids[r], rater_ids[r], np.unique(rater_segs[r]), rater_labels[r])
                        print("{}: pid {}, nodule {}, rater {}, seg unique {}, label {}".format(
                            *finding))
                        findings_df.loc[findings_df.shape[0]] = finding

        return findings_df

    def verify_seg_label_pairings(self, processes=os.cpu_count()):
        """wrapper to multi-process verification of seg-label pairings.
        """

        pool = Pool(processes=processes)
        findings_dfs = pool.map(self.verify_seg_label_pairing, self.paths, chunksize=1)
        pool.close()
        pool.join()

        findings_df = pd.concat(findings_dfs, axis=0)
        findings_df.to_pickle(os.path.join(self.cf.pp_dir, "verification_seg_label_pairings.pickle"))
        findings_df.to_csv(os.path.join(self.cf.pp_dir, "verification_seg_label_pairings.csv"))

        return findings_df.pid.tolist()

    def produce_sa_gt(self, path, pid, df, img_spacing, img_arr_shape):
        """ Keep annotations separate, i.e., every processed image has four final GTs.
            Images are always saved as npy. For meeting hard-disk-memory constraints, segmentations can optionally be
            saved as .npz instead of .npy. Dataloader is only implemented for reading .npz segs.
        """

        final_rois = np.zeros((4, *img_arr_shape), dtype='uint8')
        patient_mal_labels = []
        roi_ids = list(set([ii.split('.')[0].split('_')[-1] for ii in os.listdir(path) if '.nii.gz' in ii]))
        roi_ids.sort() # just a precaution to have same order of lesions throughout separate runs

        rix = 1
        for roi_id in roi_ids:
            roi_id_paths = [ii for ii in os.listdir(path) if '{}.nii'.format(roi_id) in ii]
            assert len(roi_id_paths)>0 and len(roi_id_paths)<=4, "pid {}: should find 0< n_rois <4, but found {}".format(pid, len(roi_id_paths))

            """ not strictly necessary precaution: in theory, segmentations of different raters could overlap also for 
                *different* rois, i.e., a later roi of a rater could (partially) cover up / destroy the roi of another 
                rater. practically this is unlikely as overlapping lesions of different raters should be regarded as the
                same lesion, but safety first. hence, the order of raters is maintained across rois, i.e., rater 0 
                (marked as rater 0 in roi's file name) always has slot 0 in rater_labels and rater_segs, thereby rois
                are certain to not overlap.
            """
            rater_labels, rater_segs = np.zeros((4,), dtype='uint8'), np.zeros((4,*img_arr_shape), dtype="float32")
            for ix, rp in enumerate(roi_id_paths): # one roi path per rater
                nodule_id = rp.split('_')[2].lstrip("0")
                assert not (nodule_id=="5728" or nodule_id=="8840"), "nodule ids {}, {} should be excluded due to seg-mal-label inconsistency.".format(5728, 8840)
                rater = int(rp.split('_')[1])
                rater_label = df[df.NoduleID == int(nodule_id)].Malignancy.values[0]
                rater_labels[rater] = rater_label

                roi = sitk.ReadImage(os.path.join(self.cf.raw_data_dir, pid, rp))
                for dim in range(len(img_arr_shape)):
                    npt.assert_almost_equal(roi.GetSpacing()[dim], img_spacing[dim])
                roi_arr = sitk.GetArrayFromImage(roi)
                roi_arr = resample_array(roi_arr, roi.GetSpacing(), self.cf.target_spacing)
                assert roi_arr.shape == img_arr_shape, [roi_arr.shape, img_arr_shape, pid, roi.GetSpacing()]
                assert not np.any(rater_segs[rater]), "overwriting existing rater's seg with roi {}".format(rp)
                rater_segs[rater] = roi_arr
            rater_segs = np.array(rater_segs)

            # rename/remap the malignancy to be positive.
            roi_mal_labels = [ii if ii > -1 else 0 for ii in rater_labels]
            assert rater_segs.shape == final_rois.shape, "rater segs shape {}, final rois shp {}".format(rater_segs.shape, final_rois.shape)

            # assert non-zero rating has non-zero seg
            for rater in range(4):
                if roi_mal_labels[rater]>0:
                    assert np.any(rater_segs[rater]>0), "rater {} mal label {} but uniq seg {}".format(rater, roi_mal_labels[rater], np.unique(rater_segs[rater]))

            # add the roi to patient. i.e., write current lesion into final labels and seg of whole patient.
            assert np.any(rater_segs), "empty segmentations for all raters should not exist in single-annotator mode, pid {}, rois: {}".format(pid, roi_id_paths)
            patient_mal_labels.append(roi_mal_labels)
            final_rois[rater_segs > 0] = rix
            rix += 1


        fg_slices = [[ii for ii in np.unique(np.argwhere(final_rois[r] != 0)[:, 0])] for r in range(4)]
        patient_mal_labels = np.array(patient_mal_labels)
        roi_ids = np.unique(final_rois[final_rois>0])
        assert len(roi_ids) == len(patient_mal_labels), "mismatch {} rois in seg, {} rois in mal labels".format(len(roi_ids), len(patient_mal_labels))

        if hasattr(self.cf, "save_sa_segs_as") and (self.cf.save_sa_segs_as=="npy" or self.cf.save_sa_segs_as==".npy"):
            np.save(os.path.join(self.pp_dir_sa, '{}_rois.npy'.format(pid)), final_rois)
        else:
            np.savez_compressed(os.path.join(self.cf.pp_dir, 'patient_gts_sa', '{}_rois.npz'.format(pid)), seg=final_rois)
        with open(os.path.join(self.pp_dir_sa, '{}_meta_info.pickle'.format(pid)), 'wb') as handle:
            meta_info_dict = {'pid': pid, 'class_target': patient_mal_labels, 'spacing': img_spacing,
                              'fg_slices': fg_slices}
            pickle.dump(meta_info_dict, handle)

    def produce_merged_gt(self, path, pid, df, img_spacing, img_arr_shape):
        """ process patient with merged annotations, i.e., only one final GT per image. save img and seg to npy, rest to
            metadata.
            annotations merging:
                - segmentations: only regard a pixel as foreground if at least two raters found it be foreground.
                - malignancy labels: average over all four rater votes. every rater who did not assign a finding or
                    assigned -1 to the RoI contributes to the average with a vote of 0.

        :param path: path to patient folder.
        """

        final_rois = np.zeros(img_arr_shape, dtype=np.uint8)
        patient_mal_labels = []
        roi_ids = set([ii.split('.')[0].split('_')[-1] for ii in os.listdir(path) if '.nii.gz' in ii])

        rix = 1
        for roi_id in roi_ids:
            roi_id_paths = [ii for ii in os.listdir(path) if '{}.nii'.format(roi_id) in ii]
            nodule_ids = [ii.split('_')[2].lstrip("0") for ii in roi_id_paths]
            rater_labels = [df[df.NoduleID == int(ii)].Malignancy.values[0] for ii in nodule_ids]
            rater_labels.extend([0] * (4 - len(rater_labels)))
            mal_label = np.mean([ii if ii > -1 else 0 for ii in rater_labels])
            rater_segs = []
            for rp in roi_id_paths:
                roi = sitk.ReadImage(os.path.join(self.cf.raw_data_dir, pid, rp))
                for dim in range(len(img_arr_shape)):
                    npt.assert_almost_equal(roi.GetSpacing()[dim], img_spacing[dim])
                roi_arr = sitk.GetArrayFromImage(roi).astype(np.uint8)
                roi_arr = resample_array(roi_arr, roi.GetSpacing(), self.cf.target_spacing)
                assert roi_arr.shape == img_arr_shape, [roi_arr.shape, img_arr_shape, pid, roi.GetSpacing()]
                rater_segs.append(roi_arr)
            rater_segs.extend([np.zeros_like(rater_segs[-1])] * (4 - len(roi_id_paths)))
            rater_segs = np.mean(np.array(rater_segs), axis=0)
            # annotations merging: if less than two raters found fg, set segmentation to bg.
            rater_segs[rater_segs < 0.5] = 0
            if np.sum(rater_segs) > 0:
                patient_mal_labels.append(mal_label)
                final_rois[rater_segs > 0] = rix
                rix += 1
            else:
                # indicate rois suppressed by majority voting of raters
                print('suppressed roi!', roi_id_paths)
                with open(os.path.join(self.pp_dir_merged, 'suppressed_rois.txt'), 'a') as handle:
                    handle.write(" ".join(roi_id_paths))

        fg_slices = [ii for ii in np.unique(np.argwhere(final_rois != 0)[:, 0])]
        patient_mal_labels = np.array(patient_mal_labels)
        assert len(patient_mal_labels) + 1 == len(np.unique(final_rois)), [len(patient_mal_labels), np.unique(final_rois), pid]
        assert final_rois.dtype == 'uint8'
        np.save(os.path.join(self.pp_dir_merged, '{}_rois.npy'.format(pid)), final_rois)

        with open(os.path.join(self.pp_dir_merged, '{}_meta_info.pickle'.format(pid)), 'wb') as handle:
            meta_info_dict = {'pid': pid, 'class_target': patient_mal_labels, 'spacing': img_spacing,
                              'fg_slices': fg_slices}
            pickle.dump(meta_info_dict, handle)

    def pp_patient(self, path):

        pid = path.split('/')[-1]
        img = sitk.ReadImage(os.path.join(path, '{}_ct_scan.nrrd'.format(pid)))
        img_arr = sitk.GetArrayFromImage(img)
        print('processing {} with GT(s) {}, spacing {} and img shape {}.'.format(
            pid, " and ".join(self.cf.gts_to_produce), img.GetSpacing(), img_arr.shape))
        img_arr = resample_array(img_arr, img.GetSpacing(), self.cf.target_spacing)
        img_arr = np.clip(img_arr, -1200, 600)
        #img_arr = (1200 + img_arr) / (600 + 1200) * 255  # a+x / (b-a) * (c-d) (c, d = new)
        img_arr = img_arr.astype(np.float32)
        img_arr = (img_arr - np.mean(img_arr)) / np.std(img_arr).astype('float16')

        df = pd.read_csv(os.path.join(self.cf.root_dir, 'characteristics.csv'), sep=';')
        df = df[df.PatientID == pid]

        np.save(os.path.join(self.cf.pp_dir, '{}_img.npy'.format(pid)), img_arr)
        if 'single_annotator' in self.cf.gts_to_produce or 'sa' in self.cf.gts_to_produce:
            self.produce_sa_gt(path, pid, df, img.GetSpacing(), img_arr.shape)
        if 'merged' in self.cf.gts_to_produce:
            self.produce_merged_gt(path, pid, df, img.GetSpacing(), img_arr.shape)


    def iterate_patients(self, processes=os.cpu_count()):
        pool = Pool(processes=processes)
        pool.map(self.pp_patient, self.paths, chunksize=1)
        pool.close()
        pool.join()
        print("finished processing raw patient data")


    def aggregate_meta_info(self):
        self.dfs = {}
        for gt_kind in self.cf.gts_to_produce:
            kind_dir = self.pp_dir_merged if gt_kind == "merged" else self.pp_dir_sa
            files = [os.path.join(kind_dir, f) for f in os.listdir(kind_dir) if 'meta_info.pickle' in f]
            self.dfs[gt_kind] = pd.DataFrame(columns=['pid', 'class_target', 'spacing', 'fg_slices'])
            for f in files:
                with open(f, 'rb') as handle:
                    self.dfs[gt_kind].loc[len(self.dfs[gt_kind])] = pickle.load(handle)

            self.dfs[gt_kind].to_pickle(os.path.join(kind_dir, 'info_df.pickle'))
            print("aggregated meta info to df with length", len(self.dfs[gt_kind]))

    def convert_copy_npz(self):
        npz_dir = os.path.join(self.cf.pp_dir+'_npz')
        print("converting to npz dir", npz_dir)
        os.makedirs(npz_dir, exist_ok=True)

        dmanager.pack_dataset(self.cf.pp_dir, destination=npz_dir, recursive=True, verbose=False)
        if hasattr(self, 'pp_dir_merged'):
            subprocess.call('rsync -avh --exclude="*.npy" {} {}'.format(self.pp_dir_merged, npz_dir), shell=True)
        if hasattr(self, 'pp_dir_sa'):
            subprocess.call('rsync -avh --exclude="*.npy" {} {}'.format(self.pp_dir_sa, npz_dir), shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', type=int, default=None, help='How many patients to maximally process.')
    args = parser.parse_args()
    total_stime = time.time()

    import configs
    cf = configs.Configs()

    # analysis finding: the following patients have unclear annotations. some raters gave more than one judgement
    # on the same roi.
    patients_to_exclude = ["0137a", "0404a", "0204a", "0252a", "0366a", "0863a", "0815a", "0060a", "0249a", "0436a", "0865a"]
    # further finding: the following patients contain nodules with segmentation-label inconsistencies
    # running Preprocessor.verify_seg_label_pairings() produces a data frame with detailed findings.
    patients_to_exclude += ["0305a", "0447a"]
    exclude_paths = [os.path.join(cf.raw_data_dir, pid) for pid in patients_to_exclude]
    # These pids are automatically found and excluded, when setting exclude_inconsistents=True at Preprocessor
    # initialization instead of passing the pre-compiled list.


    pp = Preprocessor(cf, overwrite=True, exclude_inconsistents=exclude_paths, max_count=args.number, pids_subset=None)#["0998a"])
    #pp.analyze_lesion("0305a", "5728")
    #pp.analyze_lesion("0305a", "5741")
    #pp.analyze_lesion("0447a", "8840")

    #pp.verify_seg_label_pairings()
    #load_df(os.path.join(cf.pp_dir, "verification_seg_label_pairings.pickle"))
    pp.iterate_patients(processes=8)
    # for i in ["/mnt/E130-Personal/Goetz/Datenkollektive/Lungendaten/Nodules_LIDC_IDRI/new_nrrd/0305a",
    #           "/mnt/E130-Personal/Goetz/Datenkollektive/Lungendaten/Nodules_LIDC_IDRI/new_nrrd/0447a"]:  #pp.paths[:1]:
    #      pp.pp_patient(i)
    pp.aggregate_meta_info()
    pp.convert_copy_npz()



    mins, secs = divmod((time.time() - total_stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))
