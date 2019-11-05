"""
Created at 28.05.19 16:46
@author: gregor 
"""

import os
import sys
import subprocess

import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict

import utils.exp_utils as utils

def get_cf(dataset_name, exp_dir=""):

    cf_path = os.path.join('datasets', dataset_name, exp_dir, "configs.py")
    cf_file = utils.import_module('configs', cf_path)

    return cf_file.Configs()

def vector(item):
    """ensure item is vector-like (list or array or tuple)
    :param item: anything
    """
    if not isinstance(item, (list, tuple, np.ndarray)):
        item = [item]
    return item

def load_dataset(cf, subset_ixs=None):
    """
    loads the dataset. if deployed in cloud also copies and unpacks the data to the working directory.
    :param subset_ixs: subset indices to be loaded from the dataset. used e.g. for testing to only load the test folds.
    :return: data: dictionary with one entry per patient (in this case per patient-breast, since they are treated as
    individual images for training) each entry is a dictionary containing respective meta-info as well as paths to the preprocessed
    numpy arrays to be loaded during batch-generation
    """

    p_df = pd.read_pickle(os.path.join(cf.pp_data_path, cf.input_df_name))

    exclude_pids = ["0305a", "0447a"] # due to non-bg segmentation but bg mal label in nodules 5728, 8840
    p_df = p_df[~p_df.pid.isin(exclude_pids)]

    if cf.select_prototype_subset is not None:
        prototype_pids = p_df.pid.tolist()[:cf.select_prototype_subset]
        p_df = p_df[p_df.pid.isin(prototype_pids)]
        logger.warning('WARNING: using prototyping data subset!!!')
    if subset_ixs is not None:
        subset_pids = [np.unique(p_df.pid.tolist())[ix] for ix in subset_ixs]
        p_df = p_df[p_df.pid.isin(subset_pids)]

        print('subset: selected {} instances from df'.format(len(p_df)))

    pids = p_df.pid.tolist()
    cf.data_dir = cf.pp_data_path


    imgs = [os.path.join(cf.data_dir, '{}_img.npy'.format(pid)) for pid in pids]
    segs = [os.path.join(cf.data_dir,'{}_rois.npz'.format(pid)) for pid in pids]
    orig_class_targets = p_df['class_target'].tolist()

    data = OrderedDict()
    for ix, pid in enumerate(pids):
        data[pid] = {'data': imgs[ix], 'seg': segs[ix], 'pid': pid}
        data[pid]['fg_slices'] = np.array(p_df['fg_slices'].tolist()[ix])
        if 'class' in cf.prediction_tasks:
            # malignancy scores are binarized: (benign: 1-2 --> cl 1, malignant: 3-5 --> cl 2)
            raise NotImplementedError
            # todo need to consider bg
            data[pid]['class_targets'] = np.array([ [2 if ii >= 3 else 1 for ii in four_fold_targs] for four_fold_targs in orig_class_targets[ix]])
        else:
            data[pid]['class_targets'] = np.array([ [1 if ii>0 else 0 for ii in four_fold_targs] for four_fold_targs in orig_class_targets[ix]], dtype='uint8')
        if any(['regression' in task for task in cf.prediction_tasks]):
            data[pid]["regression_targets"] = np.array([ [vector(v) for v in four_fold_targs] for four_fold_targs in orig_class_targets[ix] ], dtype='float16')
            data[pid]["rg_bin_targets"] = np.array([ [cf.rg_val_to_bin_id(v) for v in four_fold_targs] for four_fold_targs in data[pid]["regression_targets"]], dtype='uint8')

    cf.roi_items = cf.observables_rois[:]
    cf.roi_items += ['class_targets']
    if any(['regression' in task for task in cf.prediction_tasks]):
        cf.roi_items += ['regression_targets']
        cf.roi_items += ['rg_bin_targets']

    return data


def get_patient_identifiers(cf, fold_lists):


    all_data = load_dataset(cf)
    all_pids_list = np.unique([v['pid'] for (k, v) in all_data.items()])


    verifier = [] #list of folds
    for fold in range(cf.n_cv_splits):
        train_ix, val_ix, test_ix, fold_nr = fold_lists[fold]
        assert fold==fold_nr
        test_ids = [all_pids_list[ix] for ix in test_ix]
        for ix, arr in enumerate(verifier):
            inter = np.intersect1d(test_ids, arr)
            #print("intersect of fold {} with fold {}: {}".format(fold, ix, inter))
            assert len(inter)==0
        verifier.append(test_ids)


    return verifier

def convert_folds_ids(exp_dir):
    import inference_analysis
    cf = get_cf('lidc', exp_dir=exp_dir)
    cf.exp_dir = exp_dir
    with open(os.path.join(exp_dir, 'fold_ids.pickle'), 'rb') as f:
        fids = pickle.load(f)

    pid_fold_splits = get_patient_identifiers(cf, fids)

    with open(os.path.join(exp_dir, 'fold_real_ids.pickle'), 'wb') as handle:
        pickle.dump(pid_fold_splits, handle)


    #inference_analysis.find_pid_in_splits('0811a', exp_dir=exp_dir)
    return


def copy_to_new_exp_dir(old_dir, new_dir):


    cp_ids = r"rsync {} {}".format(os.path.join(old_dir, 'fold_real_ids.pickle'), new_dir)
    rn_ids = "mv {} {}".format(os.path.join(new_dir, 'fold_real_ids.pickle'), os.path.join(new_dir, 'fold_ids.pickle'))
    cp_params = r"""rsync -a --include='*/' --include='*best_params.pth' --exclude='*' --prune-empty-dirs  
    {}  {}""".format(old_dir, new_dir)
    cp_ranking = r"""rsync -a --include='*/' --include='epoch_ranking.npy' --exclude='*' --prune-empty-dirs  
        {}  {}""".format(old_dir, new_dir)
    cp_results = r"""rsync -a --include='*/' --include='pred_results.pkl' --exclude='*' --prune-empty-dirs  
        {}  {}""".format(old_dir, new_dir)

    for cmd in  [cp_ids, rn_ids, cp_params, cp_ranking, cp_results]:
        subprocess.call(cmd, shell=True)
    print("Setup {} for inference with ids, params from {}".format(new_dir, old_dir))



if __name__=="__main__":
    exp_dir = '/home/gregor/networkdrives/E132-Cluster-Projects/lidc_sa/experiments/ms12345_mrcnn3d_rgbin_bs8'
    new_exp_dir = '/home/gregor/Documents/medicaldetectiontoolkit/datasets/lidc/experiments/ms12345_mrcnn3d_rgbin_copiedparams'
    #convert_folds_ids(exp_dir)
    copy_to_new_exp_dir(exp_dir, new_exp_dir)