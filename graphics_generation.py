"""
Created at 07/03/19 11:42
@author: gregor 
"""
import plotting as plg
import matplotlib.lines as mlines

import os
import sys
import multiprocessing
from copy import deepcopy
import logging
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import confusion_matrix

import utils.exp_utils as utils
import utils.model_utils as mutils
import utils.dataloader_utils as dutils
from utils.dataloader_utils import ConvertSegToBoundingBoxCoordinates

import predictor as predictor_file
import evaluator as evaluator_file



class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonProcessPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def get_cf(dataset_name, exp_dir=""):

    cf_path = os.path.join('datasets', dataset_name, exp_dir, "configs.py")
    cf_file = utils.import_module('configs', cf_path)

    return cf_file.Configs()


def prostate_results_static(plot_dir=None):
    cf = get_cf('prostate', '')
    if plot_dir is None:
        plot_dir = os.path.join('datasets', 'prostate', 'misc')

    text_fs = 18
    fig = plg.plt.figure(figsize=(6, 3)) #w,h
    grid = plg.plt.GridSpec(1, 1, wspace=0.0, hspace=0.0, figure=fig) #r,c

    groups = ["b values", "ADC + b values", "T2"]
    splits = ["Det. U-Net", "Mask R-CNN", "Faster R-CNN+"]
    values = {"detu": [(0.296, 0.031), (0.312, 0.045), (0.090, 0.040)],
              "mask": [(0.393, 0.051), (0.382, 0.047), (0.136, 0.016)],
              "fast": [(0.424, 0.083), (0.390, 0.086), (0.036, 0.013)]}
    bar_values = [[v[0] for v in split] for split in values.values()]
    errors = [[v[1] for v in split] for split in values.values()]
    ax = fig.add_subplot(grid[0,0])
    colors = [cf.aubergine, cf.blue, cf.dark_blue]
    plg.plot_grouped_bar_chart(cf, bar_values, groups, splits, errors=errors, colors=colors, ax=ax, legend=True,
                               title="Prostate Main Results (3D)", ylabel=r"Performance as $\mathrm{AP}_{10}$", xlabel="Input Modalities")
    plg.plt.tight_layout()
    plg.plt.savefig(os.path.join(plot_dir, 'prostate_main_results.png'), dpi=600)

def prostate_GT_examples(exp_dir='', plot_dir=None, pid=8., z_ix=None):

    import datasets.prostate.data_loader as dl
    cf = get_cf('prostate', exp_dir)
    cf.exp_dir = exp_dir
    cf.fold = 0
    cf.data_sourcedir =  "/mnt/HDD2TB/Documents/data/prostate/data_di_250519_ps384_gs6071/"
    dataset = dl.Dataset(cf)
    dataset.init_FoldGenerator(cf.seed, cf.n_cv_splits)
    dataset.generate_splits(check_file=os.path.join(cf.exp_dir, 'fold_ids.pickle'))
    set_splits = dataset.fg.splits

    test_ids, val_ids = set_splits.pop(cf.fold), set_splits.pop(cf.fold - 1)
    train_ids = np.concatenate(set_splits, axis=0)

    if cf.hold_out_test_set:
        train_ids = np.concatenate((train_ids, test_ids), axis=0)
        test_ids = []
    print("data set loaded with: {} train / {} val / {} test patients".format(len(train_ids), len(val_ids),
                                                                                    len(test_ids)))


    if plot_dir is None:
        plot_dir = cf.plot_dir if hasattr(cf, 'plot_dir') else os.path.join('datasets', 'prostate', 'misc')

    text_fs = 18
    fig = plg.plt.figure(figsize=(10, 7.7)) #w,h
    grid = plg.plt.GridSpec(3, 4, wspace=0.0, hspace=0.0, figure=fig) #r,c
    text_x, text_y = 0.1, 0.8

    # ------- DWI -------
    if z_ix is None:
        z_ix_dwi = np.random.choice(dataset[pid]["fg_slices"])
    img = np.load(dataset[pid]["img"])[:,z_ix_dwi] # mods, z,y,x
    seg = np.load(dataset[pid]["seg"])[z_ix_dwi] # z,y,x
    ax = fig.add_subplot(grid[0,0])
    ax.imshow(img[0], cmap='gray')
    ax.text(text_x, text_y, "ADC", size=text_fs, color=cf.white, transform=ax.transAxes,
          bbox=dict(facecolor=cf.black, alpha=0.7, edgecolor=cf.white, clip_on=False, pad=7))
    ax.axis('off')
    ax = fig.add_subplot(grid[0,1])
    ax.imshow(img[0], cmap='gray')
    cmap = cf.class_cmap
    for r_ix in np.unique(seg[seg>0]):
        seg[seg==r_ix] = dataset[pid]["class_targets"][r_ix-1]
    ax.imshow(plg.to_rgba(seg, cmap), alpha=1)
    ax.text(text_x, text_y, "DWI GT", size=text_fs, color=cf.white, transform=ax.transAxes,
          bbox=dict(facecolor=cf.black, alpha=0.7, edgecolor=cf.white, clip_on=False, pad=7))
    ax.axis('off')
    for b_ix, b in enumerate([50,500,1000,1500]):
        ax = fig.add_subplot(grid[1, b_ix])
        ax.imshow(img[b_ix+1], cmap='gray')
        ax.text(text_x, text_y, r"{}{}".format("$b=$" if b_ix == 0 else "", b), size=text_fs, color=cf.white,
                transform=ax.transAxes,
                bbox=dict(facecolor=cf.black, alpha=0.7, edgecolor=cf.white, clip_on=False, pad=7))
        ax.axis('off')

    # ----- T2 -----
    cf.data_sourcedir = "/mnt/HDD2TB/Documents/data/prostate/data_t2_250519_ps384_gs6071/"
    dataset = dl.Dataset(cf)
    if z_ix is None:
        if z_ix_dwi in dataset[pid]["fg_slices"]:
            z_ix_t2 = z_ix_dwi
        else:
            z_ix_t2 = np.random.choice(dataset[pid]["fg_slices"])
    img = np.load(dataset[pid]["img"])[:,z_ix_t2] # mods, z,y,x
    seg = np.load(dataset[pid]["seg"])[z_ix_t2] # z,y,x
    ax = fig.add_subplot(grid[2,0])
    ax.imshow(img[0], cmap='gray')
    ax.text(text_x, text_y, "T2w", size=text_fs, color=cf.white, transform=ax.transAxes,
          bbox=dict(facecolor=cf.black, alpha=0.7, edgecolor=cf.white, clip_on=False, pad=7))
    ax.axis('off')
    ax = fig.add_subplot(grid[2,1])
    ax.imshow(img[0], cmap='gray')
    cmap = cf.class_cmap
    for r_ix in np.unique(seg[seg>0]):
        seg[seg==r_ix] = dataset[pid]["class_targets"][r_ix-1]
    ax.imshow(plg.to_rgba(seg, cmap), alpha=1)
    ax.text(text_x, text_y, "T2 GT", size=text_fs, color=cf.white, transform=ax.transAxes,
          bbox=dict(facecolor=cf.black, alpha=0.7, edgecolor=cf.white, clip_on=False, pad=7))
    ax.axis('off')

    #grid.tight_layout(fig)
    plg.plt.tight_layout()
    plg.plt.savefig(os.path.join(plot_dir, 'prostate_gt_examples.png'), dpi=600)


def prostate_dataset_stats(exp_dir='', plot_dir=None, show_splits=True,):

    import datasets.prostate.data_loader as dl
    cf = get_cf('prostate', exp_dir)
    cf.exp_dir = exp_dir
    cf.fold = 0
    dataset = dl.Dataset(cf)
    dataset.init_FoldGenerator(cf.seed, cf.n_cv_splits)
    dataset.generate_splits(check_file=os.path.join(cf.exp_dir, 'fold_ids.pickle'))
    set_splits = dataset.fg.splits

    test_ids, val_ids = set_splits.pop(cf.fold), set_splits.pop(cf.fold - 1)
    train_ids = np.concatenate(set_splits, axis=0)

    if cf.hold_out_test_set:
        train_ids = np.concatenate((train_ids, test_ids), axis=0)
        test_ids = []

    print("data set loaded with: {} train / {} val / {} test patients".format(len(train_ids), len(val_ids),
                                                                                    len(test_ids)))

    df, labels = dataset.calc_statistics(subsets={"train": train_ids, "val": val_ids, "test": test_ids}, plot_dir=None)

    if plot_dir is None:
        plot_dir = cf.plot_dir if hasattr(cf, 'plot_dir') else os.path.join('datasets', 'prostate', 'misc')

    if show_splits:
        fig = plg.plt.figure(figsize=(6, 6)) # w, h
        grid = plg.plt.GridSpec(2, 2, wspace=0.05, hspace=0.15, figure=fig) # rows, cols
    else:
        fig = plg.plt.figure(figsize=(6, 3.))
        grid = plg.plt.GridSpec(1, 1, wspace=0.0, hspace=0.15, figure=fig)

    ax = fig.add_subplot(grid[0,0])
    ax = plg.plot_data_stats(cf, df, labels, ax=ax)
    ax.set_xlabel("")
    ax.set_xticklabels(df.columns, rotation='horizontal', fontsize=11)
    ax.set_title("")
    if show_splits:
        ax.text(0.05,0.95, 'a)', horizontalalignment='center', verticalalignment='center', transform = ax.transAxes, weight='bold')
    ax.text(0, 25, "GS$=6$", horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor=(*cf.white, 0.8), edgecolor=cf.dark_green, pad=3))
    ax.text(1, 25, "GS$\geq 7a$", horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor=(*cf.white, 0.8), edgecolor=cf.red, pad=3))
    ax.margins(y=0.1)

    if show_splits:
        ax = fig.add_subplot(grid[:, 1])
        ax = plg.plot_fold_stats(cf, df, labels, ax=ax)
        ax.set_xlabel("")
        ax.set_title("")
        ax.text(0.05, 0.98, 'c)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, weight='bold')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.margins(y=0.1)

        ax = fig.add_subplot(grid[1, 0])
        cf.balance_target = "lesion_gleasons"
        dataset.df = None
        df, labels = dataset.calc_statistics(plot_dir=None, overall_stats=True)
        ax = plg.plot_data_stats(cf, df, labels, ax=ax)
        ax.set_xlabel("")
        ax.set_title("")
        ax.text(0.05, 0.95, 'b)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, weight='bold')
        ax.margins(y=0.1)
        # rename GS according to names in thesis
        renamer = {'GS60':'GS 6', 'GS71':'GS 7a', 'GS72':'GS 7b', 'GS80':'GS 8', 'GS90': 'GS 9', 'GS91':'GS 9a', 'GS92':'GS 9b'}
        x_ticklabels = [str(l.get_text()) for l in ax.xaxis.get_ticklabels()]
        ax.xaxis.set_ticklabels([renamer[l] for l in x_ticklabels])

    plg.plt.tight_layout()
    plg.plt.savefig(os.path.join(plot_dir, 'data_stats_prostate.png'), dpi=600)

    return

def lidc_merged_sa_joint_plot(exp_dir='', plot_dir=None):
    import datasets.lidc.data_loader as dl
    cf = get_cf('lidc', exp_dir)
    cf.balance_target = "regression_targets"

    if plot_dir is None:
        plot_dir = cf.plot_dir if hasattr(cf, 'plot_dir') else os.path.join('datasets', 'lidc', 'misc')

    cf.training_gts = 'merged'
    dataset = dl.Dataset(cf, mode='train')
    df, labels = dataset.calc_statistics(plot_dir=None, overall_stats=True)

    fig = plg.plt.figure(figsize=(4, 5.6)) #w, h
    # fig.subplots_adjust(hspace=0, wspace=0)
    grid = plg.plt.GridSpec(3, 1, wspace=0.0, hspace=0.7, figure=fig) #rows, cols
    fs = 9

    ax = fig.add_subplot(grid[0, 0])

    labels = [AttributeDict({ 'name': rg_val, 'color': cf.bin_id2label[cf.rg_val_to_bin_id(rg_val)].color}) for rg_val
              in df.columns]
    ax = plg.plot_data_stats(cf, df, labels, ax=ax, fs=fs)
    ax.set_xlabel("averaged multi-rater malignancy scores (ms)", fontsize=fs)
    ax.set_title("")
    ax.text(0.05, 0.91, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            weight='bold', fontsize=fs)
    ax.margins(y=0.2)

    #----- single annotator -------
    cf.training_gts = 'sa'
    dataset = dl.Dataset(cf, mode='train')
    df, labels = dataset.calc_statistics(plot_dir=None, overall_stats=True)

    ax = fig.add_subplot(grid[1, 0])
    labels = [AttributeDict({ 'name': '{:.0f}'.format(rg_val), 'color': cf.bin_id2label[cf.rg_val_to_bin_id(rg_val)].color}) for rg_val
              in df.columns]
    mapper = {rg_val:'{:.0f}'.format(rg_val) for rg_val in df.columns}
    df = df.rename(mapper, axis=1)
    ax = plg.plot_data_stats(cf, df, labels, ax=ax, fs=fs)
    ax.set_xlabel("unaggregrated single-rater malignancy scores (ms)", fontsize=fs)
    ax.set_title("")
    ax.text(0.05, 0.91, 'b)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            weight='bold', fontsize=fs)
    ax.margins(y=0.45)

    #------ binned dissent -----
    #cf.balance_target = "regression_targets"
    all_patients = [(pid,patient['rg_bin_targets']) for pid, patient in dataset.data.items()]
    non_empty_patients = [(pid, lesions) for (pid, lesions) in all_patients if len(lesions) > 0]

    mean_std_per_lesion = np.array([(np.mean(roi), np.std(roi)) for (pid, lesions) in non_empty_patients for roi in lesions])
    distribution_max_per_lesion = [np.unique(roi, return_counts=True) for (pid, lesions) in non_empty_patients for roi in lesions]
    distribution_max_per_lesion = np.array([uniq[cts.argmax()] for (uniq, cts) in distribution_max_per_lesion])

    binned_stats = [[] for bin_id in cf.bin_id2rg_val.keys()]
    for l_ix, mean_std in enumerate(mean_std_per_lesion):
        bin_id = cf.rg_val_to_bin_id(mean_std[0])
        bin_id_max = cf.rg_val_to_bin_id(distribution_max_per_lesion[l_ix])
        binned_stats[int(bin_id)].append((*mean_std, distribution_max_per_lesion[l_ix], bin_id-bin_id_max))

    ax = fig.add_subplot(grid[2, 0])
    plg.plot_binned_rater_dissent(cf, binned_stats, ax=ax, fs=fs)
    ax.set_title("")
    ax.text(0.05, 0.91, 'c)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            weight='bold', fontsize=fs)
    ax.margins(y=0.2)


    plg.plt.savefig(os.path.join(plot_dir, 'data_stats_lidc_solarized.png'), bbox_inches='tight', dpi=600)

    return

def lidc_dataset_stats(exp_dir='', plot_dir=None):

    import datasets.lidc.data_loader as dl
    cf = get_cf('lidc', exp_dir)
    cf.data_rootdir = cf.pp_data_path
    cf.balance_target = "regression_targets"

    dataset = dl.Dataset(cf, data_dir=cf.data_rootdir)
    if plot_dir is None:
        plot_dir = cf.plot_dir if hasattr(cf, 'plot_dir') else os.path.join('datasets', 'lidc', 'misc')

    df, labels = dataset.calc_statistics(plot_dir=plot_dir, overall_stats=True)

    return df, labels

def lidc_sa_dataset_stats(exp_dir='', plot_dir=None):

    import datasets.lidc_sa.data_loader as dl
    cf = get_cf('lidc_sa', exp_dir)
    #cf.data_rootdir = cf.pp_data_path
    cf.balance_target = "regression_targets"

    dataset = dl.Dataset(cf)
    if plot_dir is None:
        plot_dir = cf.plot_dir if hasattr(cf, 'plot_dir') else os.path.join('datasets', 'lidc_sa', 'misc')

    dataset.calc_statistics(plot_dir=plot_dir, overall_stats=True)

    all_patients = [(pid,patient['rg_bin_targets']) for pid, patient in dataset.data.items()]
    empty_patients = [pid for (pid, lesions) in all_patients if len(lesions) == 0]
    non_empty_patients = [(pid, lesions) for (pid, lesions) in all_patients if len(lesions) > 0]
    full_consent_patients = [(pid, lesions) for (pid, lesions) in non_empty_patients if np.all([np.unique(roi).size == 1 for roi in lesions])]
    all_lesions = [roi for (pid, lesions) in non_empty_patients for roi in lesions]
    two_vote_min = [roi for (pid, lesions) in non_empty_patients for roi in lesions if np.count_nonzero(roi) > 1]
    three_vote_min = [roi for (pid, lesions) in non_empty_patients for roi in lesions if np.count_nonzero(roi) > 2]
    mean_std_per_lesion = np.array([(np.mean(roi), np.std(roi)) for (pid, lesions) in non_empty_patients for roi in lesions])
    avg_mean_std_pl = np.mean(mean_std_per_lesion, axis=0)
    # call std dev per lesion disconsent from now on
    disconsent_std = np.std(mean_std_per_lesion[:, 1])

    distribution_max_per_lesion = [np.unique(roi, return_counts=True) for (pid, lesions) in non_empty_patients for roi in lesions]
    distribution_max_per_lesion = np.array([uniq[cts.argmax()] for (uniq, cts) in distribution_max_per_lesion])

    mean_max_delta = abs(mean_std_per_lesion[:, 0] - distribution_max_per_lesion)

    binned_stats = [[] for bin_id in cf.bin_id2rg_val.keys()]
    for l_ix, mean_std in enumerate(mean_std_per_lesion):
        bin_id = cf.rg_val_to_bin_id(mean_std[0])
        bin_id_max = cf.rg_val_to_bin_id(distribution_max_per_lesion[l_ix])
        binned_stats[int(bin_id)].append((*mean_std, distribution_max_per_lesion[l_ix], bin_id-bin_id_max))

    plg.plot_binned_rater_dissent(cf, binned_stats, out_file=os.path.join(plot_dir, "binned_dissent.png"))


    mean_max_bin_divergence = [[] for bin_id in cf.bin_id2rg_val.keys()]
    for bin_id, bin_stats in enumerate(binned_stats):
        mean_max_bin_divergence[bin_id].append([roi for roi in bin_stats if roi[3] != 0])
        mean_max_bin_divergence[bin_id].insert(0,len(mean_max_bin_divergence[bin_id][0]))


    return

def lidc_annotator_confusion(exp_dir='', plot_dir=None, normalize=None, dataset=None, plot=True):
    """
    :param exp_dir:
    :param plot_dir:
    :param normalize: str or None. str in ['truth', 'pred']
    :param dataset:
    :param plot:
    :return:
    """
    if dataset is None:
        import datasets.lidc.data_loader as dl
        cf = get_cf('lidc', exp_dir)
        # cf.data_rootdir = cf.pp_data_path
        cf.training_gts = "sa"
        cf.balance_target = "regression_targets"
        dataset = dl.Dataset(cf)
    else:
        cf = dataset.cf

    if plot_dir is None:
        plot_dir = cf.plot_dir if hasattr(cf, 'plot_dir') else os.path.join('datasets', 'lidc', 'misc')

    dataset.calc_statistics(plot_dir=plot_dir, overall_stats=True)

    all_patients = [(pid,patient['rg_bin_targets']) for pid, patient in dataset.data.items()]
    non_empty_patients = [(pid, lesions) for (pid, lesions) in all_patients if len(lesions) > 0]

    y_true, y_pred = [], []
    for (pid, lesions) in non_empty_patients:
        for roi in lesions:
            true_bin = cf.rg_val_to_bin_id(np.mean(roi))
            y_true.extend([true_bin] * len(roi))
            y_pred.extend(roi)
    cm = confusion_matrix(y_true, y_pred)
    if normalize in ["truth", "row"]:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    elif normalize in ["pred", "prediction", "column", "col"]:
        cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]

    if plot:
        plg.plot_confusion_matrix(cf, cm, out_file=os.path.join(plot_dir, "annotator_confusion.pdf"))

    return cm

def plot_lidc_dissent_and_example(confusion_matrix=True, bin_stds=False, plot_dir=None, numbering=True, example_title="Example"):
    import datasets.lidc.data_loader as dl
    dataset_name = 'lidc'
    exp_dir1 = '/home/gregor/Documents/medicaldetectiontoolkit/datasets/lidc/experiments/ms12345_mrcnn3d_rg_bs8'
    exp_dir2 = '/home/gregor/Documents/medicaldetectiontoolkit/datasets/lidc/experiments/ms12345_mrcnn3d_rgbin_bs8'
    #exp_dir1 = '/home/gregor/networkdrives/E132-Cluster-Projects/lidc_sa/experiments/ms12345_mrcnn3d_rg_bs8'
    #exp_dir2 = '/home/gregor/networkdrives/E132-Cluster-Projects/lidc_sa/experiments/ms12345_mrcnn3d_rgbin_bs8'
    cf = get_cf(dataset_name, exp_dir1)
    #file_names = [f_name for f_name in os.listdir(os.path.join(exp_dir, 'inference_analysis')) if f_name.endswith('.pkl')]
    # file_names = [os.path.join(exp_dir, "inference_analysis", f_name) for f_name in file_names]
    file_names = ["bytes_merged_boxes_fold_0_pid_0811a.pkl",]
    z_ics = [194,]
    plot_files = [
        {'files': [os.path.join(exp_dir, "inference_analysis", f_name) for exp_dir in [exp_dir1, exp_dir2]],
         'z_ix': z_ix} for (f_name, z_ix) in zip(file_names, z_ics)
    ]

    cf.training_gts = 'sa'
    info_df_path = '/mnt/HDD2TB/Documents/data/lidc/pp_20190805/patient_gts_{}/info_df.pickle'.format(cf.training_gts)
    info_df = pd.read_pickle(info_df_path)

    cf.roi_items = ['regression_targets', 'rg_bin_targets_sa'] #['class_targets'] + cf.observables_rois

    text_fs = 14
    title_fs = text_fs
    text_x, text_y = 0.06, 0.92
    fig = plg.plt.figure(figsize=(8.6, 3)) #w, h
    #fig.subplots_adjust(hspace=0, wspace=0)
    grid = plg.plt.GridSpec(1, 4, wspace=0.0, hspace=0.0, figure=fig) #rows, cols
    cf.plot_class_ids = True

    f_ix = 0
    z_ix = plot_files[f_ix]['z_ix']
    for model_ix in range(2)[::-1]:
        print("f_ix, m_ix", f_ix, model_ix)
        plot_file = utils.load_obj(plot_files[f_ix]['files'][model_ix])
        batch = plot_file["batch"]
        pid = batch["pid"][0]
        batch['patient_rg_bin_targets_sa'] = info_df[info_df.pid == pid]['class_target'].tolist()
        # apply same filter as with merged GTs: need at least two non-zero votes to consider a RoI.
        batch['patient_rg_bin_targets_sa'] = [[four_votes.astype("uint8") for four_votes in batch_el if
                                               np.count_nonzero(four_votes>0)>=2] for batch_el in
                                              batch['patient_rg_bin_targets_sa']]
        results_dict = plot_file["res_dict"]

        # pred
        ax = fig.add_subplot(grid[0, model_ix+2])
        plg.view_batch_thesis(cf, batch, res_dict=results_dict, legend=False, sample_picks=None, fontsize=text_fs*1.3,
                              vol_slice_picks=[z_ix, ], show_gt_labels=True, box_score_thres=0.2, plot_mods=False,
                              seg_cmap="rg", show_cl_ids=False,
                              out_file=None, dpi=600, patient_items=True, return_fig=False, axes={'pred': ax})

        #ax.set_title("{}".format("Reg R-CNN" if model_ix==0 else "Mask R-CNN"), size=title_fs)
        ax.set_title("")
        ax.set_xlabel("{}".format("Reg R-CNN" if model_ix == 0 else "Mask R-CNN"), size=title_fs)
        if numbering:
            ax.text(text_x, text_y, chr(model_ix+99)+")", horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, weight='bold', color=cf.white, fontsize=title_fs)
        #ax.axis("off")
        ax.axis("on")
        plg.suppress_axes_lines(ax)

        # GT
        if model_ix==0:
            ax.set_title(example_title, fontsize=title_fs)
            ax = fig.add_subplot(grid[0, 1])
            # ax.imshow(batch['patient_data'][0, 0, :, :, z_ix], cmap='gray')
            # ax.imshow(plg.to_rgba(batch['patient_seg'][0,0,:,:,z_ix], cf.cmap), alpha=0.8)
            plg.view_batch_thesis(cf, batch, res_dict=results_dict, legend=True, sample_picks=None, fontsize=text_fs*1.3,
                                  vol_slice_picks=[z_ix, ], show_gt_labels=True, box_score_thres=0.13, plot_mods=False, seg_cmap="rg",
                                  out_file=None, dpi=600, patient_items=True, return_fig=False, axes={'gt':ax})
            if numbering:
                ax.text(text_x, text_y, "b)", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            weight='bold', color=cf.white, fontsize=title_fs)
            #ax.set_title("Ground Truth", size=title_fs)
            ax.set_title("")
            ax.set_xlabel("Ground Truth", size=title_fs)
            plg.suppress_axes_lines(ax)
            #ax.axis('off')
    #----- annotator dissent plot(s) ------

    cf.training_gts = 'sa'
    cf.balance_targets = 'rg_bin_targets'
    dataset = dl.Dataset(cf, mode='train')

    if bin_stds:
        #------ binned dissent -----
        #cf = get_cf('lidc', "")

        #cf.balance_target = "regression_targets"
        all_patients = [(pid,patient['rg_bin_targets']) for pid, patient in dataset.data.items()]
        non_empty_patients = [(pid, lesions) for (pid, lesions) in all_patients if len(lesions) > 0]

        mean_std_per_lesion = np.array([(np.mean(roi), np.std(roi)) for (pid, lesions) in non_empty_patients for roi in lesions])
        distribution_max_per_lesion = [np.unique(roi, return_counts=True) for (pid, lesions) in non_empty_patients for roi in lesions]
        distribution_max_per_lesion = np.array([uniq[cts.argmax()] for (uniq, cts) in distribution_max_per_lesion])

        binned_stats = [[] for bin_id in cf.bin_id2rg_val.keys()]
        for l_ix, mean_std in enumerate(mean_std_per_lesion):
            bin_id = cf.rg_val_to_bin_id(mean_std[0])
            bin_id_max = cf.rg_val_to_bin_id(distribution_max_per_lesion[l_ix])
            binned_stats[int(bin_id)].append((*mean_std, distribution_max_per_lesion[l_ix], bin_id-bin_id_max))

        ax = fig.add_subplot(grid[0, 0])
        plg.plot_binned_rater_dissent(cf, binned_stats, ax=ax, fs=text_fs)
        if numbering:
            ax.text(text_x, text_y, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                    weight='bold', fontsize=title_fs)
        ax.margins(y=0.2)
        ax.set_xlabel("Malignancy-Score Bins", fontsize=title_fs)
        #ax.yaxis.set_label_position("right")
        #ax.yaxis.tick_right()
        ax.set_yticklabels([])
        #ax.xaxis.set_label_position("top")
        #ax.xaxis.tick_top()
        ax.set_title("Average Rater Dissent", fontsize=title_fs)

    if confusion_matrix:
        #------ confusion matrix -------
        cm = lidc_annotator_confusion(dataset=dataset, plot=False, normalize="truth")
        ax = fig.add_subplot(grid[0, 0])
        cmap = plg.make_colormap([(1,1,1), cf.dkfz_blue])
        plg.plot_confusion_matrix(cf, cm, ax=ax, fs=text_fs, color_bar=False, cmap=cmap )#plg.plt.cm.Purples)
        ax.set_xticks(np.arange(cm.shape[1]))
        if numbering:
            ax.text(-0.16, text_y, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                    weight='bold', fontsize=title_fs)
        ax.margins(y=0.2)
        ax.set_title("Annotator Dissent", fontsize=title_fs)

    #fig.suptitle("               Example", fontsize=title_fs)
    #fig.text(0.63, 1.03, "Example", va="center", ha="center", size=title_fs, transform=fig.transFigure)

    #fig_patches = fig_leg.get_patches()
    #patches= [plg.mpatches.Patch(color=label.color, label="{:.10s}".format(label.name)) for label in cf.bin_id2label.values() if label.id!=0]
    #fig.legends.append(fig_leg)
    #plg.plt.figlegend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, 0.0), borderaxespad=0.,
    # ncol=len(patches), bbox_transform=fig.transFigure, title="Binned Malignancy Score", fontsize= text_fs)
    plg.plt.tight_layout()
    if plot_dir is None:
        plot_dir = "datasets/lidc/misc"
    out_file = os.path.join(plot_dir, "regrcnn_lidc_diss_example.png")
    if out_file is not None:
        plg.plt.savefig(out_file, dpi=600, bbox_inches='tight')

def lidc_annotator_dissent_images(exp_dir='', plot_dir=None):
    if plot_dir is None:
        plot_dir = "datasets/lidc/misc"

    import datasets.lidc.data_loader as dl
    cf = get_cf('lidc', exp_dir)
    cf.training_gts = "sa"

    dataset = dl.Dataset(cf, mode='train')

    pids = {'0069a': 132, '0493a':125, '1008a': 164}#, '0355b': 138, '0484a': 86} # pid : (z_ix to show)
    # add_pids = dataset.set_ids[65:80]
    # for pid in add_pids:
    #     try:
    #
    #         pids[pid] = int(np.median(dataset.data[pid]['fg_slices'][0]))
    #
    #     except (IndexError, ValueError):
    #         print("pid {} has no foreground".format(pid))

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    out_file = os.path.join(plot_dir, "lidc_example_rater_dissent.png")

    #cf.training_gts = 'sa'
    cf.roi_items = ['regression_targets', 'rg_bin_targets_sa'] #['class_targets'] + cf.observables_rois

    title_fs = 14
    text_fs = 14
    fig = plg.plt.figure(figsize=(10, 5.9)) #w, h
    #fig.subplots_adjust(hspace=0, wspace=0)
    grid = plg.plt.GridSpec(len(pids.keys()), 5, wspace=0.0, hspace=0.0, figure=fig) #rows, cols
    cf.plot_class_ids = True
    cmap = {id : (label.color if id!=0 else (0.,0.,0.)) for id, label in cf.bin_id2label.items()}
    legend_handles = set()
    window_size = (250,250)

    for p_ix, (pid, z_ix) in enumerate(pids.items()):
        try:
            print("plotting pid, z_ix", pid, z_ix)
            patient = dataset[pid]
            img = np.load(patient['data'], mmap_mode='r')[z_ix] # z,y,x --> y,x
            seg = np.load(patient['seg'], mmap_mode='r')['seg'][:,z_ix] # rater,z,y,x --> rater,y,x
            rg_bin_targets = patient['rg_bin_targets']

            contours = np.nonzero(seg[0])
            center_y, center_x = np.median(contours[0]), np.median(contours[1])
            #min_y, min_x = np.min(contours[0]), np.min(contours[1])
            #max_y, max_x = np.max(contours[0]), np.max(contours[1])
            #buffer_y, buffer_x = int(seg.shape[1]*0.5), int(seg.shape[2]*0.5)
            #y_range = np.arange(max(min_y-buffer_y, 0), min(min_y+buffer_y, seg.shape[1]))
            #x_range =  np.arange(max(min_x-buffer_x, 0), min(min_x+buffer_x, seg.shape[2]))
            y_range = np.arange(max(int(center_y-window_size[0]/2), 0), min(int(center_y+window_size[0]/2), seg.shape[1]))

            min_x = int(center_x-window_size[1]/2)
            max_x = int(center_x+window_size[1]/2)
            if min_x<0:
                max_x += abs(min_x)
            elif max_x>seg.shape[2]:
                min_x -= max_x-seg.shape[2]
            x_range =  np.arange(max(min_x, 0), min(max_x, seg.shape[2]))
            img = img[y_range][:,x_range]
            seg = seg[:, y_range][:,:,x_range]
            # data
            ax = fig.add_subplot(grid[p_ix, 0])
            ax.imshow(img, cmap='gray')

            plg.suppress_axes_lines(ax)
            # key = "spec" if "spec" in batch.keys() else "pid"
            ylabel = str(pid) + "/" + str(z_ix)
            ax.set_ylabel("{:s}".format(ylabel), fontsize=title_fs)  # show id-number
            if p_ix == 0:
                ax.set_title("Image", fontsize=title_fs)

            # raters
            for r_ix in range(seg.shape[0]):
                rater_bin_targets = rg_bin_targets[:,r_ix]
                for roi_ix, rating in enumerate(rater_bin_targets):
                    seg[r_ix][seg[r_ix]==roi_ix+1] = rating
                ax = fig.add_subplot(grid[p_ix, r_ix+1])
                ax.imshow(seg[r_ix], cmap='gray')
                ax.imshow(plg.to_rgba(seg[r_ix], cmap), alpha=0.8)
                ax.axis('off')
                if p_ix == 0:
                    ax.set_title("Rating {}".format(r_ix+1), fontsize=title_fs)
                legend_handles.update([cf.bin_id2label[id] for id in np.unique(seg[r_ix]) if id!=0])
        except:
            print("failed pid", pid)
            pass

    legend_handles = [plg.mpatches.Patch(color=label.color, label="{:.10s}".format(label.name)) for label in legend_handles]
    legend_handles = sorted(legend_handles, key=lambda h: h._label)
    fig.suptitle("LIDC Single-Rater Annotations", fontsize=title_fs)
    #patches= [plg.mpatches.Patch(color=label.color, label="{:.10s}".format(label.name)) for label in cf.bin_id2label.values() if label.id!=0]

    legend = fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), borderaxespad=0, fontsize=text_fs,
                      bbox_transform=fig.transFigure, ncol=len(legend_handles), title="Malignancy Score")
    plg.plt.setp(legend.get_title(), fontsize=title_fs)
    #grid.tight_layout(fig)
    #plg.plt.tight_layout(rect=[0, 0.00, 1, 1.5])
    if out_file is not None:
        plg.plt.savefig(out_file, dpi=600, bbox_inches='tight')



    return

def lidc_results_static(xlabels=None, plot_dir=None, in_percent=True):
    cf = get_cf('lidc', '')
    if plot_dir is None:
        plot_dir = os.path.join('datasets', 'lidc', 'misc')

    text_fs = 18
    fig = plg.plt.figure(figsize=(3, 2.5)) #w,h
    grid = plg.plt.GridSpec(2, 1, wspace=0.0, hspace=0.0, figure=fig) #r,c

    #--- LIDC 3D -----


    splits = ["Reg R-CNN", "Mask R-CNN"]#, "Reg R-CNN 2D", "Mask R-CNN 2D"]
    values = {"reg3d": [(0.259, 0.035), (0.628, 0.038), (0.477, 0.035)],
              "mask3d": [(0.235, 0.027), (0.622, 0.029), (0.411, 0.026)],}
    groups = [r"$\mathrm{AVP}_{10}$", "$\mathrm{AP}_{10}$", "Bin Acc."]
    if in_percent:
        bar_values = [[v[0]*100 for v in split] for split in values.values()]
        errors = [[v[1]*100 for v in split] for split in values.values()]
    else:
        bar_values = [[v[0] for v in split] for split in values.values()]
        errors = [[v[1] for v in split] for split in values.values()]

    ax = fig.add_subplot(grid[0,0])
    colors = [cf.blue, cf.dkfz_blue]
    plg.plot_grouped_bar_chart(cf, bar_values, groups, splits, errors=errors, colors=colors, ax=ax, legend=False, label_format="{:.1f}",
                               title="LIDC Results", ylabel=r"3D Perf. (%)", xlabel="Metric", yticklabels=[], ylim=(0,80 if in_percent else 0.8))
    #------ LIDC 2D -------

    splits = ["Reg R-CNN", "Mask R-CNN"]
    values = {"reg2d": [(0.148, 0.046), (0.414, 0.052), (0.468, 0.057)],
              "mask2d": [(0.127, 0.034), (0.406, 0.040), (0.447, 0.018)]}
    groups = [r"$\mathrm{AVP}_{10}$", "$\mathrm{AP}_{10}$", "Bin Acc."]
    if in_percent:
        bar_values = [[v[0]*100 for v in split] for split in values.values()]
        errors = [[v[1]*100 for v in split] for split in values.values()]
    else:
        bar_values = [[v[0] for v in split] for split in values.values()]
        errors = [[v[1] for v in split] for split in values.values()]
    ax = fig.add_subplot(grid[1,0])
    colors = [cf.blue, cf.dkfz_blue]
    plg.plot_grouped_bar_chart(cf, bar_values, groups, splits, errors=errors, colors=colors, ax=ax, legend=False, label_format="{:.1f}",
                               title="", ylabel=r"2D Perf.", xlabel="Metric", xticklabels=xlabels, yticklabels=[], ylim=(None,60 if in_percent else 0.6))
    plg.plt.tight_layout()
    plg.plt.savefig(os.path.join(plot_dir, 'lidc_static_results.png'), dpi=700)

def toy_results_static(xlabels=None, plot_dir=None, in_percent=True):
    cf = get_cf('toy', '')
    if plot_dir is None:
        plot_dir = os.path.join('datasets', 'toy', 'misc')

    text_fs = 18
    fig = plg.plt.figure(figsize=(3, 2.5)) #w,h
    grid = plg.plt.GridSpec(2, 1, wspace=0.0, hspace=0.0, figure=fig) #r,c

    #--- Toy 3D -----
    groups = [r"$\mathrm{AVP}_{10}$", "$\mathrm{AP}_{10}$", "Bin Acc."]
    splits = ["Reg R-CNN", "Mask R-CNN"]#, "Reg R-CNN 2D", "Mask R-CNN 2D"]
    values = {"reg3d": [(0.881, 0.014), (0.998, 0.004), (0.887, 0.014)],
              "mask3d": [(0.822, 0.070), (1.0, 0.0), (0.826, 0.069)],}
    if in_percent:
        bar_values = [[v[0]*100 for v in split] for split in values.values()]
        errors = [[v[1]*100 for v in split] for split in values.values()]
    else:
        bar_values = [[v[0] for v in split] for split in values.values()]
        errors = [[v[1] for v in split] for split in values.values()]
    ax = fig.add_subplot(grid[0,0])
    colors = [cf.blue, cf.dkfz_blue]
    plg.plot_grouped_bar_chart(cf, bar_values, groups, splits, errors=errors, colors=colors, ax=ax, legend=True, label_format="{:.1f}",
                               title="Toy Results", ylabel=r"3D Perf. (%)", xlabel="Metric", yticklabels=[], ylim=(0,130 if in_percent else .3))
    #------ Toy 2D -------
    groups = [r"$\mathrm{AVP}_{10}$", "$\mathrm{AP}_{10}$", "Bin Acc."]
    splits = ["Reg R-CNN", "Mask R-CNN"]
    values = {"reg2d": [(0.859, 0.021), (1., 0.0), (0.860, 0.021)],
              "mask2d": [(0.748, 0.022), (1., 0.0), (0.748, 0.021)]}
    if in_percent:
        bar_values = [[v[0]*100 for v in split] for split in values.values()]
        errors = [[v[1]*100 for v in split] for split in values.values()]
    else:
        bar_values = [[v[0] for v in split] for split in values.values()]
        errors = [[v[1] for v in split] for split in values.values()]
    ax = fig.add_subplot(grid[1,0])
    colors = [cf.blue, cf.dkfz_blue]
    plg.plot_grouped_bar_chart(cf, bar_values, groups, splits, errors=errors, colors=colors, ax=ax, legend=False, label_format="{:.1f}",
                               title="", ylabel=r"2D Perf.", xlabel="Metric", xticklabels=xlabels, yticklabels=[], ylim=(None,130 if in_percent else 1.3))
    plg.plt.tight_layout()
    plg.plt.savefig(os.path.join(plot_dir, 'toy_static_results.png'), dpi=700)

def analyze_test_df(dataset_name, exp_dir='', cf=None, logger=None, plot_dir=None):
    evaluator_file = utils.import_module('evaluator', "evaluator.py")
    if cf is None:
        cf = get_cf(dataset_name, exp_dir)
        cf.exp_dir = exp_dir
        cf.test_dir = os.path.join(exp_dir, 'test')
    if logger is None:
        logger = utils.get_logger(cf.exp_dir, False)
    evaluator = evaluator_file.Evaluator(cf, logger, mode='test')

    fold_df_paths = sorted([ii for ii in os.listdir(cf.test_dir) if 'test_df.pkl' in ii])
    fold_seg_df_paths = sorted([ii for ii in os.listdir(cf.test_dir) if 'test_seg_df.pkl' in ii])
    metrics_to_score = ['ap', 'auc']#, 'patient_ap', 'patient_auc', 'patient_dice'] #'rg_bin_accuracy_weighted_tp', 'rg_MAE_w_std_weighted_tp'] #cf.metrics
    if cf.evaluate_fold_means:
        means_to_score = [m for m in metrics_to_score] #+ ['rg_MAE_w_std_weighted_tp']
    #metrics_to_score += ['rg_MAE_std']
    metrics_to_score = []


    cf.fold = 'overall'
    dfs_list = [pd.read_pickle(os.path.join(cf.test_dir, ii)) for ii in fold_df_paths]
    evaluator.test_df = pd.concat(dfs_list, sort=True)

    seg_dfs_list = [pd.read_pickle(os.path.join(cf.test_dir, ii)) for ii in fold_seg_df_paths]
    if len(seg_dfs_list) > 0:
        evaluator.seg_df = pd.concat(seg_dfs_list, sort=True)

    # stats, _ = evaluator.return_metrics(evaluator.test_df, cf.class_dict)
    # results_table_path = os.path.join(cf.exp_dir, "../", "semi_man_summary.csv")
    # # ---column headers---
    # col_headers = ["Experiment Name", "CV Folds", "Spatial Dim", "Clustering Kind", "Clustering IoU", "Merge-2D-to-3D IoU"]
    # if hasattr(cf, "test_against_exact_gt"):
    #     col_headers.append('Exact GT')
    # for s in stats:
    #     assert "overall" in s['name'].split(" ")[0]
    #     if cf.class_dict[cf.patient_class_of_interest] in s['name']:
    #         for metric in metrics_to_score:
    #             #if metric in s.keys() and not np.isnan(s[metric]):
    #             col_headers.append('{}_{} : {}'.format(*s['name'].split(" ")[1:], metric))
    #         for mean in means_to_score:
    #             if mean == "rg_MAE_w_std_weighted_tp":
    #                 col_headers.append('(MAE_fold_mean\u00B1std_fold_mean)\u00B1fold_mean_std\u00B1fold_std_std)'.format(*s['name'].split(" ")[1:], mean))
    #             elif mean in s.keys() and not np.isnan(s[mean]):
    #                 col_headers.append('{}_{} : {}'.format(*s['name'].split(" ")[1:], mean))
    #             else:
    #                 print("skipping {}".format(mean))
    # with open(results_table_path, 'a') as handle:
    #     with open(results_table_path, 'r') as doublehandle:
    #         last_header = doublehandle.readlines()
    #     if len(last_header)==0 or len(col_headers)!=len(last_header[1].split(",")[:-1]) or \
    #             not all([col_headers[ix]==lhix for ix, lhix in enumerate(last_header[1].split(",")[:-1])]):
    #         handle.write('\n')
    #         for head in col_headers:
    #             handle.write(head+',')
    #         handle.write('\n')
    #
    #     # --- columns content---
    #     handle.write('{},'.format(cf.exp_dir.split(os.sep)[-1]))
    #     handle.write('{},'.format(str(evaluator.test_df.fold.unique().tolist()).replace(",", "")))
    #     handle.write('{}D,'.format(cf.dim))
    #     handle.write('{},'.format(cf.clustering))
    #     handle.write('{},'.format(cf.clustering_iou if cf.clustering else str("N/A")))
    #     handle.write('{},'.format(cf.merge_3D_iou if cf.merge_2D_to_3D_preds else str("N/A")))
    #     if hasattr(cf, "test_against_exact_gt"):
    #         handle.write('{},'.format(cf.test_against_exact_gt))
    #     for s in stats:
    #         if cf.class_dict[cf.patient_class_of_interest] in s['name']:
    #             for metric in metrics_to_score:
    #                 #if metric in s.keys() and not np.isnan(s[metric]):  # needed as long as no dice on patient level poss
    #                 handle.write('{:0.3f}, '.format(s[metric]))
    #             for mean in means_to_score:
    #                 #if metric in s.keys() and not np.isnan(s[metric]):
    #                 if mean=="rg_MAE_w_std_weighted_tp":
    #                     handle.write('({:0.3f}\u00B1{:0.3f})\u00B1({:0.3f}\u00B1{:0.3f}),'.format(*s[mean + "_folds_mean"], *s[mean + "_folds_std"]))
    #                 elif mean in s.keys() and not np.isnan(s[mean]):
    #                     handle.write('{:0.3f}\u00B1{:0.3f},'.format(s[mean+"_folds_mean"], s[mean+"_folds_std"]))
    #                 else:
    #                     print("skipping {}".format(mean))
    #
    #     handle.write('\n')

    return evaluator.test_df

def cluster_results_to_df(dataset_name, exp_dir='', overall_df=None, cf=None, logger=None, plot_dir=None):
    evaluator_file = utils.import_module('evaluator', "evaluator.py")
    if cf is None:
        cf = get_cf(dataset_name, exp_dir)
        cf.exp_dir = exp_dir
        cf.test_dir = os.path.join(exp_dir, 'test')
    if logger is None:
        logger = utils.get_logger(cf.exp_dir, False)
    evaluator = evaluator_file.Evaluator(cf, logger, mode='test')
    cf.fold = 'overall'
    metrics_to_score = ['ap', 'auc']#, 'patient_ap', 'patient_auc', 'patient_dice'] #'rg_bin_accuracy_weighted_tp', 'rg_MAE_w_std_weighted_tp'] #cf.metrics
    if cf.evaluate_fold_means:
        means_to_score = [m for m in metrics_to_score] #+ ['rg_MAE_w_std_weighted_tp']
    #metrics_to_score += ['rg_MAE_std']
    metrics_to_score = []

    # use passed overall_df or, if not given, read dfs from file
    if overall_df is None:
        fold_df_paths = sorted([ii for ii in os.listdir(cf.test_dir) if 'test_df.pkl' in ii])
        fold_seg_df_paths = sorted([ii for ii in os.listdir(cf.test_dir) if 'test_seg_df.pkl' in ii])
        for paths in [fold_df_paths, fold_seg_df_paths]:
            assert len(paths) <= cf.n_cv_splits, "found {} > nr of cv splits results dfs in {}".format(len(paths), cf.test_dir)
        dfs_list = [pd.read_pickle(os.path.join(cf.test_dir, ii)) for ii in fold_df_paths]
        evaluator.test_df = pd.concat(dfs_list, sort=True)

        # seg_dfs_list = [pd.read_pickle(os.path.join(cf.test_dir, ii)) for ii in fold_seg_df_paths]
        # if len(seg_dfs_list) > 0:
        #     evaluator.seg_df = pd.concat(seg_dfs_list, sort=True)

    else:
        evaluator.test_df = overall_df
        # todo seg_df if desired

    stats, _ = evaluator.return_metrics(evaluator.test_df, cf.class_dict)
    # ---column headers---
    col_headers = ["Experiment Name", "Model", "CV Folds", "Spatial Dim", "Clustering Kind", "Clustering IoU", "Merge-2D-to-3D IoU"]
    for s in stats:
        assert "overall" in s['name'].split(" ")[0]
        if cf.class_dict[cf.patient_class_of_interest] in s['name']:
            for metric in metrics_to_score:
                #if metric in s.keys() and not np.isnan(s[metric]):
                col_headers.append('{}_{} : {}'.format(*s['name'].split(" ")[1:], metric))
            for mean in means_to_score:
                if mean in s.keys() and not np.isnan(s[mean]):
                    col_headers.append('{}_{} : {}'.format(*s['name'].split(" ")[1:], mean+"_folds_mean"))
                else:
                    print("skipping {}".format(mean))
    results_df = pd.DataFrame(columns=col_headers)
    # --- columns content---
    row = []
    row.append('{}'.format(cf.exp_dir.split(os.sep)[-1]))
    model = 'frcnn' if (cf.model=="mrcnn" and cf.frcnn_mode) else cf.model
    row.append('{}'.format(model))
    row.append('{}'.format(str(evaluator.test_df.fold.unique().tolist()).replace(",", "")))
    row.append('{}D'.format(cf.dim))
    row.append('{}'.format(cf.clustering))
    row.append('{}'.format(cf.clustering_iou if cf.clustering else "N/A"))
    row.append('{}'.format(cf.merge_3D_iou if cf.merge_2D_to_3D_preds else "N/A"))
    for s in stats:
        if cf.class_dict[cf.patient_class_of_interest] in s['name']:
            for metric in metrics_to_score:
                #if metric in s.keys() and not np.isnan(s[metric]):  # needed as long as no dice on patient level poss
                row.append('{:0.3f} '.format(s[metric]))
            for mean in means_to_score:
                #if metric in s.keys() and not np.isnan(s[metric]):
                if mean+"_folds_mean" in s.keys() and not np.isnan(s[mean+"_folds_mean"]):
                    row.append('{:0.3f}\u00B1{:0.3f}'.format(s[mean+"_folds_mean"], s[mean+"_folds_std"]))
                else:
                    print("skipping {}".format(mean+"_folds_mean"))
    #print("row, clustering, iou, exp", row, cf.clustering, cf.clustering_iou, cf.exp_dir)
    results_df.loc[0] = row

    return results_df

def multiple_clustering_results(dataset_name, exp_dir, plot_dir=None, plot_hist=False):
    print("Gathering exp {}".format(exp_dir))
    cf = get_cf(dataset_name, exp_dir)
    cf.n_workers = 1
    logger = logging.getLogger("dummy")
    logger.setLevel(logging.DEBUG)
    #logger.addHandler(logging.StreamHandler())
    cf.exp_dir = exp_dir
    cf.test_dir = os.path.join(exp_dir, 'test')
    cf.plot_prediction_histograms = False
    if plot_dir is None:
        #plot_dir = os.path.join(cf.test_dir, 'histograms')
        plot_dir = os.path.join("datasets", dataset_name, "misc")
        os.makedirs(plot_dir, exist_ok=True)

    # fold_dirs = sorted([os.path.join(cf.exp_dir, f) for f in os.listdir(cf.exp_dir) if
    #                     os.path.isdir(os.path.join(cf.exp_dir, f)) and f.startswith("fold")])
    folds = range(cf.n_cv_splits)
    clusterings = {None: ['lol'], 'wbc': [0.0, 0.1, 0.2, 0.3, 0.4], 'nms': [0.0, 0.1, 0.2, 0.3, 0.4]}
    #clusterings = {'wbc': [0.1,], 'nms': [0.1,]}
    #clusterings = {None: ['lol']}
    if plot_hist:
        clusterings = {None: ['lol'], 'nms': [0.1, ], 'wbc': [0.1, ]}
    class_of_interest = cf.patient_class_of_interest

    try:
        if plot_hist:
            title_fs, text_fs = 16, 13
            fig = plg.plt.figure(figsize=(11, 8)) #width, height
            grid = plg.plt.GridSpec(len(clusterings.keys()), max([len(v) for v in clusterings.values()])+1, wspace=0.0,
                                    hspace=0.0, figure=fig) #rows, cols
            plg.plt.suptitle("Faster R-CNN+", fontsize=title_fs, va='bottom', y=0.925)

        results_df = pd.DataFrame()
        for cl_ix, (clustering, ious) in enumerate(clusterings.items()):
            cf.clustering = clustering
            for iou_ix, iou in enumerate(ious):
                cf.clustering_iou = iou
                print(r"Producing Results for Clustering {} @ IoU {}".format(cf.clustering, cf.clustering_iou))
                overall_test_df = pd.DataFrame()
                for fold in folds[:]:
                    cf.fold = fold
                    cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(cf.fold))

                    predictor = predictor_file.Predictor(cf, net=None, logger=logger, mode='analysis')
                    results_list = predictor.load_saved_predictions()
                    logger.info('starting evaluation...')
                    evaluator = evaluator_file.Evaluator(cf, logger, mode='test')
                    evaluator.evaluate_predictions(results_list)
                    #evaluator.score_test_df(max_fold=100)
                    overall_test_df = overall_test_df.append(evaluator.test_df)

                results_df = results_df.append(cluster_results_to_df(dataset_name, overall_df=overall_test_df,cf=cf,
                                                                     logger=logger))

                if plot_hist:
                    if clustering=='wbc' and iou_ix==len(ious)-1:
                        # plot n_missing histogram for last wbc clustering only
                        out_filename = os.path.join(plot_dir, 'analysis_n_missing_overall_hist_{}_{}.png'.format(clustering, iou))
                        ax = fig.add_subplot(grid[cl_ix, iou_ix+1])
                        plg.plot_wbc_n_missing(cf, overall_test_df, outfile=out_filename, fs=text_fs, ax=ax)
                        ax.set_title("WBC Missing Predictions per Cluster.", fontsize=title_fs)
                        #ax.set_ylabel(r"Average Missing Preds per Cluster (%)")
                        ax.yaxis.tick_right()
                        ax.yaxis.set_label_position("right")
                        ax.text(0.07, 0.87, "{}) WBC".format(chr(len(clusterings.keys())*len(ious)+97)), transform=ax.transAxes, color=cf.white, fontsize=title_fs,
                                bbox=dict(boxstyle='square', facecolor='black', edgecolor='none', alpha=0.9))
                    overall_test_df = overall_test_df[overall_test_df.pred_class == class_of_interest]
                    overall_test_df = overall_test_df[overall_test_df.det_type!='patient_tn']
                    out_filename = "analysis_fold_overall_hist_{}_{}.png".format(clustering, iou)
                    out_filename = os.path.join(plot_dir, out_filename)
                    ax = fig.add_subplot(grid[cl_ix, iou_ix])
                    plg.plot_prediction_hist(cf, overall_test_df, out_filename, fs=text_fs, ax=ax)
                    ax.text(0.11, 0.87, "{}) {}".format(chr((cl_ix+1)*len(ious)+96), clustering.upper() if clustering else "Raw Preds"), transform=ax.transAxes, color=cf.white,
                            bbox=dict(boxstyle='square', facecolor='black', edgecolor='none', alpha=0.9), fontsize=title_fs)
                    if cl_ix==0 and iou_ix==0:
                        ax.set_title("Prediction Histograms Malignant Class", fontsize=title_fs)
                        ax.legend(loc="best", fontsize=text_fs)
                    else:
                        ax.set_title("")
                #analyze_test_df(dataset_name, cf=cf, logger=logger)
        if plot_hist:
            #plg.plt.subplots_adjust(top=0.)
            plg.plt.savefig(os.path.join(plot_dir, "combined_hist_plot.pdf"), dpi=600, bbox_inches='tight')

    except FileNotFoundError as e:
        print("Ignoring exp dir {} due to\n{}".format(exp_dir, e))
    logger.handlers = []
    del cf; del logger
    return results_df

def gather_clustering_results(dataset_name, exp_parent_dir, exps_filter=None, processes=os.cpu_count()//2):
    exp_dirs = [os.path.join(exp_parent_dir, i) for i in os.listdir(exp_parent_dir + "/") if
                os.path.isdir(os.path.join(exp_parent_dir, i))]#[:1]
    if exps_filter is not None:
        exp_dirs = [ed for ed in exp_dirs if not exps_filter in ed]
    # for debugging
    #exp_dir = "/home/gregor/networkdrives/E132-Cluster-Projects/prostate/experiments/gs6071_frcnn3d_cl_bs6"
    #exp_dirs = [exp_dir,]
    #exp_dirs = ["/home/gregor/networkdrives/E132-Cluster-Projects/prostate/experiments/gs6071_detfpn2d_cl_bs10",]

    results_df = pd.DataFrame()

    p = NoDaemonProcessPool(processes=processes)
    mp_inputs = [(dataset_name, exp_dir) for exp_dir in exp_dirs][:]
    results_dfs = p.starmap(multiple_clustering_results, mp_inputs)
    p.close()
    p.join()
    for df in results_dfs:
        results_df = results_df.append(df)

    results_df.to_csv(os.path.join(exp_parent_dir, "df_cluster_summary.csv"), index=False)

    return results_df

def plot_cluster_results_grid(cf, res_df, ylim=None, out_file=None):
    """
    :param cf:
    :param res_df: results over a single dimension setting (2D or 3D), over all clustering methods and ious.
    :param out_file:
    :return:
    """
    is_2d = np.all(res_df["Spatial Dim"]=="2D")
    # pandas has problems with recognising "N/A" string --> replace by None
    #res_df['Merge-2D-to-3D IoU'].iloc[res_df['Merge-2D-to-3D IoU'] == "N/A"] = None
    n_rows = 3#4 if is_2d else 3
    grid = plg.plt.GridSpec(n_rows, 5, wspace=0.4, hspace=0.3)

    fig = plg.plt.figure(figsize=(11,6))

    splits = res_df["Model"].unique().tolist() # need to be model names
    for split in splits:
        assoc_exps = res_df[res_df["Model"]==split]["Experiment Name"].unique()
        if len(assoc_exps)>1:
            print("Model {} has multiple experiments:\n{}".format(split, assoc_exps))
            #res_df = res_df.where(~(res_df["Model"] == split), res_df["Experiment Name"], axis=0)
            raise Exception("Multiple Experiments")

    sort_map = {'detection_fpn': 0, 'mrcnn':1, 'frcnn':2, 'retina_net':3, 'retina_unet':4}
    splits.sort(key=sort_map.__getitem__)
    #colors = [cf.color_palette[ix+3 % len(cf.color_palette)] for ix in range(len(splits))]
    color_map = {'detection_fpn': cf.magenta, 'mrcnn':cf.blue, 'frcnn': cf.dark_blue, 'retina_net': cf.aubergine, 'retina_unet': cf.purple}

    colors = [color_map[split] for split in splits]
    alphas =  [0.9,] * len(splits)
    legend_handles = []
    model_renamer = {'detection_fpn': "Detection U-Net", 'mrcnn': "Mask R-CNN", 'frcnn': "Faster R-CNN+", 'retina_net': "RetinaNet", 'retina_unet': "Retina U-Net"}

    for rix, c_kind in zip([0, 1],['wbc', 'nms']):
        kind_df = res_df[res_df['Clustering Kind'] == c_kind]
        groups = kind_df['Clustering IoU'].unique()
        #for cix, iou in enumerate(groups):
        assert np.all([split in splits for split in kind_df["Model"].unique()]) #need to be model names
        ax = fig.add_subplot(grid[rix,:])
        bar_values = [kind_df[kind_df["Model"]==split]["rois_malignant : ap_folds_mean"] for split in splits]
        bar_stds = [[float(val.split('\u00B1')[1]) for val in split_vals] for split_vals in bar_values]
        bar_values = [ [float(val.split('\u00B1')[0]) for val in split_vals] for split_vals in bar_values ]


        xlabel='' if rix == 0 else "Clustering IoU"
        ylabel = str(c_kind.upper()) + " / AP"
        lh = plg.plot_grouped_bar_chart(cf, bar_values, groups, splits, colors=colors, alphas=alphas, errors=bar_stds,
                                        ax=ax, ylabel=ylabel, xlabel=xlabel)
        legend_handles.append(lh)
        if rix == 0:
            ax.axes.get_xaxis().set_ticks([])
            #ax.spines['top'].set_visible(False)
            #ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            #ax.spines['left'].set_visible(False)
        else:
            ax.spines['top'].set_visible(False)
            #ticklab = ax.xaxis.get_ticklabels()
            #trans = ticklab.get_transform()
            ax.xaxis.set_label_coords(0.05, -0.05)
        ax.set_ylim(0.,ylim)

    if is_2d:
        # only 2d-3d merging @ 0.1
        ax = fig.add_subplot(grid[2, 1])
        kind_df = res_df[(res_df['Clustering Kind'] == 'None') & ~(res_df['Merge-2D-to-3D IoU'].isna())]
        groups = kind_df['Clustering IoU'].unique()
        bar_values = [kind_df[kind_df["Model"] == split]["rois_malignant : ap_folds_mean"] for split in splits]
        bar_stds = [[float(val.split('\u00B1')[1]) for val in split_vals] for split_vals in bar_values]
        bar_values = np.array([[float(val.split('\u00B1')[0]) for val in split_vals] for split_vals in bar_values])
        lh = plg.plot_grouped_bar_chart(cf, bar_values, groups, splits, colors=colors, alphas=alphas, errors=bar_stds,
                                        ax=ax, ylabel="2D-3D Merging\nOnly / AP")
        legend_handles.append(lh)
        ax.axes.get_xaxis().set_ticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_ylim(0., ylim)

        next_row = 2
        next_col = 2
    else:
        next_row = 2
        next_col = 2

    # No clustering at all
    ax = fig.add_subplot(grid[next_row, next_col])
    kind_df = res_df[(res_df['Clustering Kind'] == 'None') & (res_df['Merge-2D-to-3D IoU'].isna())]
    groups = kind_df['Clustering IoU'].unique()
    bar_values = [kind_df[kind_df["Model"] == split]["rois_malignant : ap_folds_mean"] for split in splits]
    bar_stds = [[float(val.split('\u00B1')[1]) for val in split_vals] for split_vals in bar_values]
    bar_values = np.array([[float(val.split('\u00B1')[0]) for val in split_vals] for split_vals in bar_values])
    lh = plg.plot_grouped_bar_chart(cf, bar_values, groups, splits, colors=colors, alphas=alphas, errors=bar_stds,
                                    ax=ax, ylabel="No Clustering / AP")
    legend_handles.append(lh)
    #plg.suppress_axes_lines(ax)
    #ax = fig.add_subplot(grid[next_row, 0])
    #ax.set_ylabel("No Clustering")
    #plg.suppress_axes_lines(ax)
    ax.axes.get_xaxis().set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylim(0., ylim)


    ax = fig.add_subplot(grid[next_row, 3])
    # awful hot fix: only legend_handles[0] used in order to have same order as in plots.
    legend_handles = [plg.mpatches.Patch(color=handle[0], alpha=handle[1], label=model_renamer[handle[2]]) for handle in legend_handles[0]]
    ax.legend(handles=legend_handles)
    ax.axis('off')

    fig.suptitle('Prostate {} Results over Clustering Settings'.format(res_df["Spatial Dim"].unique().item()), fontsize=14)

    if out_file is not None:
        plg.plt.savefig(out_file)

    return

def get_plot_clustering_results(dataset_name, exp_parent_dir, res_from_file=True, exps_filter=None):
    if not res_from_file:
        results_df = gather_clustering_results(dataset_name, exp_parent_dir, exps_filter=exps_filter)
    else:
        results_df = pd.read_csv(os.path.join(exp_parent_dir, "df_cluster_summary.csv"))
        if os.path.isfile(os.path.join(exp_parent_dir, "df_cluster_summary_no_clustering_2D.csv")):
            results_df = results_df.append(pd.read_csv(os.path.join(exp_parent_dir, "df_cluster_summary_no_clustering_2D.csv")))

    cf = get_cf(dataset_name)
    if np.count_nonzero(results_df["Spatial Dim"] == "3D") >0:
        # 3D
        plot_cluster_results_grid(cf, results_df[results_df["Spatial Dim"] == "3D"], ylim=0.52, out_file=os.path.join(exp_parent_dir, "cluster_results_3D.pdf"))
    if np.count_nonzero(results_df["Spatial Dim"] == "2D") > 0:
        # 2D
        plot_cluster_results_grid(cf, results_df[results_df["Spatial Dim"]=="2D"], ylim=0.4, out_file=os.path.join(exp_parent_dir, "cluster_results_2D.pdf"))


def plot_single_results(cf, exp_dir, plot_files, res_df=None):
    out_file = os.path.join(exp_dir, "inference_analysis", "single_results.pdf")

    plot_files = utils.load_obj(plot_files)
    batch = plot_files["batch"]
    results_dict = plot_files["res_dict"]
    cf.roi_items = ['class_targets']

    class_renamer = {1: "GS 6", 2: "GS $\geq 7$"}
    gs_renamer = {60: "6", 71: "7a"}

    if "adcb" in exp_dir:
        modality = "adcb"
    elif "t2" in exp_dir:
        modality = "t2"
    else:
        modality = "b"
    text_fs = 16

    if modality=="t2":
        n_rows, n_cols = 2, 3
        gt_col = 1
        fig_w, fig_h = 14, 4
        input_x, input_y = 0.05, 0.9
        z_ix = 11
        thresh = 0.22
        input_title = "Input"
    elif modality=="b":
        n_rows, n_cols = 2, 6
        gt_col = 2 # = gt_span
        fig_w, fig_h = 14, 4
        input_x, input_y = 0.08, 0.8
        z_ix = 8
        thresh = 0.16
        input_title = "                                 Input"
    elif modality=="adcb":
        n_rows, n_cols = 2, 7
        gt_col = 3
        fig_w, fig_h = 14, 4
        input_x, input_y = 0.08, 0.8
        z_ix = 8
        thresh = 0.16
        input_title = "Input"
    fig_w, fig_h = 12, 3.87
    fig = plg.plt.figure(figsize=(fig_w, fig_h))
    grid = plg.plt.GridSpec(n_rows, n_cols, wspace=0.0, hspace=0.0, figure=fig)
    cf.plot_class_ids = True

    if modality=="t2":
        ax = fig.add_subplot(grid[:, 0])
        ax.imshow(batch['patient_data'][0, 0, :, :, z_ix], cmap='gray')
        ax.set_title("Input", size=text_fs)
        ax.text(0.05, 0.9, "T2", size=text_fs, color=cf.white, transform=ax.transAxes,
                bbox=dict(facecolor=cf.black, alpha=0.7, edgecolor=cf.white, clip_on=False, pad=7))
        ax.axis("off")
    elif modality=="b":
        for m_ix, b in enumerate([50, 500, 1000, 1500]):
            ax = fig.add_subplot(grid[int(np.round(m_ix/4+0.0001)), m_ix%2])
            print(int(np.round(m_ix/4+0.0001)), m_ix%2)
            ax.imshow(batch['patient_data'][0, m_ix, :, :, z_ix], cmap='gray')
            ax.text(input_x, input_y, r"{}{}".format("$b=$" if m_ix==0 else "", b), size=text_fs, color=cf.white, transform=ax.transAxes,
                    bbox=dict(facecolor=cf.black, alpha=0.7, edgecolor=cf.white, clip_on=False, pad=7))
            ax.axis("off")
            if b==50:
                ax.set_title(input_title, size=text_fs)
    elif modality=="adcb":
        for m_ix, b in enumerate(["ADC", 50, 500, 1000, 1500]):
            p_ix = m_ix + 1 if m_ix>2 else m_ix
            ax = fig.add_subplot(grid[int(np.round(p_ix/6+0.0001)), p_ix%3])
            print(int(np.round(p_ix/4+0.0001)), p_ix%2)
            ax.imshow(batch['patient_data'][0, m_ix, :, :, z_ix], cmap='gray')
            ax.text(input_x, input_y, r"{}{}".format("$b=$" if m_ix==1 else "", b), size=text_fs, color=cf.white, transform=ax.transAxes,
                    bbox=dict(facecolor=cf.black, alpha=0.7, edgecolor=cf.white, clip_on=False, pad=7))
            ax.axis("off")
            if b==50:
                ax.set_title(input_title, size=text_fs)

    ax_gt = fig.add_subplot(grid[:, gt_col:gt_col+2]) # GT
    ax_pred = fig.add_subplot(grid[:, gt_col+2:gt_col+4]) # Prediction
    #ax.imshow(batch['patient_data'][0, 0, :, :, z_ix], cmap='gray')
    #ax.imshow(batch['patient_data'][0, 0, :, :, z_ix], cmap='gray')
    #ax.imshow(plg.to_rgba(batch['patient_seg'][0,0,:,:,z_ix], cf.cmap), alpha=0.8)
    plg.view_batch_thesis(cf, batch, res_dict=results_dict, legend=True, sample_picks=None, patient_items=True,
                          vol_slice_picks=[z_ix,], show_gt_labels=True, box_score_thres=thresh, plot_mods=True,
                          out_file=None, dpi=600, return_fig=False, axes={'gt':ax_gt, 'pred':ax_pred}, fontsize=text_fs)


    ax_gt.set_title("Ground Truth", size=text_fs)
    ax_pred.set_title("Prediction", size=text_fs)
    texts = list(ax_gt.texts)
    ax_gt.texts = []
    for text in texts:
        cl_id = int(text.get_text())
        x, y = text.get_position()
        text_str = "GS="+str(gs_renamer[cf.class_id2label[cl_id].gleasons[0]])
        ax_gt.text(x-4*text_fs//2, y,  text_str, color=text.get_color(),
        fontsize=text_fs, bbox=dict(facecolor=text.get_bbox_patch().get_facecolor(), alpha=0.7, edgecolor='none', clip_on=True, pad=0))
    texts = list(ax_pred.texts)
    ax_pred.texts = []
    for text in texts:
        x, y = text.get_position()
        x -= 4 * text_fs // 2
        try:
            cl_id = int(text.get_text())
            text_str = class_renamer[cl_id]
        except ValueError:
            text_str = text.get_text()
        if text.get_bbox_patch().get_facecolor()[:3]==cf.dark_green:
            x -= 4* text_fs
        ax_pred.text(x, y,  text_str, color=text.get_color(),
        fontsize=text_fs, bbox=dict(facecolor=text.get_bbox_patch().get_facecolor(), alpha=0.7, edgecolor='none', clip_on=True, pad=0))

    ax_gt.axis("off")
    ax_pred.axis("off")

    plg.plt.tight_layout()

    if out_file is not None:
        plg.plt.savefig(out_file, dpi=600, bbox_inches='tight')



    return

def find_suitable_examples(exp_dir1, exp_dir2):
    test_df1 = analyze_test_df('lidc',exp_dir1)
    test_df2 = analyze_test_df('lidc', exp_dir2)
    test_df1 = test_df1[test_df1.pred_score>0.3]
    test_df2 = test_df2[test_df2.pred_score > 0.3]

    tp_df1 = test_df1[test_df1.det_type == 'det_tp']

    tp_pids = tp_df1.pid.unique()
    tp_fp_pids = test_df2[(test_df2.pid.isin(tp_pids)) &
                          ((test_df2.regressions-test_df2.rg_targets).abs()>1)].pid.unique()
    cand_df = tp_df1[tp_df1.pid.isin(tp_fp_pids)]
    sorter = (cand_df.regressions - cand_df.rg_targets).abs().argsort()
    cand_df = cand_df.iloc[sorter]
    print("Good guesses for examples: ", cand_df.pid.unique()[:20])
    return

def plot_single_results_lidc():
    dataset_name = 'lidc'
    exp_dir1 = '/home/gregor/Documents/medicaldetectiontoolkit/datasets/lidc/experiments/ms12345_mrcnn3d_rg_copiedparams'
    exp_dir2 = '/home/gregor/Documents/medicaldetectiontoolkit/datasets/lidc/experiments/ms12345_mrcnn3d_rgbin_copiedparams'
    cf = get_cf(dataset_name, exp_dir1)
    #file_names = [f_name for f_name in os.listdir(os.path.join(exp_dir, 'inference_analysis')) if f_name.endswith('.pkl')]
    # file_names = [os.path.join(exp_dir, "inference_analysis", f_name) for f_name in file_names]
    file_names = ['bytes_merged_boxes_fold_0_pid_0296a.pkl', 'bytes_merged_boxes_fold_2_pid_0416a.pkl',
                  'bytes_merged_boxes_fold_1_pid_0635a.pkl', "bytes_merged_boxes_fold_0_pid_0811a.pkl",
                  "bytes_merged_boxes_fold_0_pid_0969a.pkl",
                  # 'bytes_merged_boxes_fold_0_pid_0484a.pkl', 'bytes_merged_boxes_fold_0_pid_0492a.pkl',
                  # 'bytes_merged_boxes_fold_0_pid_0505a.pkl','bytes_merged_boxes_fold_2_pid_0164a.pkl',
                  # 'bytes_merged_boxes_fold_3_pid_0594a.pkl',


                  ]
    z_ics = [167, 159,
             107, 194,
             177,
             # 84, 145,
             # 212, 219,
             # 67
             ]
    plot_files = [
        {'files': [os.path.join(exp_dir, "inference_analysis", f_name) for exp_dir in [exp_dir1, exp_dir2]],
         'z_ix': z_ix} for (f_name, z_ix) in zip(file_names, z_ics)
    ]

    info_df_path = '/mnt/HDD2TB/Documents/data/lidc/pp_20190318/patient_gts_{}/info_df.pickle'.format(cf.training_gts)
    info_df = pd.read_pickle(info_df_path)

    #cf.training_gts = 'sa'
    cf.roi_items = ['regression_targets', 'rg_bin_targets_sa'] #['class_targets'] + cf.observables_rois

    text_fs = 8
    fig = plg.plt.figure(figsize=(6, 9.9)) #w, h
    #fig = plg.plt.figure(figsize=(6, 6.5))
    #fig.subplots_adjust(hspace=0, wspace=0)
    grid = plg.plt.GridSpec(len(plot_files), 3, wspace=0.0, hspace=0.0, figure=fig) #rows, cols
    cf.plot_class_ids = True


    for f_ix, pack in enumerate(plot_files):
        z_ix = plot_files[f_ix]['z_ix']
        for model_ix in range(2)[::-1]:
            print("f_ix, m_ix", f_ix, model_ix)
            plot_file = utils.load_obj(plot_files[f_ix]['files'][model_ix])
            batch = plot_file["batch"]
            pid = batch["pid"][0]
            batch['patient_rg_bin_targets_sa'] = info_df[info_df.pid == pid]['class_target'].tolist()
            # apply same filter as with merged GTs: need at least two non-zero votes to consider a RoI.
            batch['patient_rg_bin_targets_sa'] = [[four_votes for four_votes in batch_el if
                                                   np.count_nonzero(four_votes>0)>=2] for batch_el in
                                                  batch['patient_rg_bin_targets_sa']]
            results_dict = plot_file["res_dict"]

            # pred
            ax = fig.add_subplot(grid[f_ix, model_ix+1])
            plg.view_batch_thesis(cf, batch, res_dict=results_dict, legend=True, sample_picks=None,
                                              vol_slice_picks=[z_ix, ], show_gt_labels=True, box_score_thres=0.2,
                                              plot_mods=False,
                                              out_file=None, dpi=600, patient_items=True, return_fig=False,
                                              axes={'pred': ax})
            if f_ix==0:
                ax.set_title("{}".format("Reg R-CNN" if model_ix==0 else "Mask R-CNN"), size=text_fs*1.3)
            else:
                ax.set_title("")

            ax.axis("off")
            #grid.tight_layout(fig)

            # GT
            if model_ix==0:
                ax = fig.add_subplot(grid[f_ix, 0])
                # ax.imshow(batch['patient_data'][0, 0, :, :, z_ix], cmap='gray')
                # ax.imshow(plg.to_rgba(batch['patient_seg'][0,0,:,:,z_ix], cf.cmap), alpha=0.8)
                boxes_fig = plg.view_batch_thesis(cf, batch, res_dict=results_dict, legend=True, sample_picks=None,
                                                  vol_slice_picks=[z_ix, ], show_gt_labels=True, box_score_thres=0.1,
                                                  plot_mods=False, seg_cmap="rg",
                                                  out_file=None, dpi=600, patient_items=True, return_fig=False,
                                                  axes={'gt':ax})
                ax.set_ylabel(r"$\mathbf{"+chr(f_ix+97)+")}$ " + ax.get_ylabel())
                ax.set_ylabel("")
                if f_ix==0:
                    ax.set_title("Ground Truth", size=text_fs*1.3)
                else:
                    ax.set_title("")


    #fig_patches = fig_leg.get_patches()
    patches= [plg.mpatches.Patch(color=label.color, label="{:.10s}".format(label.name)) for label in cf.bin_id2label.values() if not label.id in [0,]]
    #fig.legends.append(fig_leg)
    plg.plt.figlegend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, 0.0), borderaxespad=0.,
                      ncol=len(patches), bbox_transform=fig.transFigure, title="Binned Malignancy Score",
                      fontsize= text_fs)
    plg.plt.tight_layout()
    out_file = os.path.join(exp_dir1, "inference_analysis", "lidc_example_results_solarized.pdf")
    if out_file is not None:
        plg.plt.savefig(out_file, dpi=600, bbox_inches='tight')


def box_clustering(exp_dir='', plot_dir=None):
    import datasets.prostate.data_loader as dl
    cf = get_cf('prostate', exp_dir)
    if plot_dir is None:
        plot_dir = cf.plot_dir if hasattr(cf, 'plot_dir') else os.path.join('datasets', 'prostate', 'misc')

    fig = plg.plt.figure(figsize=(10, 4))
    #fig.subplots_adjust(hspace=0, wspace=0)
    grid = plg.plt.GridSpec(2, 3, wspace=0.0, hspace=0., figure=fig)
    fs = 14
    xyA = (.9, 0.5)
    xyB = (0.05, .5)

    patch_size = np.array([200, 320])
    clustering_iou = 0.1
    img_y, img_x = patch_size

    boxes = [
        {'box_coords': [img_y * 0.2, img_x * 0.04, img_y * 0.55, img_x * 0.31], 'box_score': 0.45, 'box_cl': 1,
         'regression': 2., 'rg_bin': cf.rg_val_to_bin_id(1.),
         'box_patch_center_factor': 1., 'ens_ix': 1, 'box_n_overlaps': 1.},
        {'box_coords': [img_y*0.05, img_x*0.05, img_y*0.5, img_x*0.3], 'box_score': 0.85, 'box_cl': 2,
         'regression': 1., 'rg_bin': cf.rg_val_to_bin_id(1.),
         'box_patch_center_factor': 1., 'ens_ix':1, 'box_n_overlaps':1.},
        {'box_coords': [img_y * 0.1, img_x * 0.2, img_y * 0.4, img_x * 0.7], 'box_score': 0.95, 'box_cl': 2,
         'regression': 1., 'rg_bin': cf.rg_val_to_bin_id(1.),
         'box_patch_center_factor': 1., 'ens_ix':1, 'box_n_overlaps':1.},
        {'box_coords': [img_y * 0.80, img_x * 0.35, img_y * 0.95, img_x * 0.85], 'box_score': 0.6, 'box_cl': 2,
         'regression': 1., 'rg_bin': cf.rg_val_to_bin_id(1.),
         'box_patch_center_factor': 1., 'ens_ix': 1, 'box_n_overlaps': 1.},
        {'box_coords': [img_y * 0.85, img_x * 0.4, img_y * 0.93, img_x * 0.9], 'box_score': 0.85, 'box_cl': 2,
         'regression': 1., 'rg_bin': cf.rg_val_to_bin_id(1.),
         'box_patch_center_factor': 1., 'ens_ix':1, 'box_n_overlaps':1.},
    ]
    for box in boxes:
        c = box['box_coords']
        box_centers = np.array([(c[ii + 2] - c[ii]) / 2 for ii in range(len(c) // 2)])
        box['box_patch_center_factor'] = np.mean(
            [norm.pdf(bc, loc=pc, scale=pc * 0.8) * np.sqrt(2 * np.pi) * pc * 0.8 for bc, pc in
             zip(box_centers, patch_size / 2)])
        print("pc fact", box['box_patch_center_factor'])

    box_coords = np.array([box['box_coords'] for box in boxes])
    box_scores = np.array([box['box_score'] for box in boxes])
    box_cl_ids = np.array([box['box_cl'] for box in boxes])
    ax0 = fig.add_subplot(grid[:,:2])
    plg.plot_boxes(cf, box_coords, patch_size, box_scores, box_cl_ids, out_file=os.path.join(plot_dir, "demo_boxes_unclustered.png"), ax=ax0)
    ax0.text(*xyA, 'a) Raw ', horizontalalignment='right', verticalalignment='center', transform=ax0.transAxes,
            weight='bold', fontsize=fs)

    nms_boxes = []
    for cl in range(1,3):
        cl_boxes = [box for box in boxes if box['box_cl'] == cl ]
        box_coords = np.array([box['box_coords'] for box in cl_boxes])
        box_scores = np.array([box['box_score'] for box in cl_boxes])
        if 0 not in box_scores.shape:
            keep_ix = mutils.nms_numpy(box_coords, box_scores, thresh=clustering_iou)
        else:
            keep_ix = []
        nms_boxes += [cl_boxes[ix] for ix in keep_ix]
        box_coords = np.array([box['box_coords'] for box in nms_boxes])
        box_scores = np.array([box['box_score'] for box in nms_boxes])
        box_cl_ids = np.array([box['box_cl'] for box in nms_boxes])
    ax1 = fig.add_subplot(grid[1, 2])
    nms_color = cf.black
    plg.plot_boxes(cf, box_coords, patch_size, box_scores, box_cl_ids, out_file=os.path.join(plot_dir, "demo_boxes_nms_iou_{}.png".format(clustering_iou)), ax=ax1)
    ax1.text(*xyB, ' c) NMS', horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes,
            weight='bold', color=nms_color, fontsize=fs)

    #------ WBC -------------------
    regress_flag = False

    wbc_boxes = []
    for cl in range(1,3):
        cl_boxes = [box for box in boxes if box['box_cl'] == cl]
        box_coords = np.array([box['box_coords'] for box in cl_boxes])
        box_scores = np.array([box['box_score'] for box in cl_boxes])
        box_center_factor = np.array([b['box_patch_center_factor'] for b in cl_boxes])
        box_n_overlaps = np.array([b['box_n_overlaps'] for b in cl_boxes])
        box_ens_ix = np.array([b['ens_ix'] for b in cl_boxes])
        box_regressions = np.array([b['regression'] for b in cl_boxes]) if regress_flag else None
        box_rg_bins = np.array([b['rg_bin'] if 'rg_bin' in b.keys() else float('NaN') for b in cl_boxes])
        box_rg_uncs = np.array([b['rg_uncertainty'] if 'rg_uncertainty' in b.keys() else float('NaN') for b in cl_boxes])
        if 0 not in box_scores.shape:
            keep_scores, keep_coords, keep_n_missing, keep_regressions, keep_rg_bins, keep_rg_uncs = \
                predictor_file.weighted_box_clustering(box_coords, box_scores, box_center_factor, box_n_overlaps, box_rg_bins, box_rg_uncs,
                                        box_regressions, box_ens_ix, clustering_iou, n_ens=1)

            for boxix in range(len(keep_scores)):
                clustered_box = {'box_type': 'det', 'box_coords': keep_coords[boxix],
                                 'box_score': keep_scores[boxix], 'cluster_n_missing': keep_n_missing[boxix],
                                 'box_pred_class_id': cl}
                if regress_flag:
                    clustered_box.update({'regression': keep_regressions[boxix],
                                          'rg_uncertainty': keep_rg_uncs[boxix],
                                          'rg_bin': keep_rg_bins[boxix]})
                wbc_boxes.append(clustered_box)

    box_coords = np.array([box['box_coords'] for box in wbc_boxes])
    box_scores = np.array([box['box_score'] for box in wbc_boxes])
    box_cl_ids = np.array([box['box_pred_class_id'] for box in wbc_boxes])
    ax2 = fig.add_subplot(grid[0, 2])
    wbc_color = cf.black
    plg.plot_boxes(cf, box_coords, patch_size, box_scores, box_cl_ids, out_file=os.path.join(plot_dir, "demo_boxes_wbc_iou_{}.png".format(clustering_iou)), ax=ax2)
    ax2.text(*xyB, ' b) WBC', horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes,
            weight='bold', color=wbc_color, fontsize=fs)
    # ax2.spines['bottom'].set_color(wbc_color)
    # ax2.spines['top'].set_color(wbc_color)
    # ax2.spines['right'].set_color(wbc_color)
    # ax2.spines['left'].set_color(wbc_color)

    from matplotlib.patches import ConnectionPatch
    con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="axes fraction", coordsB="axes fraction",
                          axesA=ax0, axesB=ax2, color=wbc_color, lw=1.5, arrowstyle='-|>')
    ax0.add_artist(con)

    con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="axes fraction", coordsB="axes fraction",
                          axesA=ax0, axesB=ax1, color=nms_color, lw=1.5, arrowstyle='-|>')
    ax0.add_artist(con)
    # ax0.text(0.5, 0.5, "Test", size=30, va="center", ha="center", rotation=30,
    #          bbox=dict(boxstyle="angled,pad=0.5", alpha=0.2))
    plg.plt.tight_layout()
    plg.plt.savefig(os.path.join(plot_dir, "box_clustering.pdf"), bbox_inches='tight')

def sketch_AP_AUC(plot_dir=None, draw_auc=True):
    from sklearn.metrics import roc_curve, roc_auc_score
    from understanding_metrics import get_det_types
    import matplotlib.transforms as mtrans
    cf = get_cf('prostate', '')
    if plot_dir is None:
        plot_dir = cf.plot_dir if hasattr(cf, 'plot_dir') else os.path.join('.')

    if draw_auc:
        fig = plg.plt.figure(figsize=(7, 6)) #width, height
        # fig.subplots_adjust(hspace=0, wspace=0)
        grid = plg.plt.GridSpec(2, 2, wspace=0.23, hspace=.45, figure=fig) #rows, cols
    else:
        fig = plg.plt.figure(figsize=(12, 3)) #width, height
        # fig.subplots_adjust(hspace=0, wspace=0)
        grid = plg.plt.GridSpec(1, 3, wspace=0.23, hspace=.45, figure=fig) #rows, cols
    fs = 13
    text_fs = 11
    optim_color = cf.dark_green
    non_opt_color = cf.aubergine

    df = pd.DataFrame(columns=['pred_score', 'class_label', 'pred_class', 'det_type', 'match_iou'])
    df2 = df.copy()
    df["pred_score"] = [0,0.3,0.25,0.2, 0.8, 0.9, 0.9, 0.9, 0.9]
    df["class_label"] = [0,0,0,0, 1, 1, 1, 1, 1]
    df["det_type"] = get_det_types(df)
    df["match_iou"] = [0.1] * len(df)

    df2["pred_score"] = [0, 0.77, 0.5, 1., 0.5, 0.35, 0.3, 0., 0.7, 0.85, 0.9]
    df2["class_label"] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    df2["det_type"] = get_det_types(df2)
    df2["match_iou"] = [0.1] * len(df2)

    #------ PRC -------
    # optimal
    if draw_auc:
        ax = fig.add_subplot(grid[1, 0])
    else:
        ax = fig.add_subplot(grid[0, 2])
    pr, rc = evaluator_file.compute_prc(df)
    ax.plot(rc, pr, color=optim_color, label="Optimal Detection")
    ax.fill_between(rc, pr, alpha=0.33, color=optim_color)

    # suboptimal
    pr, rc = evaluator_file.compute_prc(df2)
    ax.plot(rc, pr, color=non_opt_color, label="Suboptimal")
    ax.fill_between(rc, pr, alpha=0.33, color=non_opt_color)
    #plt.title()
    #plt.legend(loc=3 if c == 'prc' else 4)
    ax.set_ylabel('precision', fontsize=text_fs)
    ax.set_ylim((0., 1.1))
    ax.set_xlabel('recall', fontsize=text_fs)
    ax.set_title('Precision-Recall Curves', fontsize=fs)
    #ax.legend(ncol=2, loc='center')#, bbox_to_anchor=(0.5, 1.05))


    #---- ROC curve
    if draw_auc:
        ax = fig.add_subplot(grid[1, 1])
        roc = roc_curve(df.class_label.tolist(), df.pred_score.tolist())
        ax.plot(roc[0], roc[1], color=optim_color)
        ax.fill_between(roc[0], roc[1], alpha=0.33, color=optim_color)
        ax.set_xlabel('false-positive rate', fontsize=text_fs)
        ax.set_ylim((0., 1.1))
        ax.set_ylabel('recall', fontsize=text_fs)

        roc = roc_curve(df2.class_label.tolist(), df2.pred_score.tolist())
        ax.plot(roc[0], roc[1], color=non_opt_color)
        ax.fill_between(roc[0], roc[1], alpha=0.33, color=non_opt_color)

        roc = ([0, 1], [0, 1])
        ax.plot(roc[0], roc[1], color=cf.gray, linestyle='dashed', label="random predictor")

        ax.set_title('ROC Curves', fontsize=fs)
        ax.legend(ncol=2, loc='lower right', fontsize=text_fs)

    #--- hist optimal
    text_left = 0.05
    ax = fig.add_subplot(grid[0, 0])
    tn_count = df.det_type.tolist().count('det_tn')
    AUC = roc_auc_score(df.class_label, df.pred_score)
    df = df[(df.det_type=="det_tp") | (df.det_type=="det_fp") | (df.det_type=="det_fn")]
    labels = df.class_label.values
    preds = df.pred_score.values
    type_list = df.det_type.tolist()

    ax.hist(preds[labels == 0], alpha=0.3, color=cf.red, range=(0, 1), bins=50, label="FP")
    ax.hist(preds[labels == 1], alpha=0.3, color=cf.blue, range=(0, 1), bins=50, label="FN at score 0 and TP")
    #ax.axvline(x=cf.min_det_thresh, alpha=0.4, color=cf.orange, linewidth=1.5, label="min det thresh")
    fp_count = type_list.count('det_fp')
    fn_count = type_list.count('det_fn')
    tp_count = type_list.count('det_tp')
    pos_count = fn_count + tp_count
    if draw_auc:
        text = "AP: {:.2f} ROC-AUC: {:.2f}\n".format(evaluator_file.get_roi_ap_from_df((df, 0.0, False)), AUC)
    else:
        text = "AP: {:.2f}\n".format(evaluator_file.get_roi_ap_from_df((df, 0.0, False)))
    text += 'TP: {} FP: {} FN: {} TN: {}\npositives: {}'.format(tp_count, fp_count, fn_count, tn_count, pos_count)

    ax.text(text_left,4, text, fontsize=text_fs)
    ax.set_yscale('log')
    ax.set_ylim(bottom=10**-2, top=10**2)
    ax.set_xlabel("prediction score", fontsize=text_fs)
    ax.set_ylabel("occurences", fontsize=text_fs)
    #autoAxis = ax.axis()
    # rec = plg.mpatches.Rectangle((autoAxis[0] - 0.7, autoAxis[2] - 0.2), (autoAxis[1] - autoAxis[0]) + 1,
    #                 (autoAxis[3] - autoAxis[2]) + 0.4, fill=False, lw=2)
    # rec = plg.mpatches.Rectangle((autoAxis[0] , autoAxis[2] ), (autoAxis[1] - autoAxis[0]) ,
    #                 (autoAxis[3] - autoAxis[2]) , fill=False, lw=2, color=optim_color)
    # rec = ax.add_patch(rec)
    # rec.set_clip_on(False)
    plg.plt.setp(ax.spines.values(), color=optim_color, linewidth=2)
    ax.set_facecolor((*optim_color,0.1))
    ax.set_title("Detection Histograms", fontsize=fs)

    ax = fig.add_subplot(grid[0, 1])
    tn_count = df2.det_type.tolist().count('det_tn')
    AUC = roc_auc_score(df2.class_label, df2.pred_score)
    df2 = df2[(df2.det_type=="det_tp") | (df2.det_type=="det_fp") | (df2.det_type=="det_fn")]
    labels = df2.class_label.values
    preds = df2.pred_score.values
    type_list = df2.det_type.tolist()

    ax.hist(preds[labels == 0], alpha=0.3, color=cf.red, range=(0, 1), bins=50, label="FP")
    ax.hist(preds[labels == 1], alpha=0.3, color=cf.blue, range=(0, 1), bins=50, label="FN at score 0 and TP")
    # ax.axvline(x=cf.min_det_thresh, alpha=0.4, color=cf.orange, linewidth=1.5, label="min det thresh")
    fp_count = type_list.count('det_fp')
    fn_count = type_list.count('det_fn')
    tp_count = type_list.count('det_tp')
    pos_count = fn_count + tp_count
    if draw_auc:
        text = "AP: {:.2f} ROC-AUC: {:.2f}\n".format(evaluator_file.get_roi_ap_from_df((df2, 0.0, False)), AUC)
    else:
        text = "AP: {:.2f}\n".format(evaluator_file.get_roi_ap_from_df((df2, 0.0, False)))
    text += 'TP: {} FP: {} FN: {} TN: {}\npositives: {}'.format(tp_count, fp_count, fn_count, tn_count, pos_count)

    ax.text(text_left, 4*10**0, text, fontsize=text_fs)
    ax.set_yscale('log')
    ax.margins(y=10e2)
    ax.set_ylim(bottom=10**-2, top=10**2)
    ax.set_xlabel("prediction score", fontsize=text_fs)
    ax.set_yticks([])
    plg.plt.setp(ax.spines.values(), color=non_opt_color, linewidth=2)
    ax.set_facecolor((*non_opt_color, 0.05))
    ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.18), fontsize=text_fs)

    if draw_auc:
        # Draw a horizontal line
        line = plg.plt.Line2D([0.1, .9], [0.48, 0.48], transform=fig.transFigure, color="black")
        fig.add_artist(line)

    outfile = os.path.join(plot_dir, "metrics.png")
    print("Saving plot to {}".format(outfile))
    plg.plt.savefig(outfile, bbox_inches='tight', dpi=600)

    return

def draw_toy_cylinders(plot_dir=None):
    source_path = "datasets/toy"
    if plot_dir is None:
        plot_dir = os.path.join(source_path, "misc")
        #plot_dir = '/home/gregor/Dropbox/Thesis/Main/tmp'
    os.makedirs(plot_dir, exist_ok=True)

    cf = get_cf('toy', '')
    cf.pre_crop_size = [2200, 2200,1] #y,x,z;
    #cf.dim = 2
    cf.ambiguities = {"radius_calib": (1., 1. / 6) }
    cf.pp_blur_min_intensity = 0.2

    generate_toys = utils.import_module("generate_toys", os.path.join(source_path, 'generate_toys.py'))
    ToyGen = generate_toys.ToyGenerator(cf)

    fig = plg.plt.figure(figsize=(10, 8.2)) #width, height
    grid = plg.plt.GridSpec(4, 5, wspace=0.0, hspace=.0, figure=fig) #rows, cols
    fs, text_fs = 16, 14
    text_x, text_y = 0.5, 0.85
    true_gt_col, dist_gt_col = cf.dark_green, cf.blue
    true_cmap = {1:true_gt_col}

    img = np.random.normal(loc=0.0, scale=cf.noise_scale, size=ToyGen.sample_size)
    img[img < 0.] = 0.
    # one-hot-encoded seg
    seg = np.zeros((cf.num_classes + 1, *ToyGen.sample_size)).astype('uint8')
    undistorted_seg = np.copy(seg)
    applied_gt_distort = False

    class_id, shape = 1, 'cylinder'
    #all_radii = ToyGen.generate_sample_radii(class_ids, shapes)
    enlarge_f = 20
    all_radii = np.array([np.mean(label.bin_vals) if label.id!=5 else label.bin_vals[0]+5 for label in cf.bin_labels if label.id!=0])
    bins = [(min(label.bin_vals), max(label.bin_vals)) for label in cf.bin_labels]
    bin_edges = [(bins[i][1] + bins[i + 1][0])*enlarge_f / 2 for i in range(len(bins) - 1)]
    all_radii = [np.array([r*enlarge_f, r*enlarge_f, 1]) for r in all_radii] # extend to required 3D format
    regress_targets, undistorted_rg_targets = [], []
    ics = np.argwhere(np.ones(seg[0].shape)) # indices ics equal positions within img/volume
    center = np.array([dim//2 for dim in img.shape])

    # for illustrating GT distribution, keep scale same size
    #x = np.linspace(mu - 300, mu + 300, 100)
    x = np.linspace(0, 50*enlarge_f, 500)
    ax_gauss = fig.add_subplot(grid[3, :])
    mus, sigmas = [], []

    for roi_ix, radii in enumerate(all_radii):
        print('processing {} {}'.format(roi_ix, radii))
        cur_img, cur_seg, cur_undistorted_seg, cur_regress_targets, cur_undistorted_rg_targets, cur_applied_gt_distort = \
            ToyGen.draw_object(img.copy(), seg.copy(), undistorted_seg, ics, regress_targets, undistorted_rg_targets, applied_gt_distort,
                             roi_ix, class_id, shape, np.copy(radii), center)

        ax = fig.add_subplot(grid[0,roi_ix])
        ax.imshow(cur_img[...,0], cmap='gray', vmin=0)
        ax.set_title("r{}".format(roi_ix+1), fontsize=fs)
        if roi_ix==0:
            ax.set_ylabel(r"$\mathbf{a)}$ Input", fontsize=fs)
            plg.suppress_axes_lines(ax)
        else:
            ax.axis('off')

        ax = fig.add_subplot(grid[1, roi_ix])
        ax.imshow(cur_img[..., 0], cmap='gray')
        ax.imshow(plg.to_rgba(np.argmax(cur_undistorted_seg[...,0], axis=0), true_cmap), alpha=0.8)
        ax.text(text_x, text_y, r"$r_{a}=$"+"{:.1f}".format(cur_undistorted_rg_targets[roi_ix][0]/enlarge_f), transform=ax.transAxes,
                color=cf.white, bbox=dict(facecolor=true_gt_col, alpha=0.7, edgecolor=cf.white, clip_on=False,pad=2.5),
                fontsize=text_fs, ha='center', va='center')
        if roi_ix==0:
            ax.set_ylabel(r"$\mathbf{b)}$ Exact GT", fontsize=fs)
            plg.suppress_axes_lines(ax)
        else:
            ax.axis('off')
        ax = fig.add_subplot(grid[2, roi_ix])
        ax.imshow(cur_img[..., 0], cmap='gray')
        ax.imshow(plg.to_rgba(np.argmax(cur_seg[..., 0], axis=0), cf.cmap), alpha=0.7)
        ax.text(text_x, text_y, r"$r_{a}=$"+"{:.1f}".format(cur_regress_targets[roi_ix][0]/enlarge_f), transform=ax.transAxes,
                color=cf.white, bbox=dict(facecolor=cf.blue, alpha=0.7, edgecolor=cf.white, clip_on=False,pad=2.5),
                fontsize=text_fs, ha='center', va='center')
        if roi_ix == 0:
            ax.set_ylabel(r"$\mathbf{c)}$ Noisy GT", fontsize=fs)
            plg.suppress_axes_lines(ax)
        else:
            ax.axis('off')

        # GT distributions
        assert radii[0]==radii[1]
        mu, sigma = radii[0], radii[0] * cf.ambiguities["radius_calib"][1]
        ax_gauss.axvline(mu, color=true_gt_col)
        ax_gauss.text(mu, -0.003, "$r=${:.0f}".format(mu/enlarge_f), color=true_gt_col, fontsize=text_fs, ha='center', va='center',
                      bbox = dict(facecolor='none', alpha=0.7, edgecolor=true_gt_col, clip_on=False, pad=2.5))
        mus.append(mu); sigmas.append(sigma)
        lower_bound = max(bin_edges[roi_ix], min(x))# if roi_ix>0 else 2*mu-bin_edges[roi_ix+1]
        upper_bound = bin_edges[roi_ix+1] if len(bin_edges)>roi_ix+1 else max(x)#2*mu-bin_edges[roi_ix]
        if roi_ix<len(all_radii)-1:
            ax_gauss.axvline(upper_bound, color='white', linewidth=7)
        ax_gauss.axvspan(lower_bound, upper_bound, ymax=0.9999, facecolor=true_gt_col, alpha=0.4, edgecolor='none')
        if roi_ix == 0:
            ax_gauss.set_ylabel(r"$\mathbf{d)}$ GT Distr.", fontsize=fs)
            #plg.suppress_axes_lines(ax_gauss)
            #min_x, max_x = min(x/enlarge_f), max(x/enlarge_f)
            #ax_gauss.xaxis.set_ticklabels(["{:.0f}".format(x_tick) for x_tick in np.arange(min_x, max_x, (max_x-min_x)/5)])
            ax_gauss.xaxis.set_ticklabels([])
            ax_gauss.axes.yaxis.set_ticks([])
            ax_gauss.spines['top'].set_visible(False)
            ax_gauss.spines['right'].set_visible(False)
            #ax.spines['bottom'].set_visible(False)
            ax_gauss.spines['left'].set_visible(False)
    for d_ix, (mu, sigma) in enumerate(zip(mus, sigmas)):
        ax_gauss.plot(x, norm.pdf(x, mu, sigma), color=dist_gt_col, alpha=0.6+d_ix/10)
    ax_gauss.margins(x=0)
    # in-axis coordinate cross
    arrow_x, arrow_y, arrow_dx, arrow_dy = 30, ax_gauss.get_ylim()[1]/3, 30, ax_gauss.get_ylim()[1]/3
    ax_gauss.arrow(arrow_x, arrow_y, 0., arrow_dy, length_includes_head=False, head_width=10, head_length=0.001, head_starts_at_zero=False, shape="full", width=0.5, fc="black", ec="black")
    ax_gauss.arrow(arrow_x, arrow_y, arrow_dx, 0, length_includes_head=False, head_width=0.001, head_length=8,
                   head_starts_at_zero=False, shape="full", width=0.00005, fc="black", ec="black")
    ax_gauss.text(arrow_x-20, arrow_y + arrow_dy*0.5, r"$prob$", fontsize=text_fs, ha='center', va='center', rotation=90)
    ax_gauss.text(arrow_x + arrow_dx * 0.5, arrow_y *0.85, r"$r$", fontsize=text_fs, ha='center', va='center', rotation=0)
    # ax_gauss.annotate(r"$p$", xytext=(0, 0), xy=(0, arrow_y), fontsize=fs,
    #             arrowprops=dict(arrowstyle="-|>, head_length = 0.05, head_width = .005", lw=1))
    #ax_gauss.arrow(1, 0.5, 0., 0.1)
    handles = [plg.mpatches.Patch(facecolor=dist_gt_col, label='Inexact Seg.', alpha=0.7, edgecolor='none'),
               mlines.Line2D([], [], color=dist_gt_col, marker=r'$\curlywedge$', linestyle='none', markersize=11, label='GT Sampling Distr.'),
               mlines.Line2D([], [], color=true_gt_col, marker='|', markersize=12, label='Exact GT Radius.', linestyle='none'),
               plg.mpatches.Patch(facecolor=true_gt_col, label='a)-c) Exact Seg., d) Bin', alpha=0.7, edgecolor='none')]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=text_fs)
    outfile = os.path.join(plot_dir, "toy_cylinders.png")
    print("Saving plot to {}".format(outfile))
    plg.plt.savefig(outfile, bbox_inches='tight', dpi=600)


    return

def seg_det_cityscapes_example(plot_dir=None):
    cf = get_cf('cityscapes', '')
    source_path = "datasets/cityscapes"
    if plot_dir is None:
        plot_dir = os.path.join(source_path, "misc")
    os.makedirs(plot_dir, exist_ok=True)


    dl = utils.import_module("dl", os.path.join(source_path, 'data_loader.py'))
    #from utils.dataloader_utils import ConvertSegToBoundingBoxCoordinates
    data_set = dl.Dataset(cf)
    Converter = dl.ConvertSegToBoundingBoxCoordinates(2, cf.roi_items)

    fig = plg.plt.figure(figsize=(9, 3)) #width, height
    grid = plg.plt.GridSpec(1, 2, wspace=0.05, hspace=.0, figure=fig) #rows, cols
    fs, text_fs = 12, 10

    nice_imgs = ["bremen000099000019", "hamburg000000033506", "frankfurt000001058914",]
    img_id = nice_imgs[2]
    #img_id = np.random.choice(data_set.set_ids)


    print("Selected img", img_id)
    img = np.load(data_set[img_id]["img"]).transpose(1,2,0)
    seg = np.load(data_set[img_id]["seg"])
    cl_targs = data_set[img_id]["class_targets"]
    roi_ids = np.unique(seg[seg > 0])
    # ---- detection example -----
    cl_id2name = {1: "h", 2: "v"}
    color_palette = [cf.purple, cf.aubergine, cf.magenta, cf.dark_blue, cf.blue, cf.bright_blue, cf.cyan, cf.dark_green,
                     cf.green, cf.dark_yellow, cf.yellow, cf.orange,  cf.red, cf.dark_red, cf.bright_red]
    n_colors = len(color_palette)
    cmap = {roi_id : color_palette[(roi_id-1)%n_colors] for roi_id in roi_ids}
    cmap[0] = (1,1,1,0.)

    ax = fig.add_subplot(grid[0, 1])
    ax.imshow(img)
    ax.imshow(plg.to_rgba(seg, cmap), alpha=0.7)

    data_dict = Converter(**{'seg':seg[np.newaxis, np.newaxis], 'class_targets': [cl_targs]}) # needs batch dim and channel
    for roi_ix, bb_target in enumerate(data_dict['bb_target'][0]):
        [y1, x1, y2, x2] = bb_target
        width, height = x2 - x1, y2 - y1
        cl_id = cl_targs[roi_ix]
        label = cf.class_id2label[cl_id]
        text_x, text_y = x2, y1
        id_text = cl_id2name[cl_id]
        text_str = '{}'.format(id_text)
        text_settings = dict(facecolor=label.color, alpha=0.5, edgecolor='none', clip_on=True, pad=0)
        #ax.text(text_x, text_y, text_str, color=cf.white, bbox=text_settings, fontsize=text_fs, ha="center", va="center")
        edgecolor = label.color
        bbox = plg.mpatches.Rectangle((x1, y1), width, height, linewidth=1.05, edgecolor=edgecolor, facecolor='none')
        ax.add_patch(bbox)
    ax.axis('off')

    # ---- seg example -----
    for roi_id in roi_ids:
        seg[seg==roi_id] = cl_targs[roi_id-1]

    ax = fig.add_subplot(grid[0,0])
    ax.imshow(img)
    ax.imshow(plg.to_rgba(seg, cf.cmap), alpha=0.7)
    ax.axis('off')

    plg.plt.tight_layout()
    outfile = os.path.join(plot_dir, "cityscapes_example.png")
    print("Saving plot to {}".format(outfile))
    plg.plt.savefig(outfile, bbox_inches='tight', dpi=600)





if __name__=="__main__":
    stime = time.time()
    #seg_det_cityscapes_example()
    #box_clustering()
    #sketch_AP_AUC(draw_auc=False)
    #draw_toy_cylinders()
    #prostate_GT_examples(plot_dir="/home/gregor/Dropbox/Thesis/Main/MFPPresentation/graphics")
    #prostate_results_static()
    #prostate_dataset_stats(plot_dir="/home/gregor/Dropbox/Thesis/Main/MFPPresentation/graphics", show_splits=False)
    #lidc_dataset_stats()
    #lidc_sa_dataset_stats()
    #lidc_annotator_confusion()
    #lidc_merged_sa_joint_plot()
    #lidc_annotator_dissent_images()
    exp_dir = "/home/gregor/networkdrives/E132-Cluster-Projects/prostate/experiments/gs6071_frcnn3d_cl_bs6"
    #multiple_clustering_results('prostate', exp_dir, plot_hist=True)
    exp_parent_dir = "/home/gregor/networkdrives/E132-Cluster-Projects/prostate/experiments"
    exp_parent_dir = "/home/gregor/networkdrives/E132-Cluster-Projects/prostate/experiments_debug_retinas"
    #get_plot_clustering_results('prostate', exp_parent_dir, res_from_file=False)

    exp_dir = "/home/gregor/networkdrives/E132-Cluster-Projects/prostate/experiments/gs6071_frcnn3d_cl_bs6"
    #cf = get_cf('prostate', exp_dir)
    #plot_file = os.path.join(exp_dir, "inference_analysis/bytes_merged_boxes_fold_1_pid_177.pkl")
    #plot_single_results(cf, exp_dir, plot_file)

    exp_dir1 = "/home/gregor/networkdrives/E132-Cluster-Projects/lidc_sa/experiments/ms12345_mrcnn3d_rg_bs8"
    exp_dir2 = "/home/gregor/networkdrives/E132-Cluster-Projects/lidc_sa/experiments/ms12345_mrcnn3d_rgbin_bs8"
    #find_suitable_examples(exp_dir1, exp_dir2)
    #plot_single_results_lidc()
    plot_dir = "/home/gregor/Dropbox/Thesis/MICCAI2019/Graphics"
    #lidc_results_static(plot_dir=plot_dir)
    #toy_results_static(plot_dir=plot_dir)
    plot_lidc_dissent_and_example(plot_dir=plot_dir, confusion_matrix=True, numbering=False, example_title="LIDC example result")

    mins, secs = divmod((time.time() - stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))