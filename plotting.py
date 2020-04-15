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

import matplotlib
# matplotlib.rcParams['font.family'] = ['serif']
# matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.use('Agg') #complains with spyder editor, bc spyder imports mpl at startup
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import StrMethodFormatter, ScalarFormatter
import SimpleITK as sitk
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer

import sys
import os
import warnings
import time

from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.interpolate as interpol

from utils.exp_utils import IO_safe

warnings.filterwarnings("ignore", module="matplotlib.image")


def make_colormap(seq):
    """ Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
bw_cmap = make_colormap([(1.,1.,1.), (0.,0.,0.)])

#------------------------------------------------------------------------
#------------- plotting functions, not all are used ---------------------


def shape_small_first(shape):
    """sort a tuple so that the smallest entry is swapped to the beginning
    """
    if len(shape) <= 2:  # no changing dimensions if channel-dim is missing
        return shape
    smallest_dim = np.argmin(shape)
    if smallest_dim != 0:  # assume that smallest dim is color channel
        new_shape = np.array(shape)  # to support mask indexing
        new_shape = (new_shape[smallest_dim],
                     *new_shape[(np.arange(len(shape), dtype=int) != smallest_dim)])
        return new_shape
    else:
        return shape

def RGB_to_rgb(RGB):
    rgb = np.array(RGB) / 255.
    return rgb

def mod_to_rgb(arr, cmap=None):
    """convert a single-channel modality img to 3-color-channel img.
    :param arr: input img, expected in shape (b,c,)x,y with c=1
    :return: img of shape (...,c') with c'=3
    """
    if len(arr.shape) == 3:
        arr = np.squeeze(arr)
    elif len(arr.shape) != 2:
        raise Exception("Invalid input arr shape: {}".format(arr.shape))

    if cmap is None:
        cmap = "gray"
    norm = matplotlib.colors.Normalize()
    norm.autoscale(arr)
    arr = norm(arr)
    arr = np.stack((arr,) * 3, axis=-1)

    return arr

def to_rgb(arr, cmap):
    """
    Transform an integer-labeled segmentation map using an rgb color-map.
    :param arr: img_arr w/o a color-channel
    :param cmap: dictionary mapping from integer class labels to rgb values
    :return: img of shape (...,c)
    """
    new_arr = np.zeros(shape=(arr.shape) + (3,))
    for l in cmap.keys():
        ixs = np.where(arr == l)
        new_arr[ixs] = np.array([cmap[l][i] for i in range(3)])

    return new_arr

def to_rgba(arr, cmap):
    """
    Transform an integer-labeled segmentation map using an rgba color-map.
    :param arr: img_arr w/o a color-channel
    :param cmap: dictionary mapping from integer class labels to rgba values
    :return: new array holding rgba-image
    """
    new_arr = np.zeros(shape=(arr.shape) + (4,))
    for lab, val in cmap.items():
        # in case no alpha, complement with 100% alpha
        if len(val) == 3:
            cmap[lab] = (*val, 1.)
        assert len(cmap[lab]) == 4, "cmap has color with {} entries".format(len(val))

    for lab in cmap.keys():
        ixs = np.where(arr == lab)
        rgb = np.array(cmap[lab][:3])
        new_arr[ixs] = np.append(rgb, cmap[lab][3])

    return new_arr

def bin_seg_to_rgba(arr, color):
    """
    Transform a continuously labelled binary segmentation map using an rgba color-map.
    values are expected to be 0-1, will give alpha-value
    :param arr: img_arr w/o a color-channel
    :param color: color to give img
    :return: new array holding rgba-image
    """
    new_arr = np.zeros(shape=(arr.shape) + (4,))

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            new_arr[i][j] = (*color, arr[i][j])

    return new_arr

def suppress_axes_lines(ax):
    """
    :param ax: pyplot axes object
    """
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return

def label_bar(ax, rects, labels=None, colors=None, fontsize=10):
    """Attach a text label above each bar displaying its height

    :param ax:
    :param rects: rectangles as returned by plt.bar()
    :param labels:
    :param colors:
    """
    for ix, rect in enumerate(rects):
        height = rect.get_height()
        if labels is not None and labels[ix] is not None:
            label = labels[ix]
        else:
            label = '{:g}'.format(height)
        if colors is not None and colors[ix] is not None and np.any(np.array(colors[ix])<1):
            color = colors[ix]
        else:
            color = 'black'
        ax.text(rect.get_x() + rect.get_width() / 2., 1.007 * height, label, color=color, ha='center', va='bottom',
                bbox=dict(facecolor=(1., 1., 1.), edgecolor='none', clip_on=True, pad=0, alpha=0.75), fontsize=fontsize)

def draw_box_into_arr(arr, box_coords, box_color=None, lw=2):
    """
    :param arr: imgs shape, (3,y,x)
    :param box_coords: (x1,y1,x2,y2), in ascending order
    :param box_color: arr of shape (3,)
    :param lw: linewidth in pixels
    """
    if box_color is None:
        box_color = [1., 0.4, 0.]

    (x1, y1, x2, y2) = box_coords[:4]

    arr = np.swapaxes(arr, 0, -1)
    arr[..., y1:y2, x1:x1 + lw, :], arr[..., y1:y2 + lw, x2:x2 + lw, :] = box_color, box_color
    arr[..., y1:y1 + lw, x1:x2, :], arr[..., y2:y2 + lw, x1:x2, :] = box_color, box_color
    arr = np.swapaxes(arr, 0, -1)

    return arr

def draw_boxes_into_batch(imgs, batch_boxes, type2color=None, cmap=None):
    """
    :param imgs: either the actual batch imgs or a tuple with shape of batch imgs,
    need to have 3 color channels, need to be rgb;
    """
    if isinstance(imgs, tuple):
        img_oshp = imgs
        imgs = None
    else:
        img_oshp = imgs[0].shape

    img_shp = shape_small_first(img_oshp)  # c,x/y,y/x now
    imgs = np.reshape(imgs, (-1, *img_shp))
    box_imgs = np.empty((len(batch_boxes), *(img_shp)))

    for sample, boxes in enumerate(batch_boxes):
        # imgs in batch have shape b,c,x,y, swap c to end
        sample_img = np.full(img_shp, 1.) if imgs is None else imgs[sample]
        for box in boxes:
            if len(box["box_coords"]) > 0:
                if type2color is not None and "box_type" in box.keys():
                    sample_img = draw_box_into_arr(sample_img, box["box_coords"].astype(np.int32),
                                                   type2color[box["box_type"]])
                else:
                    sample_img = draw_box_into_arr(sample_img, box["box_coords"].astype(np.int32))
        box_imgs[sample] = sample_img

    return box_imgs


def plot_prediction_hist(cf, spec_df, outfile, title=None, fs=11, ax=None):

    labels = spec_df.class_label.values
    preds = spec_df.pred_score.values
    type_list = spec_df.det_type.tolist() if hasattr(spec_df, "det_type") else None
    if title is None:
        title = outfile.split('/')[-1] + ' count:{}'.format(len(labels))
    close=False
    if ax is None:
        fig = plt.figure(tight_layout=True)
        ax = fig.add_subplot(1,1,1)
        close=True
    ax.set_yscale('log')

    ax.set_xlabel("Prediction Score", fontsize=fs)
    ax.set_ylabel("Occurences", fontsize=fs)
    
    ax.hist(preds[labels == 0], alpha=0.3, color=cf.red, range=(0, 1), bins=50, label="fp")
    ax.hist(preds[labels == 1], alpha=0.3, color=cf.blue, range=(0, 1), bins=50, label="fn at score 0 and tp")
    ax.axvline(x=cf.min_det_thresh, alpha=1, color=cf.orange, linewidth=1.5, label="min det thresh")

    if type_list is not None:
        fp_count = type_list.count('det_fp')
        fn_count = type_list.count('det_fn')
        tp_count = type_list.count('det_tp')
        pos_count = fn_count + tp_count
        title += '\ntp:{} fp:{} fn:{} pos:{}'.format(tp_count, fp_count, fn_count, pos_count)

    ax.set_title(title, fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.tick_params(axis='both', which='minor', labelsize=fs)

    if close:
        ax.legend(loc="best", fontsize=fs)
        if cf.server_env:
            IO_safe(plt.savefig, fname=outfile, _raise=False)
        else:
            plt.savefig(outfile)
            pass
        plt.close()

def plot_wbc_n_missing(cf, df, outfile, fs=11, ax=None):
    """ WBC (weighted box clustering) has parameter n_missing, which shows how many boxes are missing per cluster.
        This function plots the average relative amount of missing boxes sorted by cluster score.
    :param cf: config.
    :param df: dataframe.
    :param outfile: path to save image under.
    :param fs: fontsize.
    :param ax: axes object.
    """

    bins = np.linspace(0., 1., 10)
    names = ["{:.1f}".format((bins[i]+(bins[i+1]-bins[i])/2.)*100) for i in range(len(bins)-1)]
    classes = df.pred_class.unique()
    colors = [cf.class_id2label[cl_id].color for cl_id in classes]

    binned_df = df.copy()
    binned_df.loc[:,"pred_score"] = pd.cut(binned_df["pred_score"], bins)

    close=False
    if ax is None:
        ax = plt.subplot()
        close=True
    width = 1 / (len(classes) + 1)
    group_positions = np.arange(len(names))
    legend_handles = []

    for ix, cl_id in enumerate(classes):
        cl_df = binned_df[binned_df.pred_class==cl_id].groupby("pred_score").agg({"cluster_n_missing": 'mean'})
        ax.bar(group_positions + ix * width, cl_df.cluster_n_missing.values, width=width, color=colors[ix],
                       alpha=0.4 + ix / 2 / len(classes), edgecolor=colors[ix])
        legend_handles.append(mpatches.Patch(color=colors[ix], label=cf.class_dict[cl_id]))

    title = "Fold {} WBC Missing Preds\nAverage over scores and classes: {:.1f}%".format(cf.fold, df.cluster_n_missing.mean())
    ax.set_title(title, fontsize=fs)
    ax.legend(handles=legend_handles, title="Class", loc="best", fontsize=fs, title_fontsize=fs)
    ax.set_xticks(group_positions + (len(classes) - 1) * width / 2)
    # ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.1f}')) THIS WONT WORK... no clue!
    ax.set_xticklabels(names)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.tick_params(axis='both', which='minor', labelsize=fs)

    ax.set_axisbelow(True)
    ax.grid()
    ax.set_ylabel(r"Average Missing Preds per Cluster (%)", fontsize=fs)
    ax.set_xlabel("Prediction Score", fontsize=fs)

    if close:
        if cf.server_env:
            IO_safe(plt.savefig, fname=outfile, _raise=False)
        else:
            plt.savefig(outfile)
        plt.close()

def plot_stat_curves(cf, stats, outfile, fill=False):
    """ Plot precision-recall and/or receiver-operating-characteristic curve(s).
    :param cf: config.
    :param stats: statistics as supplied by Evaluator.
    :param outfile: path to save plot under.
    :param fill: whether to colorize space between plot and x-axis.
    :return:
    """

    for c in ['roc', 'prc']:
        plt.figure()
        empty_plot = True
        for ix, s in enumerate(stats):
            if s[c] is not np.nan:
                plt.plot(s[c][1], s[c][0], label=s['name'] + '_' + c, marker=None,
                         color=cf.color_palette[ix%len(cf.color_palette)])
                empty_plot = False
                if fill:
                    plt.fill_between(s[c][1], s[c][0], alpha=0.33,  color=cf.color_palette[ix%len(cf.color_palette)])
        if not empty_plot:
            plt.title(outfile.split('/')[-1] + '_' + c)
            plt.legend(loc=3 if c == 'prc' else 4)
            plt.ylabel('precision' if c == 'prc' else '1-spec.')
            plt.ylim((0.,1))
            plt.xlabel('recall')

            plt.savefig(outfile + '_' + c)
            plt.close()


def plot_grouped_bar_chart(cf, bar_values, groups, splits, colors=None, alphas=None, errors=None, ylabel='', xlabel='',
                           xticklabels=None, yticks=None, yticklabels=None, ylim=None, label_format="{:.3f}",
                           title=None, ax=None, out_file=None, legend=False, fs=11):
    """ Plot a categorically grouped bar chart.
    :param cf: config.
    :param bar_values: values of the bars.
    :param groups: groups/categories that bars belong to.
    :param splits: splits within groups, i.e., names of bars.
    :param colors: colors.
    :param alphas: 1-opacity.
    :param errors: values for errorbars.
    :param ylabel: label of y-axis.
    :param xlabel: label of x-axis.
    :param title: plot title.
    :param ax: axes object to draw into. if None, new is created.
    :param out_file: path to save plot.
    :param legend: whether to show a legend.
    :param fs: fontsize.
    :return: legend handles.
    """
    bar_values = np.array(bar_values)
    if alphas is None:
        alphas = [1.,] * len(splits)
    if colors is None:
        colors = [cf.color_palette[ix%len(cf.color_palette)] for ix in range(len(splits))]
    if errors is None:
        errors = np.zeros_like(bar_values)
    # patterns = ('/', '\\', '*', 'O', '.', '-', '+', 'x',  'o')
    # patterns = tuple([patterns[ix%len(patterns)] for ix in range(len(splits))])
    close=False
    if ax is None:
        ax = plt.subplot()
        close=True
    width = 1 / (len(splits) +0.25)
    group_positions = np.arange(len(groups))

    for ix, split in enumerate(splits):
        rects = ax.bar(group_positions + ix * width, bar_values[ix], width=width, color=(*colors[ix], 0.8),
                       edgecolor=colors[ix], yerr=errors[ix], ecolor=(*np.array(colors[ix])*0.8, 1.), capsize=5)
        # for ix, bar in enumerate(rects):
        # bar.set_hatch(patterns[ix])
        labels = [label_format.format(val) for val in bar_values[ix]]
        label_bar(ax, rects, labels, [colors[ix]]*len(labels), fontsize=fs)

    legend_handles = [mpatches.Patch(color=colors[ix], alpha=alphas[ix], label=split) for ix, split in
                     enumerate(splits)]
    if legend:
        ax.legend(handles=legend_handles, fancybox=True, framealpha=1., loc="lower center")
    legend_handles = [(colors[ix], alphas[ix], split) for ix, split in enumerate(splits)]

    if title is not None:
        ax.set_title(title, fontsize=fs)

    ax.set_xticks(group_positions + (len(splits) - 1) * width / 2)
    if xticklabels is None:
        ax.set_xticklabels(groups, fontsize=fs)
    else:
        ax.set_xticklabels(xticklabels, fontsize=fs)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.tick_params(labelsize=fs)

    ax.grid(axis='y')
    ax.set_ylabel(ylabel, fontsize=fs)
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, fontsize=fs)
    if ylim is not None:
        ax.set_ylim(ylim)

    if out_file is not None:
        plt.savefig(out_file, dpi=600)
    if close:
        plt.close()

    return legend_handles

def plot_binned_rater_dissent(cf, binned_stats, out_file=None, ax=None, legend=True, fs=11):
    """ LIDC-specific plot: rater disagreement as standard deviations within each bin.
    :param cf: config.
    :param binned_stats: list, ix==bin_id, item: [(roi_mean, roi_std, roi_max, roi_bin_id-roi_max_bin_id) for roi in bin]
    :return:
    """

    dissent = [np.array([roi[1] for roi in bin]) for bin in binned_stats]
    avg_dissent_first_degree = [np.mean(bin) for bin in dissent]

    groups = list(cf.bin_id2label.keys())
    splits = [r"$1^{st}$ std. dev.",]
    colors = [cf.bin_id2label[bin_id].color[:3] for bin_id in groups]
    #colors = [cf.blue for bin_id in groups]
    alphas = [0.9,]
    #patterns = ('/', '\\', '*', 'O', '.', '-', '+', 'x',  'o')
    #patterns = tuple([patterns[ix%len(patterns)] for ix in range(len(splits))])

    close=False
    if ax is None:
        ax = plt.subplot()
        close=True
    width = 1/(len(splits)+1)
    group_positions = np.arange(len(groups))

    #total_counts = [df.loc[split].sum() for split in splits]
    dissent = np.array(avg_dissent_first_degree)
    ix=0
    rects = ax.bar(group_positions+ix*width, dissent, color=colors, alpha=alphas[ix],
                   edgecolor=colors)
    #for ix, bar in enumerate(rects):
        #bar.set_hatch(patterns[ix])
    labels = ["{:.2f}".format(diss) for diss in dissent]
    label_bar(ax, rects, labels, colors, fontsize=fs)
    bin_edge_color = cf.blue
    ax.axhline(y=0.5, color=bin_edge_color)
    ax.text(2.5, 0.38, "bin edge", color=cf.white, fontsize=fs, horizontalalignment="center",
            bbox=dict(boxstyle='round', facecolor=(*bin_edge_color, 0.85), edgecolor='none', clip_on=True, pad=0))

    if legend:
        legend_handles = [mpatches.Patch(color=cf.blue ,alpha=alphas[ix], label=split) for ix, split in enumerate(splits)]
        ax.legend(handles=legend_handles, loc='lower center', fontsize=fs)

    title = "LIDC-IDRI: Average Std Deviation per Lesion"
    plt.title(title)

    ax.set_xticks(group_positions + (len(splits)-1)*width/2)
    ax.set_xticklabels(groups, fontsize=fs)
    ax.set_axisbelow(True)
    #ax.tick_params(axis='both', which='major', labelsize=fs)
    #ax.tick_params(axis='both', which='minor', labelsize=fs)
    ax.grid()
    ax.set_ylabel(r"Average Dissent (MS)", fontsize=fs)
    ax.set_xlabel("binned malignancy-score value (ms)", fontsize=fs)
    ax.tick_params(labelsize=fs)
    if out_file is not None:
        plt.savefig(out_file, dpi=600)

    if close:
        plt.close()

    return

def plot_confusion_matrix(cf, cm, out_file=None, ax=None, fs=11, cmap=plt.cm.Blues, color_bar=True):
    """ Plot a confusion matrix.
    :param cf: config.
    :param cm: confusion matrix, e.g., as supplied by metrics.confusion_matrix from scikit-learn.
    :return:
    """

    close=False
    if ax is None:
        ax = plt.subplot()
        close=True

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if color_bar:
        ax.figure.colorbar(im, ax=ax)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.0%' if np.mod(cm, 1).any() else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel(r"Binned Mean MS", fontsize=fs)
    ax.set_xlabel("Single-Annotator MS", fontsize=fs)
    #ax.tick_params(labelsize=fs)
    if close and out_file is not None:
        plt.savefig(out_file, dpi=600)

    if close:
        plt.close()
    else:
        return ax

def plot_data_stats(cf, df, labels=None, out_file=None, ax=None, fs=11):
    """ Plot data-set statistics. Shows target counts. Mainly used by Dataset Class in dataloader.py.
    :param cf: configs obj
    :param df: pandas dataframe
    :param out_file: path to save fig in
    """
    names = df.columns
    if labels is not None:
        colors = [label.color for name in names for label in labels if label.name==name]
    else:
        colors = [cf.color_palette[ix%len(cf.color_palette)] for ix in range(len(names))]
    #patterns = ('/', '\\', '*', 'O', '.', '-', '+', 'x',  'o')
    #patterns = tuple([patterns[ix%len(patterns)] for ix in range(len(splits))])
    if ax is None:
        fig, ax = plt.subplots(figsize=(14,6), dpi=300)
        return_ax = False
    else:
        return_ax = True

    plt.margins(x=0.01)
    plt.subplots_adjust(bottom=0.15)
    bar_positions = np.arange(len(names))
    name_counts = df.sum()
    total_count = name_counts.sum()

    rects = ax.bar(bar_positions, name_counts, color=colors, alpha=0.9, edgecolor=colors)
    labels = ["{:.0f}%".format(count/ total_count*100) for count in name_counts]
    label_bar(ax, rects, labels, colors, fontsize=fs)

    title= "Data Set RoI-Target Balance\nTotal #RoIs: {}".format(int(total_count))
    ax.set_title(title, fontsize=fs)
    ax.set_xticks(bar_positions)
    rotation = "vertical" if np.any([len(str(name)) > 3 for name in names]) else None
    if all([isinstance(name, (float, int)) for name in names]):
        ax.set_xticklabels(["{:.2f}".format(name) for name in names], rotation=rotation, fontsize=fs)
    else:
        ax.set_xticklabels(names, rotation=rotation, fontsize=fs)

    ax.set_axisbelow(True)
    ax.grid()
    ax.set_ylabel(r"#RoIs", fontsize=fs)
    ax.set_xlabel(str(df._metadata[0]), fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.tick_params(axis='both', which='minor', labelsize=fs)

    if out_file is not None:
        plt.savefig(out_file)

    if return_ax:
        return ax
    else:
        plt.close()

def plot_fold_stats(cf, df, labels=None, out_file=None, ax=None):
    """ Similar as plot_data_stats but per single cross-val fold.
    :param cf: configs obj
    :param df: pandas dataframe
    :param out_file: path to save fig in
    """
    names = df.columns
    splits = df.index
    if labels is not None:
        colors = [label.color for name in names for label in labels if label.name==name]
    else:
        colors = [cf.color_palette[ix%len(cf.color_palette)] for ix in range(len(names))]
    #patterns = ('/', '\\', '*', 'O', '.', '-', '+', 'x',  'o')
    #patterns = tuple([patterns[ix%len(patterns)] for ix in range(len(splits))])
    if ax is None:
        ax = plt.subplot()
        return_ax = False
    else:
        return_ax = True
    width = 1/(len(names)+1)
    group_positions = np.arange(len(splits))
    legend_handles = []

    total_counts = [df.loc[split].sum() for split in splits]

    for ix, name in enumerate(names):
        rects = ax.bar(group_positions+ix*width, df.loc[:,name], width=width, color=colors[ix], alpha=0.9,
                       edgecolor=colors[ix])
        #for ix, bar in enumerate(rects):
            #bar.set_hatch(patterns[ix])
        labels = ["{:.0f}%".format(df.loc[split, name]/ total_counts[ii]*100) for ii, split in enumerate(splits)]
        label_bar(ax, rects, labels, [colors[ix]]*len(group_positions))

        legend_handles.append(mpatches.Patch(color=colors[ix] ,alpha=0.9, label=name))

    title= "Fold {} RoI-Target Balances\nTotal #RoIs: {}".format(cf.fold,
                 int(df.values.sum()))
    plt.title(title)
    ax.legend(handles=legend_handles)
    ax.set_xticks(group_positions + (len(names)-1)*width/2)
    ax.set_xticklabels(splits, rotation="vertical" if len(splits)>2 else None, size=12)
    ax.set_axisbelow(True)
    ax.grid()
    ax.set_ylabel(r"#RoIs")
    ax.set_xlabel("Set split")

    if out_file is not None:
        plt.savefig(out_file)
    if return_ax:
        return ax
    plt.close()

def plot_batchgen_distribution(cf, pids, p_probs, balance_target, out_file=None):
    """plot top n_pids probabilities for drawing a pid into a batch.
    :param cf: experiment config object
    :param pids: sorted iterable of patient ids
    :param p_probs: pid's drawing likelihood, order needs to match the one of pids.
    :param out_file:
    :return:
    """
    n_pids = len(pids)
    zip_sorted = np.array(sorted(list(zip(p_probs, pids)), reverse=True))
    names, probs = zip_sorted[:n_pids,1], zip_sorted[:n_pids,0].astype('float32') * 100
    try:
        names = [str(int(n)) for n in names]
    except ValueError:
        names = [str(n) for n in names]
    lowest_p = min(p_probs)*100
    fig, ax = plt.subplots(1,1,figsize=(17,5), dpi=200)
    rects = ax.bar(names, probs, color=cf.blue, alpha=0.9, edgecolor=cf.blue)
    ax = plt.gca()
    ax.text(0.8, 0.92, "Lowest prob.: {:.5f}%".format(lowest_p), transform=ax.transAxes, color=cf.white,
            bbox=dict(boxstyle='round', facecolor=cf.blue, edgecolor='none', alpha=0.9))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    ax.set_xticklabels(names, rotation="vertical", fontsize=7)
    plt.margins(x=0.01)
    plt.subplots_adjust(bottom=0.15)
    if balance_target=="class_targets":
        balance_target = "Class"
    elif balance_target=="lesion_gleasons":
        balance_target = "GS"
    ax.set_title(str(balance_target)+"-Balanced Train Generator: Sampling Likelihood per PID")
    ax.set_axisbelow(True)
    ax.grid(axis='y')
    ax.set_ylabel("Sampling Likelihood (%)")
    ax.set_xlabel("PID")
    plt.tight_layout()

    if out_file is not None:
        plt.savefig(out_file)

    plt.close()

def plot_batchgen_stats(cf, stats, empties, target_name, unique_ts, out_file=None):
    """Plot bar chart showing RoI frequencies and empty-sample count of batch stats recorded by BatchGenerator.
    :param cf: config.
    :param stats: statistics as supplied by BatchGenerator class.
    :param out_file: path to save plot.
    """

    total_samples = cf.num_epochs*cf.num_train_batches*cf.batch_size
    if target_name=="class_targets":
        target_name = "Class"
        label_dict = {cl_id: label for (cl_id, label) in cf.class_id2label.items()}
    elif target_name=="lesion_gleasons":
        target_name = "Lesion's Gleason Score"
        label_dict = cf.gs2label
    elif target_name=="rg_bin_targets":
        target_name = "Regression-Bin ID"
        label_dict = cf.bin_id2label
    else:
        raise NotImplementedError
    names = [label_dict[t_id].name for t_id in unique_ts]
    colors = [label_dict[t_id].color for t_id in unique_ts]

    title = "Training Target Frequencies"
    title += "\nempty samples: {}".format(empties)
    rects = plt.bar(names, stats['roi_counts'], color=colors, alpha=0.9, edgecolor=colors)
    ax = plt.gca()

    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid()
    ax.set_ylabel(r"#RoIs")
    ax.set_xlabel(target_name)

    total_count = np.sum(stats["roi_counts"])
    labels = ["{:.0f}%".format(count/total_count*100) for count in stats["roi_counts"]]
    label_bar(ax, rects, labels, colors)

    if out_file is not None:
        plt.savefig(out_file)

    plt.close()


def view_3D_array(arr, outfile, elev=30, azim=30):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.voxels(arr)
    ax.view_init(elev=elev, azim=azim)

    plt.savefig(outfile)

def view_batch(cf, batch, res_dict=None, out_file=None, legend=True, show_info=True, has_colorchannels=False,
               isRGB=True, show_seg_ids="all", show_seg_pred=True, show_gt_boxes=True, show_gt_labels=False,
               roi_items="all", sample_picks=None, vol_slice_picks=None, box_score_thres=None, plot_mods=True,
               dpi=300, vmin=None, return_fig=False, get_time=True):
    r""" View data and target entries of a batch.

    Batch expected as dic with entries 'data' and 'seg' holding np.arrays of
    size :math:`batch\_size \times modalities \times h \times w` for data
    and :math:`batch\_size \times classes \times h \times w` or
    :math:`batch\_size \times 1 \times h \times w`  for segs.
    Classes, even if just dummy, are always needed for plotting since they determine colors.
    Pyplot expects dimensions in order y,x,chans (height, width, chans) for imshow.

    :param cf: config.
    :param batch: batch.
    :param res_dict: results dictionary.
    :param out_file: path to save plot.
    :param legend: whether to show a legend.
    :param show_info: whether to show text info about img sizes and type in plot.
    :param has_colorchannels: whether image has color channels.
    :param isRGB: if image is RGB.
    :param show_seg_ids: "all" or None or list with seg classes to show (seg_ids)
    :param show_seg_pred: whether to the predicted segmentation.
    :param show_gt_boxes:  whether to show ground-truth boxes.
    :param show_gt_labels: whether to show labels of ground-truth boxes.
    :param roi_items: which roi items to show: strings "all" or "targets". --> all roi items in cf.roi_items or only
     those which are targets, or list holding keys/names of entries in cf.roi_items to plot additionally on roi boxes.
      empty iterator to show none.
    :param sample_picks:  which indices of the batch to display. None for all.
    :param vol_slice_picks: when batch elements are 3D: which slices to display. None for all, or tuples
        ("random", int: amt) / (float€[0,1]: fg_prob, int: amt) for random pick / fg_slices pick w probability fg_prob
        of amt slices. fg pick requires gt seg.
    :param box_score_thres: plot only boxes with pred_score > box_score_thres. None or 0. for no threshold.
    :param plot_mods: whether to plot input modality/modalities.
    :param dpi: graphics resolution.
    :param vmin: min value for gray-scale cmap in imshow, set to a fix value for inter-batch normalization, or None for
     intra-batch.
    :param return_fig: whether to return created figure.
    """
    stime = time.time()
    # pfix = prefix, ptfix = postfix
    patched_patient = 'patch_crop_coords' in list(batch.keys())
    pfix = 'patient_' if patched_patient else ''
    ptfix = '_2d' if (patched_patient and cf.dim == 2 and pfix + 'class_targets_2d' in batch.keys()) else ''
    # -------------- get data, set flags -----------------
    try:
        btype = type(batch[pfix + 'data'])
        data = batch[pfix + 'data'].astype("float32")
        seg = batch[pfix + 'seg']
    except AttributeError: # in this case: assume it's single-annotator ground truths
        btype = type(batch[pfix + 'data'])
        data = batch[pfix + 'data'].astype("float32")
        seg = batch[pfix + 'seg'][0]
        print("Showing only gts of rater 0")

    data_init_shp, seg_init_shp = data.shape, seg.shape
    seg = np.copy(seg) if show_seg_ids else None
    plot_bg = batch['plot_bg'] if 'plot_bg' in batch.keys() and not isinstance(batch['plot_bg'], (int, float)) else None
    plot_bg_chan = batch['plot_bg'] if 'plot_bg' in batch.keys() and isinstance(batch['plot_bg'], (int, float)) else 0
    gt_boxes = batch[pfix+'bb_target'+ptfix] if pfix+'bb_target'+ptfix in batch.keys() and show_gt_boxes else None
    class_targets = batch[pfix+'class_targets'+ptfix] if pfix+'class_targets'+ptfix in batch.keys() else None
    cf_roi_items = [pfix+it+ptfix for it in cf.roi_items]
    if roi_items == "all":
        roi_items = [it for it in cf_roi_items]
    elif roi_items == "targets":
        roi_items = [it for it in cf_roi_items if 'targets' in it]
    else:
        roi_items = [it for it in cf_roi_items if it in roi_items]

    if res_dict is not None:
        seg_preds = res_dict["seg_preds"] if (show_seg_pred is not None and 'seg_preds' in res_dict.keys()
                                              and show_seg_ids) else None
        if '2D_boxes' in res_dict.keys():
            assert cf.dim==2
            pr_boxes = res_dict["2D_boxes"]
        elif 'boxes' in res_dict.keys():
            pr_boxes = res_dict["boxes"]
        else:
            pr_boxes = None
    else:
        seg_preds = None
        pr_boxes = None

    # -------------- get shapes, apply sample selection -----------------
    (n_samples, mods, h, w), d = data.shape[:4], 0

    z_ics = [slice(None)]
    if has_colorchannels: #has to be 2D
        data = np.transpose(data, axes=(0, 2, 3, 1))  # now b,y,x,c
        mods = 1
    else:
        if len(data.shape) == 5:  # 3dim case
            d = data.shape[4]
            if vol_slice_picks is None:
                z_ics = np.arange(0, d)
            elif hasattr(vol_slice_picks, "__iter__") and vol_slice_picks[0]=="random":
                z_ics = np.random.choice(np.arange(0, d), size=min(vol_slice_picks[1], d), replace=False)
            else:
                z_ics = vol_slice_picks

    sample_ics = range(n_samples)
    # 8000 approx value of pixels that are displayable in one figure dim (pyplot has a render limit), depends on dpi however
    if data.shape[0]*data.shape[2]*len(z_ics)>8000:
        n_picks = max(1, int(8000/(data.shape[2]*len(z_ics))))
        if len(z_ics)>1 and vol_slice_picks is None:
            z_ics = np.random.choice(np.arange(0, data.shape[4]),
                                     size=min(data.shape[4], max(1,int(8000/(n_picks*data.shape[2])))), replace=False)
        if sample_picks is None:
            sample_picks = np.random.choice(data.shape[0], n_picks, replace=False)

    if sample_picks is not None:
        sample_ics = [s for s in sample_picks if s in sample_ics]
        n_samples = len(sample_ics)

    if not plot_mods:
        mods = 0
    if show_seg_ids=="all":
        show_seg_ids = np.unique(seg)
    if seg_preds is not None and not type(show_seg_ids)==str:
        seg_preds = np.copy(seg_preds)
        seg_preds = np.where(np.isin(seg_preds, show_seg_ids), seg_preds, 0)
    if seg is not None:
        if not type(show_seg_ids)==str: #to save time
            seg = np.where(np.isin(seg, show_seg_ids), seg, 0)
        legend_items = {cf.seg_id2label[seg_id] for seg_id in np.unique(seg) if seg_id != 0}  # add seg labels
    else:
        legend_items = set()

    # -------------- setup figure -----------------
    if isRGB:
        data = RGB_to_rgb(data)
        if plot_bg is not None:
            plot_bg = RGB_to_rgb(plot_bg)
    n_cols = mods
    if seg is not None or gt_boxes is not None:
        n_cols += 1
    if seg_preds is not None or pr_boxes is not None:
        n_cols += 1

    n_rows = n_samples*len(z_ics)
    grid = gridspec.GridSpec(n_rows, n_cols, wspace=0.01, hspace=0.0)
    fig = plt.figure(figsize=((n_cols + 1)*2, n_rows*2), tight_layout=True)
    title_fs = 12  # fontsize

    sample_ics, z_ics = sorted(sample_ics), sorted(z_ics)
    row = 0  # current row
    for s_count, s_ix in enumerate(sample_ics):
        for z_ix in z_ics:
            col = 0  # current col
            # ----visualise input data -------------
            if has_colorchannels:
                if plot_mods:
                    ax = fig.add_subplot(grid[row, col])
                    ax.imshow(data[s_ix][...,z_ix])
                    ax.axis("off")
                    if row == 0:
                        plt.title("Input", fontsize=title_fs)
                    if col == 0:
                        specs = batch.get('spec', batch['pid'])
                        intra_patient_ix = s_ix if type(z_ix) == slice else z_ix
                        ylabel = str(specs[s_ix])[-5:] + "/" + str(intra_patient_ix) if show_info else str(specs[s_ix])[-5:]
                        ax.set_ylabel("{:s}".format(ylabel), fontsize=title_fs)  # show id-number
                    col += 1
                bg_img = plot_bg[s_ix][...,z_ix] if plot_bg is not None else data[s_ix][...,z_ix]
            else:
                for mod in range(mods):
                    ax = fig.add_subplot(grid[row, col])
                    ax.imshow(data[s_ix, mod][...,z_ix], cmap="gray", vmin=vmin)
                    suppress_axes_lines(ax)
                    if row == 0:
                        plt.title("Mod. " + str(mod), fontsize=title_fs)
                    if col == 0:
                        specs = batch.get('spec', batch['pid'])
                        intra_patient_ix = s_ix if type(z_ix)==slice else z_ix
                        ylabel = str(specs[s_ix])[-5:]+"/"+str(intra_patient_ix) if show_info else str(specs[s_ix])[-5:]
                        ax.set_ylabel("{:s}".format(ylabel), fontsize=title_fs)  # show id-number
                    col += 1
                bg_img = plot_bg[s_ix][...,z_ix] if plot_bg is not None else data[s_ix, plot_bg_chan][...,z_ix]

            # ---evtly visualise groundtruths-------------------
            if seg is not None or gt_boxes is not None:
                # img as bg for gt
                ax = fig.add_subplot(grid[row, col])
                ax.imshow(bg_img, cmap="gray", vmin=vmin)
                if row == 0:
                    plt.title("Ground Truth", fontsize=title_fs)
                if col == 0:
                    specs = batch.get('spec', batch['pid'])
                    intra_patient_ix = s_ix if type(z_ix) == slice else z_ix
                    ylabel = str(specs[s_ix])[-5:] + "/" + str(intra_patient_ix) if show_info else str(specs[s_ix])[-5:]
                    ax.set_ylabel("{:s}".format(ylabel), fontsize=title_fs)  # show id-number
                    suppress_axes_lines(ax)
                else:
                    plt.axis('off')
                col += 1

            if seg is not None and seg.shape[1] == 1:
                ax.imshow(to_rgba(seg[s_ix][0][...,z_ix], cf.cmap), alpha=0.8)
            elif seg is not None:
                ax.imshow(to_rgba(np.argmax(seg[s_ix][...,z_ix], axis=0), cf.cmap), alpha=0.8)

            # gt bounding boxes
            if gt_boxes is not None and len(gt_boxes[s_ix]) > 0:
                for j, box in enumerate(gt_boxes[s_ix]):
                    if d > 0:
                        [z1, z2] = box[4:]
                        if not (z1<=z_ix and z_ix<=z2):
                            box = []
                    if len(box) > 0:
                        [y1, x1, y2, x2] = box[:4]
                        width, height = x2 - x1, y2 - y1
                        if class_targets is not None:
                            label = cf.class_id2label[class_targets[s_ix][j]]
                            legend_items.add(label)
                            if show_gt_labels:
                                text_poss, p = [(x1, y1), (x1, (y1+y2)//2)], 0
                                text_fs = title_fs // 3
                                if roi_items is not None:
                                    for name in roi_items:
                                        if name in cf_roi_items and batch[name][s_ix][j] is not None:
                                            if 'class_targets' in name and cf.plot_class_ids:
                                                text_x = x2 #- 2 * text_fs * (len(str(class_targets[s_ix][j])))  # avoid overlap of scores
                                                text_y = y1 #+ 2 * text_fs
                                                text_str = '{}'.format(class_targets[s_ix][j])
                                            elif 'regression_targets' in name:
                                                text_x, text_y = (x2, y2)
                                                text_str = "[" + " ".join(
                                                    ["{:.1f}".format(x) for x in batch[name][s_ix][j]]) + "]"
                                            elif 'rg_bin_targets' in name:
                                                text_x, text_y = (x1, y2)
                                                text_str = '{}'.format(batch[name][s_ix][j])
                                            else:
                                                text_pos = text_poss.pop(0)
                                                text_x = text_pos[0] #- 2 * text_fs * len(str(batch[name][s_ix][j]))
                                                text_y = text_pos[1] #+ 2 * text_fs
                                                text_str = '{}'.format(batch[name][s_ix][j])

                                            ax.text(text_x, text_y, text_str, color=cf.white, fontsize=text_fs,
                                                    bbox=dict(facecolor=label.color, alpha=0.7, edgecolor='none', clip_on=True,
                                                              pad=0))
                                            p+=1
                            bbox = mpatches.Rectangle((x1, y1), width, height, linewidth=0.6, edgecolor=label.color,
                                                      facecolor='none')
                            ax.add_patch(bbox)

            # -----evtly visualise predictions -------------
            if pr_boxes is not None or seg_preds is not None:
                ax = fig.add_subplot(grid[row, col])
                ax.imshow(bg_img, cmap="gray")
                ax.axis("off")
                col += 1
                if row == 0:
                    plt.title("Prediction", fontsize=title_fs)
            # ---------- pred boxes  -------------------------
            if pr_boxes is not None and len(pr_boxes[s_ix]) > 0:
                box_score_thres = cf.min_det_thresh if box_score_thres is None else box_score_thres
                for j, box in enumerate(pr_boxes[s_ix]):
                    plot_box = box["box_type"] in ["det", "prop"]  # , "pos_anchor", "neg_anchor"]
                    if box["box_type"] == "det" and (float(box["box_score"]) <= box_score_thres or box["box_pred_class_id"] == 0):
                        plot_box = False

                    if plot_box:
                        if d > 0:
                            [z1, z2] = box["box_coords"][4:]
                            if not (z1<=z_ix and z_ix<=z2):
                                box = []
                        if len(box) > 0:
                            [y1, x1, y2, x2] = box["box_coords"][:4]

                            width, height = x2 - x1, y2 - y1

                            if box["box_type"] == "det":
                                label = cf.class_id2label[box["box_pred_class_id"]]
                                legend_items.add(label)
                                text_x, text_y = x2, y1
                                id_text = str(box["box_pred_class_id"]) + "|" if cf.plot_class_ids else ""
                                text_str = '{}{:.0f}'.format(id_text, box["box_score"] * 100)
                                text_settings = dict(facecolor=label.color, alpha=0.5, edgecolor='none', clip_on=True,
                                                     pad=0)
                                ax.text(text_x, text_y, text_str, color=cf.white,
                                        bbox=text_settings, fontsize=title_fs // 4)
                                edgecolor = label.color
                                if 'regression' in box.keys():
                                    text_x, text_y = x2, y2
                                    id_text = "["+" ".join(["{:.1f}".format(x) for x in box["regression"]])+"]" #str(box["regression"]) #+ "|" if cf.plot_class_ids else ""
                                    if 'rg_uncertainty' in box.keys() and not np.isnan(box['rg_uncertainty']):
                                        id_text += " | {:.1f}".format(box['rg_uncertainty'])
                                    text_str = '{}'.format(id_text) #, box["box_score"] * 100)
                                    text_settings = dict(facecolor=label.color, alpha=0.5, edgecolor='none',
                                                         clip_on=True, pad=0)
                                    ax.text(text_x, text_y, text_str, color=cf.white,
                                            bbox=text_settings, fontsize=title_fs // 4)
                                if 'rg_bin' in box.keys():
                                    text_x, text_y = x1, y2
                                    text_str = '{}'.format(box["rg_bin"])
                                    text_settings = dict(facecolor=label.color, alpha=0.5, edgecolor='none',
                                                         clip_on=True, pad=0)
                                    ax.text(text_x, text_y, text_str, color=cf.white,
                                            bbox=text_settings, fontsize=title_fs // 4)
                            else:
                                label = cf.box_type2label[box["box_type"]]
                                legend_items.add(label)
                                edgecolor = label.color

                            bbox = mpatches.Rectangle((x1, y1), width, height, linewidth=0.6, edgecolor=edgecolor,
                                                      facecolor='none')
                            ax.add_patch(bbox)
            # ------------ pred segs --------
            if seg_preds is not None:  # and seg_preds.shape[1] == 1:
                if cf.class_specific_seg:
                    ax.imshow(to_rgba(seg_preds[s_ix][0][...,z_ix], cf.cmap), alpha=0.8)
                else:
                    ax.imshow(bin_seg_to_rgba(seg_preds[s_ix][0][...,z_ix], cf.orange), alpha=0.8)

            row += 1

    # -----actions for all batch entries----------
    if legend and len(legend_items) > 0:
        patches = []
        for label in legend_items:
            if cf.plot_class_ids and type(label) != type(cf.box_labels[0]):
                id_text = str(label.id) + ":"
            else:
                id_text = ""

            patches.append(mpatches.Patch(color=label.color, label="{}{:.10s}".format(id_text, label.name)))
        # assumes one image gives enough y-space for 5 legend items
        ncols = max(1, len(legend_items) // (5 * n_samples))
        plt.figlegend(handles=patches, loc="upper center", bbox_to_anchor=(0.99, 0.86),
                      borderaxespad=0., ncol=ncols, bbox_transform=fig.transFigure,
                      fontsize=int(2/3*title_fs))
        # fig.set_size_inches(mods+3+ncols-1,1.5+1.2*n_samples)

    if show_info:
        plt.figtext(0, 0, "Batch content is of type\n{}\nand has shapes\n".format(btype) + \
                    "{} for 'data' and {} for 'seg'".format(data_init_shp, seg_init_shp))

    if out_file is not None:
        if cf.server_env:
            IO_safe(plt.savefig, fname=out_file, dpi=dpi, pad_inches=0.0, bbox_inches='tight', _raise=False)
        else:
            plt.savefig(out_file, dpi=dpi, pad_inches=0.0, bbox_inches='tight')
    if get_time:
        print("generated {} in {:.3f}s".format("plot" if not isinstance(get_time, str) else get_time, time.time()-stime))
    if return_fig:
        return plt.gcf()
    plt.clf()
    plt.close()

def view_batch_paper(cf, batch, res_dict=None, out_file=None, legend=True, show_info=True, has_colorchannels=False,
               isRGB=True, show_seg_ids="all", show_seg_pred=True, show_gt_boxes=True, show_gt_labels=False,
               roi_items="all", split_ens_ics=False, server_env=True, sample_picks=None, vol_slice_picks=None,
               patient_items=False, box_score_thres=None, plot_mods=True, dpi=400, vmin=None, return_fig=False):
    r"""view data and target entries of a batch.

    batch expected as dic with entries 'data' and 'seg' holding tensors or nparrays of
    size :math:`batch\_size \times modalities \times h \times w` for data
    and :math:`batch\_size \times classes \times h \times w` or
    :math:`batch\_size \times 1 \times h \times w`  for segs.
    Classes, even if just dummy, are always needed for plotting since they determine colors.

    :param cf:
    :param batch:
    :param res_dict:
    :param out_file:
    :param legend:
    :param show_info:
    :param has_colorchannels:
    :param isRGB:
    :param show_seg_ids:
    :param show_seg_pred:
    :param show_gt_boxes:
    :param show_gt_labels:
    :param roi_items: strings "all" or "targets" --> all roi items in cf.roi_items or only those which are targets, or
        list holding keys/names of entries in cf.roi_items to plot additionally on roi boxes. empty iterator
        to show none.
    :param split_ens_ics:
    :param server_env:
    :param sample_picks: which indices of the batch to display. None for all.
    :param vol_slice_picks: when batch elements are 3D: which slices to display. None for all, or tuples
        ("random", int: amt) / (float€[0,1]: fg_prob, int: amt) for random pick / fg_slices pick w probability fg_prob
        of amt slices. fg pick requires gt seg.
    :param patient_items: set to true if patient-wise batch items should be displayed (need to be contained in batch
        and marked via 'patient_' prefix.
    :param box_score_thres:  plot only boxes with pred_score > box_score_thres. None or 0. for no thres.
    :param plot_mods:
    :param dpi: graphics resolution
    :param vmin: min value for gs cmap in imshow, set to fix inter-batch, or None for intra-batch.

    pyplot expects dimensions in order y,x,chans (height, width, chans) for imshow.
    show_seg_ids: "all" or None or list with seg classes to show (seg_ids)

    """
    # pfix = prefix, ptfix = postfix
    pfix = 'patient_' if patient_items else ''
    ptfix = '_2d' if (patient_items and cf.dim==2) else ''

    # -------------- get data, set flags -----------------

    btype = type(batch[pfix + 'data'])
    data = batch[pfix + 'data'].astype("float32")
    seg = batch[pfix + 'seg']

    # seg = np.array(seg).mean(axis=0, keepdims=True)
    # seg[seg>0] = 1.

    print("Showing multirater GT")
    data_init_shp, seg_init_shp = data.shape, seg.shape
    fg_slices = np.where(np.sum(np.sum(np.squeeze(seg), axis=0), axis=0)>0)[0]

    if len(fg_slices)==0:
        print("skipping empty patient")
        return
    if vol_slice_picks is None:
        vol_slice_picks = fg_slices

    print("data shp, seg shp", data_init_shp, seg_init_shp)

    plot_bg = batch['plot_bg'] if 'plot_bg' in batch.keys() and not isinstance(batch['plot_bg'], (int, float)) else None
    plot_bg_chan = batch['plot_bg'] if 'plot_bg' in batch.keys() and isinstance(batch['plot_bg'], (int, float)) else 0
    gt_boxes = batch[pfix+'bb_target'+ptfix] if pfix+'bb_target'+ptfix in batch.keys() and show_gt_boxes else None
    class_targets = batch[pfix+'class_targets'+ptfix] if pfix+'class_targets'+ptfix in batch.keys() else None
    cf_roi_items = [pfix+it+ptfix for it in cf.roi_items]
    if roi_items == "all":
        roi_items = [it for it in cf_roi_items]
    elif roi_items == "targets":
        roi_items = [it for it in cf_roi_items if 'targets' in it]
    else:
        roi_items = [it for it in cf_roi_items if it in roi_items]

    if res_dict is not None:
        seg_preds = res_dict["seg_preds"] if (show_seg_pred is not None and 'seg_preds' in res_dict.keys()
                                              and show_seg_ids) else None
        if '2D_boxes' in res_dict.keys():
            assert cf.dim==2
            pr_boxes = res_dict["2D_boxes"]
        elif 'boxes' in res_dict.keys():
            pr_boxes = res_dict["boxes"]
        else:
            pr_boxes = None
    else:
        seg_preds = None
        pr_boxes = None

    # -------------- get shapes, apply sample selection -----------------
    (n_samples, mods, h, w), d = data.shape[:4], 0

    z_ics = [slice(None)]
    if has_colorchannels: #has to be 2D
        data = np.transpose(data, axes=(0, 2, 3, 1))  # now b,y,x,c
        mods = 1
    else:
        if len(data.shape) == 5:  # 3dim case
            d = data.shape[4]
            if vol_slice_picks is None:
                z_ics = np.arange(0, d)
            # elif hasattr(vol_slice_picks, "__iter__") and vol_slice_picks[0]=="random":
            #     z_ics = np.random.choice(np.arange(0, d), size=min(vol_slice_picks[1], d), replace=False)
            else:
                z_ics = vol_slice_picks

    sample_ics = range(n_samples)
    # 8000 approx value of pixels that are displayable in one figure dim (pyplot has a render limit), depends on dpi however
    if data.shape[0]*data.shape[2]*len(z_ics)>8000:
        n_picks = max(1, int(8000/(data.shape[2]*len(z_ics))))
        if len(z_ics)>1:
            if vol_slice_picks is None:
                z_ics = np.random.choice(np.arange(0, data.shape[4]),
                                         size=min(data.shape[4], max(1,int(8000/(n_picks*data.shape[2])))), replace=False)
            else:
                z_ics = np.random.choice(vol_slice_picks,
                                         size=min(len(vol_slice_picks), max(1,int(8000/(n_picks*data.shape[2])))), replace=False)

        if sample_picks is None:
            sample_picks = np.random.choice(data.shape[0], n_picks, replace=False)

    if sample_picks is not None:
        sample_ics = [s for s in sample_picks if s in sample_ics]
        n_samples = len(sample_ics)

    if not plot_mods:
        mods = 0
    if show_seg_ids=="all":
        show_seg_ids = np.unique(seg)

    legend_items = set()

    # -------------- setup figure -----------------
    if isRGB:
        data = RGB_to_rgb(data)
        if plot_bg is not None:
            plot_bg = RGB_to_rgb(plot_bg)
    n_cols = mods
    if seg is not None or gt_boxes is not None:
        n_cols += 1
    if seg_preds is not None or pr_boxes is not None:
        n_cols += 1

    n_rows = n_samples*len(z_ics)
    grid = gridspec.GridSpec(n_rows, n_cols, wspace=0.01, hspace=0.0)
    fig = plt.figure(figsize=((n_cols + 1)*2, n_rows*2), tight_layout=True)
    title_fs = 12  # fontsize

    sample_ics, z_ics = sorted(sample_ics), sorted(z_ics)
    row = 0  # current row
    for s_count, s_ix in enumerate(sample_ics):
        for z_ix in z_ics:
            col = 0  # current col
            # ----visualise input data -------------
            if has_colorchannels:
                if plot_mods:
                    ax = fig.add_subplot(grid[row, col])
                    ax.imshow(data[s_ix][...,z_ix])
                    ax.axis("off")
                    if row == 0:
                        plt.title("Input", fontsize=title_fs)
                    if col == 0:
                        # key = "spec" if "spec" in batch.keys() else "pid"
                        specs = batch.get('spec', batch['pid'])
                        intra_patient_ix = s_ix if type(z_ix) == slice else z_ix
                        ylabel = str(specs[s_ix])[-5:] + "/" + str(intra_patient_ix) if show_info else str(specs[s_ix])[-5:]
                        ax.set_ylabel("{:s}".format(ylabel), fontsize=title_fs)  # show id-number
                    col += 1
                bg_img = plot_bg[s_ix][...,z_ix] if plot_bg is not None else data[s_ix][...,z_ix]
            else:
                for mod in range(mods):
                    ax = fig.add_subplot(grid[row, col])
                    ax.imshow(data[s_ix, mod][...,z_ix], cmap="gray", vmin=vmin)
                    suppress_axes_lines(ax)
                    if row == 0:
                        plt.title("Mod. " + str(mod), fontsize=title_fs)
                    if col == 0:
                        # key = "spec" if "spec" in batch.keys() else "pid"
                        specs = batch.get('spec', batch['pid'])
                        intra_patient_ix = s_ix if type(z_ix)==slice else z_ix
                        ylabel = str(specs[s_ix])[-5:]+"/"+str(intra_patient_ix) if show_info else str(specs[s_ix])[-5:]
                        ax.set_ylabel("{:s}".format(ylabel), fontsize=title_fs)  # show id-number
                    col += 1
                bg_img = plot_bg[s_ix][...,z_ix] if plot_bg is not None else data[s_ix, plot_bg_chan][...,z_ix]

            # ---evtly visualise groundtruths-------------------
            if seg is not None or gt_boxes is not None:
                # img as bg for gt
                ax = fig.add_subplot(grid[row, col])
                ax.imshow(bg_img, cmap="gray", vmin=vmin)
                if row == 0:
                    plt.title("Ground Truth+ Pred", fontsize=title_fs)
                if col == 0:
                    specs = batch.get('spec', batch['pid'])
                    intra_patient_ix = s_ix if type(z_ix) == slice else z_ix
                    ylabel = str(specs[s_ix])[-5:] + "/" + str(intra_patient_ix) if show_info else str(specs[s_ix])[-5:]
                    ax.set_ylabel("{:s}".format(ylabel), fontsize=title_fs)  # show id-number
                    suppress_axes_lines(ax)
                else:
                    plt.axis('off')
                col += 1

            if seg is not None and seg.shape[1] == 1:
                cmap = {1: cf.orange}
                ax.imshow(to_rgba(seg[s_ix][0][...,z_ix], cmap), alpha=0.8)

            # gt bounding boxes
            if gt_boxes is not None and len(gt_boxes[s_ix]) > 0:
                for j, box in enumerate(gt_boxes[s_ix]):
                    if d > 0:
                        [z1, z2] = box[4:]
                        if not (z1<=z_ix and z_ix<=z2):
                            box = []
                    if len(box) > 0:
                        [y1, x1, y2, x2] = box[:4]
                        # [x1,y1,x2,y2] = box[:4]#:return: coords (x1, y1, x2, y2)
                        width, height = x2 - x1, y2 - y1
                        if class_targets is not None:
                            label = cf.class_id2label[class_targets[s_ix][j]]
                            legend_items.add(label)
                            if show_gt_labels and cf.plot_class_ids:
                                text_poss, p = [(x1, y1), (x1, (y1+y2)//2)], 0
                                text_fs = title_fs // 3
                                if roi_items is not None:
                                    for name in roi_items:
                                        if name in cf_roi_items and batch[name][s_ix][j] is not None:
                                            if 'class_targets' in name:
                                                text_x = x2 #- 2 * text_fs * (len(str(class_targets[s_ix][j])))  # avoid overlap of scores
                                                text_y = y1 #+ 2 * text_fs
                                                text_str = '{}'.format(class_targets[s_ix][j])
                                            elif 'regression_targets' in name:
                                                text_x, text_y = (x2, y2)
                                                text_str = "[" + " ".join(
                                                    ["{:.1f}".format(x) for x in batch[name][s_ix][j]]) + "]"
                                            elif 'rg_bin_targets' in name:
                                                text_x, text_y = (x1, y2)
                                                text_str = '{}'.format(batch[name][s_ix][j])
                                            else:
                                                text_pos = text_poss.pop(0)
                                                text_x = text_pos[0] #- 2 * text_fs * len(str(batch[name][s_ix][j]))
                                                text_y = text_pos[1] #+ 2 * text_fs
                                                text_str = '{}'.format(batch[name][s_ix][j])

                                            ax.text(text_x, text_y, text_str, color=cf.black if label.color==cf.yellow else cf.white, fontsize=text_fs,
                                                    bbox=dict(facecolor=label.color, alpha=0.7, edgecolor='none', clip_on=True,
                                                              pad=0))
                                            p+=1
                            bbox = mpatches.Rectangle((x1, y1), width, height, linewidth=0.6, edgecolor=label.color,
                                                      facecolor='none')
                            ax.add_patch(bbox)

            # # -----evtly visualise predictions -------------
            # if pr_boxes is not None or seg_preds is not None:
            #     ax = fig.add_subplot(grid[row, col])
            #     ax.imshow(bg_img, cmap="gray")
            #     ax.axis("off")
            #     col += 1
            #     if row == 0:
            #         plt.title("Prediction", fontsize=title_fs)



            # ---------- pred boxes  -------------------------
            if pr_boxes is not None and len(pr_boxes[s_ix]) > 0:
                box_score_thres = cf.min_det_thresh if box_score_thres is None else box_score_thres
                for j, box in enumerate(pr_boxes[s_ix]):
                    plot_box = box["box_type"] in ["det", "prop"]  # , "pos_anchor", "neg_anchor"]
                    if box["box_type"] == "det" and (float(box["box_score"]) <= box_score_thres or box["box_pred_class_id"] == 0):
                        plot_box = False

                    if plot_box:
                        if d > 0:
                            [z1, z2] = box["box_coords"][4:]
                            if not (z1<=z_ix and z_ix<=z2):
                                box = []
                        if len(box) > 0:
                            [y1, x1, y2, x2] = box["box_coords"][:4]

                            width, height = x2 - x1, y2 - y1

                            if box["box_type"] == "det":
                                label = cf.bin_id2label[box["rg_bin"]]
                                color = cf.aubergine
                                legend_items.add(label)
                                text_x, text_y = x2, y1
                                #id_text = str(box["box_pred_class_id"]) + "|" if cf.plot_class_ids else ""
                                id_text = "fg: "
                                text_str = '{}{:.0f}'.format(id_text, box["box_score"] * 100)
                                text_settings = dict(facecolor=color, alpha=0.5, edgecolor='none', clip_on=True,
                                                     pad=0.2)
                                ax.text(text_x, text_y, text_str, color=cf.black if label.color==cf.yellow else cf.white,
                                        bbox=text_settings, fontsize=title_fs // 2)
                                edgecolor = color #label.color
                                if 'regression' in box.keys():
                                    text_x, text_y = x2, y2
                                    id_text = "ms: "+" ".join(["{:.1f}".format(x) for x in box["regression"]])+""
                                    text_str = '{}'.format(id_text) #, box["box_score"] * 100)
                                    text_settings = dict(facecolor=color, alpha=0.5, edgecolor='none',
                                                         clip_on=True, pad=0.2)
                                    ax.text(text_x, text_y, text_str, color=cf.black if label.color==cf.yellow else cf.white,
                                            bbox=text_settings, fontsize=title_fs // 2)
                                if 'rg_bin' in box.keys():
                                    text_x, text_y = x1, y2
                                    text_str = '{}'.format(box["rg_bin"])
                                    text_settings = dict(facecolor=label.color, alpha=0.5, edgecolor='none',
                                                         clip_on=True, pad=0)
                                    # ax.text(text_x, text_y, text_str, color=cf.white,
                                    #         bbox=text_settings, fontsize=title_fs // 4)
                                if split_ens_ics and "ens_ix" in box.keys():
                                    n_aug = box["ens_ix"].split("_")[1]
                                    edgecolor = [c for c in cf.color_palette if not c == cf.green][
                                        int(n_aug) % (len(cf.color_palette) - 1)]
                                    text_x, text_y = x1, y2
                                    text_str = "{}".format(box["ens_ix"][2:])
                                    ax.text(text_x, text_y, text_str, color=cf.white, bbox=text_settings,
                                            fontsize=title_fs // 6)
                            else:
                                label = cf.box_type2label[box["box_type"]]
                                legend_items.add(label)
                                edgecolor = label.color

                            bbox = mpatches.Rectangle((x1, y1), width, height, linewidth=0.6, edgecolor=edgecolor,
                                                      facecolor='none')
                            ax.add_patch(bbox)
            row += 1

    # -----actions for all batch entries----------
    if legend and len(legend_items) > 0:
        patches = []
        for label in legend_items:
            if cf.plot_class_ids and type(label) != type(cf.box_labels[0]):
                id_text = str(label.id) + ":"
            else:
                id_text = ""

            patches.append(mpatches.Patch(color=label.color, label="{}{:.10s}".format(id_text, label.name)))
        # assumes one image gives enough y-space for 5 legend items
        ncols = max(1, len(legend_items) // (5 * n_samples))
        plt.figlegend(handles=patches, loc="upper center", bbox_to_anchor=(0.99, 0.86),
                      borderaxespad=0., ncol=ncols, bbox_transform=fig.transFigure,
                      fontsize=int(2/3*title_fs))
        # fig.set_size_inches(mods+3+ncols-1,1.5+1.2*n_samples)

    if show_info:
        plt.figtext(0, 0, "Batch content is of type\n{}\nand has shapes\n".format(btype) + \
                    "{} for 'data' and {} for 'seg'".format(data_init_shp, seg_init_shp))

    if out_file is not None:
        plt.savefig(out_file, dpi=dpi, pad_inches=0.0, bbox_inches='tight', tight_layout=True)
    if return_fig:
        return plt.gcf()
    if not (server_env or cf.server_env):
        plt.show()
    plt.clf()
    plt.close()

def view_batch_thesis(cf, batch, res_dict=None, out_file=None, legend=True, has_colorchannels=False,
               isRGB=True, show_seg_ids="all", show_seg_pred=True, show_gt_boxes=True, show_gt_labels=False, show_cl_ids=True,
               roi_items="all", server_env=True, sample_picks=None, vol_slice_picks=None, fontsize=12, seg_cmap="class",
               patient_items=False, box_score_thres=None, plot_mods=True, dpi=400, vmin=None, return_fig=False, axes=None):
    r"""view data and target entries of a batch.

    batch expected as dic with entries 'data' and 'seg' holding tensors or nparrays of
    size :math:`batch\_size \times modalities \times h \times w` for data
    and :math:`batch\_size \times classes \times h \times w` or
    :math:`batch\_size \times 1 \times h \times w`  for segs.
    Classes, even if just dummy, are always needed for plotting since they determine colors.

    :param cf:
    :param batch:
    :param res_dict:
    :param out_file:
    :param legend:
    :param show_info:
    :param has_colorchannels:
    :param isRGB:
    :param show_seg_ids:
    :param show_seg_pred:
    :param show_gt_boxes:
    :param show_gt_labels:
    :param roi_items: strings "all" or "targets" --> all roi items in cf.roi_items or only those which are targets, or
        list holding keys/names of entries in cf.roi_items to plot additionally on roi boxes. empty iterator
        to show none.
    :param split_ens_ics:
    :param server_env:
    :param sample_picks: which indices of the batch to display. None for all.
    :param vol_slice_picks: when batch elements are 3D: which slices to display. None for all, or tuples
        ("random", int: amt) / (float€[0,1]: fg_prob, int: amt) for random pick / fg_slices pick w probability fg_prob
        of amt slices. fg pick requires gt seg.
    :param patient_items: set to true if patient-wise batch items should be displayed (need to be contained in batch
        and marked via 'patient_' prefix.
    :param box_score_thres:  plot only boxes with pred_score > box_score_thres. None or 0. for no thres.
    :param plot_mods:
    :param dpi: graphics resolution
    :param vmin: min value for gs cmap in imshow, set to fix inter-batch, or None for intra-batch.

    pyplot expects dimensions in order y,x,chans (height, width, chans) for imshow.
    show_seg_ids: "all" or None or list with seg classes to show (seg_ids)

    """
    # pfix = prefix, ptfix = postfix
    pfix = 'patient_' if patient_items else ''
    ptfix = '_2d' if (patient_items and cf.dim==2) else ''

    # -------------- get data, set flags -----------------

    btype = type(batch[pfix + 'data'])
    data = batch[pfix + 'data'].astype("float32")
    seg = batch[pfix + 'seg']

    data_init_shp, seg_init_shp = data.shape, seg.shape
    fg_slices = np.where(np.sum(np.sum(np.squeeze(seg), axis=0), axis=0)>0)[0]

    if len(fg_slices)==0:
        print("skipping empty patient")
        return
    if vol_slice_picks is None:
        vol_slice_picks = fg_slices

    #print("data shp, seg shp", data_init_shp, seg_init_shp)

    plot_bg = batch['plot_bg'] if 'plot_bg' in batch.keys() and not isinstance(batch['plot_bg'], (int, float)) else None
    plot_bg_chan = batch['plot_bg'] if 'plot_bg' in batch.keys() and isinstance(batch['plot_bg'], (int, float)) else 0
    gt_boxes = batch[pfix+'bb_target'+ptfix] if pfix+'bb_target'+ptfix in batch.keys() and show_gt_boxes else None
    class_targets = batch[pfix+'class_targets'+ptfix] if pfix+'class_targets'+ptfix in batch.keys() else None
    cl_targets_sa = batch[pfix+'class_targets_sa'+ptfix] if pfix+'class_targets_sa'+ptfix in batch.keys() else None
    cf_roi_items = [pfix+it+ptfix for it in cf.roi_items]
    if roi_items == "all":
        roi_items = [it for it in cf_roi_items]
    elif roi_items == "targets":
        roi_items = [it for it in cf_roi_items if 'targets' in it]
    else:
        roi_items = [it for it in cf_roi_items if it in roi_items]

    if res_dict is not None:
        seg_preds = res_dict["seg_preds"] if (show_seg_pred is not None and 'seg_preds' in res_dict.keys()
                                              and show_seg_ids) else None
        if '2D_boxes' in res_dict.keys():
            assert cf.dim==2
            pr_boxes = res_dict["2D_boxes"]
        elif 'boxes' in res_dict.keys():
            pr_boxes = res_dict["boxes"]
        else:
            pr_boxes = None
    else:
        seg_preds = None
        pr_boxes = None

    # -------------- get shapes, apply sample selection -----------------
    (n_samples, mods, h, w), d = data.shape[:4], 0

    z_ics = [slice(None)]
    if has_colorchannels: #has to be 2D
        data = np.transpose(data, axes=(0, 2, 3, 1))  # now b,y,x,c
        mods = 1
    else:
        if len(data.shape) == 5:  # 3dim case
            d = data.shape[4]
            if vol_slice_picks is None:
                z_ics = np.arange(0, d)
            else:
                z_ics = vol_slice_picks

    sample_ics = range(n_samples)
    # 8000 approx value of pixels that are displayable in one figure dim (pyplot has a render limit), depends on dpi however
    if data.shape[0]*data.shape[2]*len(z_ics)>8000:
        n_picks = max(1, int(8000/(data.shape[2]*len(z_ics))))
        if len(z_ics)>1 and vol_slice_picks is None:
            z_ics = np.random.choice(np.arange(0, data.shape[4]),
                                     size=min(data.shape[4], max(1,int(8000/(n_picks*data.shape[2])))), replace=False)
        if sample_picks is None:
            sample_picks = np.random.choice(data.shape[0], n_picks, replace=False)

    if sample_picks is not None:
        sample_ics = [s for s in sample_picks if s in sample_ics]
        n_samples = len(sample_ics)

    if not plot_mods:
        mods = 0
    if show_seg_ids=="all":
        show_seg_ids = np.unique(seg)

    legend_items = set()

    # -------------- setup figure -----------------
    if isRGB:
        data = RGB_to_rgb(data)
        if plot_bg is not None:
            plot_bg = RGB_to_rgb(plot_bg)
    n_cols = mods
    if seg is not None or gt_boxes is not None:
        n_cols += 1
    if seg_preds is not None or pr_boxes is not None:
        n_cols += 1

    n_rows = n_samples*len(z_ics)
    grid = gridspec.GridSpec(n_rows, n_cols, wspace=0.01, hspace=0.0)
    fig = plt.figure(figsize=((n_cols + 1)*2, n_rows*2), tight_layout=True)
    title_fs = fontsize  # fontsize
    text_fs = title_fs * 2 / 3

    sample_ics, z_ics = sorted(sample_ics), sorted(z_ics)
    row = 0  # current row
    for s_count, s_ix in enumerate(sample_ics):
        for z_ix in z_ics:
            col = 0  # current col
            # ----visualise input data -------------
            if has_colorchannels:
                if plot_mods:
                    ax = fig.add_subplot(grid[row, col])
                    ax.imshow(data[s_ix][...,z_ix])
                    ax.axis("off")
                    if row == 0:
                        plt.title("Input", fontsize=title_fs)
                    if col == 0:
                        # key = "spec" if "spec" in batch.keys() else "pid"
                        specs = batch.get('spec', batch['pid'])
                        intra_patient_ix = s_ix if type(z_ix) == slice else z_ix
                        ylabel = str(specs[s_ix])[-5:] + "/" + str(intra_patient_ix) if show_info else str(specs[s_ix])[-5:]
                        ax.set_ylabel("{:s}".format(ylabel), fontsize=title_fs)  # show id-number
                    col += 1
                bg_img = plot_bg[s_ix][...,z_ix] if plot_bg is not None else data[s_ix][...,z_ix]
            else:
                for mod in range(mods):
                    ax = fig.add_subplot(grid[row, col])
                    ax.imshow(data[s_ix, mod][...,z_ix], cmap="gray", vmin=vmin)
                    suppress_axes_lines(ax)
                    if row == 0:
                        plt.title("Mod. " + str(mod), fontsize=title_fs)
                    if col == 0:
                        # key = "spec" if "spec" in batch.keys() else "pid"
                        specs = batch.get('spec', batch['pid'])
                        intra_patient_ix = s_ix if type(z_ix)==slice else z_ix
                        ylabel = str(specs[s_ix])[-5:]+"/"+str(intra_patient_ix)
                        ax.set_ylabel("{:s}".format(ylabel), fontsize=title_fs)  # show id-number
                    col += 1
                bg_img = plot_bg[s_ix][...,z_ix] if plot_bg is not None else data[s_ix, plot_bg_chan][...,z_ix]

            # ---evtly visualise groundtruths-------------------
            if seg is not None or gt_boxes is not None:
                # img as bg for gt
                if axes is not None and 'gt' in axes.keys():
                    ax = axes['gt']
                else:
                    ax = fig.add_subplot(grid[row, col])
                ax.imshow(bg_img, cmap="gray", vmin=vmin)
                if row == 0:
                    ax.set_title("Ground Truth", fontsize=title_fs)
                if col == 0:
                    # key = "spec" if "spec" in batch.keys() else "pid"
                    specs = batch.get('spec', batch['pid'])
                    intra_patient_ix = s_ix if type(z_ix) == slice else z_ix
                    ylabel = str(specs[s_ix])[-5:] + "/" + str(intra_patient_ix) # str(specs[s_ix])[-5:]
                    ax.set_ylabel("{:s}".format(ylabel), fontsize=text_fs*1.3)  # show id-number
                    suppress_axes_lines(ax)
                else:
                    ax.axis('off')
                col += 1

            # gt bounding boxes
            if gt_boxes is not None and len(gt_boxes[s_ix]) > 0:
                for j, box in enumerate(gt_boxes[s_ix]):
                    if d > 0:
                        [z1, z2] = box[4:]
                        if not (z1<=z_ix and z_ix<=z2):
                            box = []
                    if len(box) > 0:
                        [y1, x1, y2, x2] = box[:4]
                        # [x1,y1,x2,y2] = box[:4]#:return: coords (x1, y1, x2, y2)
                        width, height = x2 - x1, y2 - y1
                        if class_targets is not None:
                            try:
                                label = cf.bin_id2label[cf.rg_val_to_bin_id(batch['patient_regression_targets'][s_ix][j])]
                            except AttributeError:
                                label = cf.class_id2label[class_targets[s_ix][j]]
                            legend_items.add(label)
                            if show_gt_labels and cf.plot_class_ids:
                                bbox = mpatches.Rectangle((x1, y1), width, height, linewidth=0.6, edgecolor=label.color,
                                                          facecolor='none')
                                if height<=text_fs*6:
                                    y1 -= text_fs*1.5
                                    y2 += text_fs*2
                                text_poss, p = [(x1, y1), (x1, (y1+y2)//2)], 0
                                if roi_items is not None:
                                    for name in roi_items:
                                        if name in cf_roi_items and batch[name][s_ix][j] is not None:
                                            if 'class_targets' in name:
                                                text_str = '{}'.format(class_targets[s_ix][j])
                                                text_x, text_y = (x2 + 0 * len(text_str) // 4, y2)
                                            elif 'regression_targets' in name:
                                                text_str = 'agg. MS: {:.2f}'.format(batch[name][s_ix][j][0])
                                                text_x, text_y = (x2 + 0 * len(text_str) // 4, y2)
                                            elif 'rg_bin_targets_sa' in name:
                                                text_str = 'sa. MS: {}'.format(batch[name][s_ix][j])
                                                text_x, text_y = (x2-0*len(text_str)*text_fs//4, y1)
                                            # elif 'rg_bin_targets' in name:
                                            #     text_str = 'agg. ms:{}'.format(batch[name][s_ix][j])
                                            #     text_x, text_y = (x2+0*len(text_str)//4, y1)


                                            ax.text(text_x, text_y, text_str, color=cf.black if
                                            (label.color[:3]==cf.yellow or label.color[:3]==cf.green) else cf.white,
                                                    fontsize=text_fs,
                                                    bbox=dict(facecolor=label.color, alpha=0.7, edgecolor='none', clip_on=True, pad=0))
                                            p+=1
                                ax.add_patch(bbox)
            if seg is not None and seg.shape[1] == 1:
                #cmap = {1: cf.orange}
                # cmap = {label_id: label.color for label_id, label in cf.bin_id2label.items()}
                # this whole function is totally only hacked together for a quick very specific case
                if seg_cmap == "rg" or seg_cmap=="regression":
                    cmap = {1: cf.bin_id2label[cf.rg_val_to_bin_id(batch['patient_regression_targets'][s_ix][0])].color}
                else:
                    cmap = cf.class_cmap
                ax.imshow(to_rgba(seg[s_ix][0][...,z_ix], cmap), alpha=0.8)


            # # -----evtly visualise predictions -------------
            if pr_boxes is not None or seg_preds is not None:
                if axes is not None and 'pred' in axes.keys():
                    ax = axes['pred']
                else:
                    ax = fig.add_subplot(grid[row, col])
                ax.imshow(bg_img, cmap="gray")
                ax.axis("off")
                col += 1
                if row == 0:
                    ax.set_title("Prediction", fontsize=title_fs)

            # ---------- pred boxes  -------------------------
            if pr_boxes is not None and len(pr_boxes[s_ix]) > 0:
                alpha = 0.7
                box_score_thres = cf.min_det_thresh if box_score_thres is None else box_score_thres
                for j, box in enumerate(pr_boxes[s_ix]):
                    plot_box = box["box_type"] in ["det", "prop"]  # , "pos_anchor", "neg_anchor"]
                    if box["box_type"] == "det" and (float(box["box_score"]) <= box_score_thres or box["box_pred_class_id"] == 0):
                        plot_box = False

                    if plot_box:
                        if d > 0:
                            [z1, z2] = box["box_coords"][4:]
                            if not (z1<=z_ix and z_ix<=z2):
                                box = []
                        if len(box) > 0:
                            [y1, x1, y2, x2] = box["box_coords"][:4]

                            width, height = x2 - x1, y2 - y1

                            if box["box_type"] == "det":
                                try:
                                    label = cf.bin_id2label[cf.rg_val_to_bin_id(box['regression'])]
                                except AttributeError:
                                    label = cf.class_id2label[box['box_pred_class_id']]
                                # assert box["rg_bin"] == cf.rg_val_to_bin_id(box['regression']), \
                                #    "box bin: {}, rg-bin {}".format(box["rg_bin"], cf.rg_val_to_bin_id(box['regression']))
                                color = label.color#cf.aubergine
                                edgecolor = color  # label.color
                                text_color = cf.black if (color[:3]==cf.yellow or color[:3]==cf.green) else cf.white
                                legend_items.add(label)
                                bbox = mpatches.Rectangle((x1, y1), width, height, linewidth=0.6, edgecolor=edgecolor,
                                                          facecolor='none')
                                if height<=text_fs*6:
                                    y1 -= text_fs*1.5
                                    y2 += text_fs*2
                                text_x, text_y = x2, y1
                                #id_text = str(box["box_pred_class_id"]) + "|" if cf.plot_class_ids else ""
                                id_text = "FG: "
                                text_str = r'{}{:.0f}%'.format(id_text, box["box_score"] * 100)
                                text_settings = dict(facecolor=color, alpha=alpha, edgecolor='none', clip_on=True,
                                                     pad=0.2)
                                ax.text(text_x, text_y, text_str, color=text_color,
                                        bbox=text_settings, fontsize=text_fs )

                                if 'regression' in box.keys():
                                    text_x, text_y = x2, y2
                                    id_text = "MS: "+" ".join(["{:.2f}".format(x) for x in box["regression"]])+""
                                    text_str = '{}'.format(id_text)
                                    text_settings = dict(facecolor=color, alpha=alpha, edgecolor='none',
                                                         clip_on=True, pad=0.2)
                                    ax.text(text_x, text_y, text_str, color=text_color,
                                            bbox=text_settings, fontsize=text_fs)
                                if 'rg_bin' in box.keys():
                                    text_x, text_y = x1, y2
                                    text_str = '{}'.format(box["rg_bin"])
                                    text_settings = dict(facecolor=color, alpha=alpha, edgecolor='none',
                                                         clip_on=True, pad=0)
                                    # ax.text(text_x, text_y, text_str, color=cf.white,
                                    #         bbox=text_settings, fontsize=title_fs // 4)
                                if 'box_pred_class_id' in box.keys() and show_cl_ids:
                                    text_x, text_y = x2, y2
                                    id_text = box["box_pred_class_id"]
                                    text_str = '{}'.format(id_text)
                                    text_settings = dict(facecolor=color, alpha=alpha, edgecolor='none',
                                                         clip_on=True, pad=0.2)
                                    ax.text(text_x, text_y, text_str, color=text_color,
                                            bbox=text_settings, fontsize=text_fs)
                            else:
                                label = cf.box_type2label[box["box_type"]]
                                legend_items.add(label)
                                edgecolor = label.color

                            ax.add_patch(bbox)
            row += 1

    # -----actions for all batch entries----------
    if legend and len(legend_items) > 0:
        patches = []
        for label in legend_items:
            if cf.plot_class_ids and type(label) != type(cf.box_labels[0]):
                id_text = str(label.id) + ":"
            else:
                id_text = ""

            patches.append(mpatches.Patch(color=label.color, label="{}{:.10s}".format(id_text, label.name)))
        # assumes one image gives enough y-space for 5 legend items
        ncols = max(1, len(legend_items) // (5 * n_samples))
        plt.figlegend(handles=patches, loc="upper center", bbox_to_anchor=(0.99, 0.86),
                      borderaxespad=0., ncol=ncols, bbox_transform=fig.transFigure,
                      fontsize=int(2/3*title_fs))
        # fig.set_size_inches(mods+3+ncols-1,1.5+1.2*n_samples)

    if out_file is not None:
        plt.savefig(out_file, dpi=dpi, pad_inches=0.0, bbox_inches='tight', tight_layout=True)
    if return_fig:
        return plt.gcf()
    if not (server_env or cf.server_env):
        plt.show()
    plt.clf()
    plt.close()


def view_slices(cf, img, seg=None, ids=None, title="", out_dir=None, legend=True,
                cmap=None, label_remap=None, instance_labels=False):
    """View slices of a 3D image overlayed with corresponding segmentations.
    
    :params img, seg: expected as 3D-arrays
    """
    if isinstance(img, sitk.SimpleITK.Image):
        img = sitk.GetArrayViewFromImage(img)
    elif isinstance(img, np.ndarray):
        #assume channels dim is smallest and in either first or last place
        if np.argmin(img.shape)==2: 
            img = np.moveaxis(img, 2,0) 
    else:
        raise Exception("view_slices got unexpected img type.")

    if seg is not None:
        if isinstance(seg, sitk.SimpleITK.Image):
            seg = sitk.GetArrayViewFromImage(seg)
        elif isinstance(img, np.ndarray):
            if np.argmin(seg.shape)==2: 
                seg = np.moveaxis(seg, 2,0)
        else:
            raise Exception("view_slices got unexpected seg type.")
       
    if label_remap is not None:
        for (key, val) in label_remap.items():
            seg[seg==key] = val

    if instance_labels:
        class Label():
            def __init__(self, id, name, color):
                self.id = id
                self.name = name
                self.color = color
        
        legend_items = {Label(seg_id, "instance_{}".format(seg_id), 
                              cf.color_palette[seg_id%len(cf.color_palette)]) for
                              seg_id in np.unique(seg)}
        if cmap is None:
            cmap = {label.id : label.color for label in legend_items}            
    else:
        legend_items = {cf.seg_id2label[seg_id] for seg_id in np.unique(seg)}
        if cmap is None:
            cmap = {label.id : label.color for label in legend_items}
        
    
    slices = img.shape[0]
    if seg is not None:
        assert slices==seg.shape[0], "Img and seg have different amt of slices."
    grid = gridspec.GridSpec(int(np.ceil(slices/4)),4)
    fig = plt.figure(figsize=(10, slices/4*2.5))
    rng = np.arange(slices, dtype='uint8')
    if not ids is None:
        rng = rng[ids]
    for s in rng:
        ax = fig.add_subplot(grid[int(s/4),int(s%4)])
        ax.imshow(img[s], cmap="gray")
        if not seg is None:
            ax.imshow(to_rgba(seg[s], cmap), alpha=0.9)
            if legend and int(s/4)==0 and int(s%4)==3:
                patches = [mpatches.Patch(color=label.color,
                                           label="{}".format(label.name)) for label in legend_items]
                ncols = 1
                plt.legend(handles=patches,bbox_to_anchor=(1.05, 1), loc=2,
                           borderaxespad=0., ncol=ncols) 
        plt.title("slice {}, {}".format(s, img[s].shape))
        plt.axis('off')

    plt.suptitle(title)
    if out_dir is not None:
        plt.savefig(out_dir, dpi=300, pad_inches=0.0, bbox_inches='tight')
    if not cf.server_env:
        plt.show()
    plt.close()


def plot_txt(cf, txts, labels=None, title="", x_label="", y_labels=["",""], y_ranges=(None,None),
             twin_axes=(), smooth=None, out_dir=None):
    """Read and plot txt data, either from file (txts is paths) or directly (txts is arrays).
    
    :param twin_axes: plot two y-axis over same x-axis. twin_axes expected as
        tuple defining which txt files (determined via indices) share the second y-axis.
    """
    if isinstance(txts, str) or not hasattr(txts, '__iter__'):
        txts = [txts]

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    if len(twin_axes)>0:
        ax2 = ax1.twinx()
    for i, txt in enumerate(txts):
        if isinstance(txt, str):
            arr = np.genfromtxt(txt, delimiter=',',skip_header=1, usecols=(1,2))
        else:
            arr = txt
        if i in twin_axes:
            ax = ax2
        else:
            ax = ax1
        if smooth is not None:
            spline_graph = interpol.UnivariateSpline(arr[:,0], arr[:,1], k=5, s=float(smooth))
            ax.plot(arr[:, 0], spline_graph(arr[:,0]), color=cf.color_palette[i % len(cf.color_palette)],
                    marker='', markersize=2, linestyle='solid')
        ax.plot(arr[:,0], arr[:,1], color=cf.color_palette[i%len(cf.color_palette)],
                 marker='', markersize=2, linestyle='solid', label=labels[i], alpha=0.5 if smooth else 1.)
    plt.title(title)
    
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_labels[0])
    if y_ranges[0] is not None:
        ax1.set_ylim(y_ranges[0])
    if len(twin_axes)>0:
        ax2.set_ylabel(y_labels[1])
        if y_ranges[1] is not None:
            ax2.set_ylim(y_ranges[1])
    
    plt.grid()
    
    if labels is not None:
        ax1.legend(loc="upper center")
        if len(twin_axes)>0:
            ax2.legend(loc=4)
        
    if out_dir is not None:
        plt.savefig(out_dir, dpi=200)
    return fig

def plot_tboard_logs(cf, log_dir, tag_filters=[""], inclusive_filters=True, out_dir=None, x_label="",
                     y_labels=["",""], y_ranges=(None,None), twin_axes=(), smooth=None):
    """Plot (only) tboard scalar logs from given log_dir for multiple runs sorted by tag.
    """    
    print("log dir", log_dir)
    mpl = EventMultiplexer().AddRunsFromDirectory(log_dir) #EventAccumulator(log_dir)
    mpl.Reload()
    
    # Print tags of contained entities, use these names to retrieve entities as below
    #print(mpl.Runs())
    scalars = {runName : data['scalars'] for (runName, data) in mpl.Runs().items() if len(data['scalars'])>0}
    print("scalars", scalars)
    tags = {}
    tag_filters = [tag_filter.lower() for tag_filter in tag_filters]
    for (runName, runtags) in scalars.items():
        print("rn", runName.lower())
        check = np.any if inclusive_filters else np.all
        if np.any([tag_filter in runName.lower() for tag_filter in tag_filters]):
            for runtag in runtags:
                #if tag_filter in runtag.lower():
                if runtag not in tags:
                    tags[runtag] = [runName]
                else:
                    tags[runtag].append(runName)
    print("tags ", tags)
    for (tag, runNames) in tags.items():
        print("runnames ", runNames)
        print("tag", tag)
        tag_scalars = []
        labels = []
        for run in runNames:
            #mpl.Scalars returns ScalarEvents array holding wall_time, step, value per time step (shape series_length x 3)
            #print(mpl.Scalars(runName, tag)[0])
            run_scalars = [(s.step, s.value) for s in mpl.Scalars(run, tag)]
            print(np.array(run_scalars).shape)
            tag_scalars.append(np.array(run_scalars))
            print("run", run)
            labels.append("/".join(run.split("/")[-2:]))
        #print("tag scalars ", tag_scalars)
        if out_dir is not None:
            out_path = os.path.join(out_dir,tag.replace("/","_"))
        else:
            out_path = None
        plot_txt(txts=tag_scalars, labels=labels, title=tag, out_dir=out_path, cf=cf,
                 x_label=x_label, y_labels=y_labels, y_ranges=y_ranges, twin_axes=twin_axes, smooth=smooth)


def plot_box_legend(cf, box_coords=None, class_id=None, out_dir=None):
    """plot a blank box explaining box annotations.
    :param cf:
    :return:
    """
    if class_id is None:
        class_id = 1

    img = np.ones(cf.patch_size[:2])
    dim_max = max(cf.patch_size[:2])
    width, height = cf.patch_size[0] // 2, cf.patch_size[1] // 2
    if box_coords is None:
        # lower left corner
        x1, y1 = width // 2, height // 2
        x2, y2 = x1 + width, y1 + height
    else:
        y1, x1, y2, x2 = box_coords

    fig = plt.figure(tight_layout=True, dpi=300)
    ax = fig.add_subplot(111)
    title_fs = 36
    label = cf.class_id2label[class_id]
    # legend_items.add(label)
    ax.set_facecolor(cf.beige)
    ax.imshow(img, cmap='gray', vmin=0., vmax=1., alpha=0)
    # ax.axis('off')
    # suppress_axes_lines(ax)
    ax.set_xticks([])
    ax.set_yticks([])

    text_x, text_y = x2 * 0.85, y1
    id_text = "class id" + " | " if cf.plot_class_ids else ""
    text_str = '{}{}'.format(id_text, "confidence")
    text_settings = dict(facecolor=label.color, alpha=0.5, edgecolor='none', clip_on=True,
                         pad=0)
    ax.text(text_x, text_y, text_str, color=cf.white,
            bbox=text_settings, fontsize=title_fs // 4)
    edgecolor = label.color
    if any(['regression' in task for task in cf.prediction_tasks]):
        text_x, text_y = x2 * 0.85, y2
        id_text = "regression"
        if any(['ken_gal' in task or 'feindt' in task for task in cf.prediction_tasks]):
            id_text += " | uncertainty"
        text_str = '{}'.format(id_text)
        ax.text(text_x, text_y, text_str, color=cf.white, bbox=text_settings, fontsize=title_fs // 4)
        if 'regression_bin' in cf.prediction_tasks or hasattr(cf, "rg_val_to_bin_id"):
            text_x, text_y = x1, y2
            text_str = 'Rg. Bin'
            ax.text(text_x, text_y, text_str, color=cf.white, bbox=text_settings, fontsize=title_fs // 4)

    if 'lesion_gleasons' in cf.observables_rois:
        text_x, text_y = x1, y1
        text_str = 'Gleason Score'
        ax.text(text_x, text_y, text_str, color=cf.white, bbox=text_settings, fontsize=title_fs // 4)

    bbox = mpatches.Rectangle((x1, y1), width, height, linewidth=1., edgecolor=edgecolor, facecolor='none')
    ax.add_patch(bbox)
    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, "box_legend.png"))

def plot_boxes(cf, box_coords, patch_size=None, scores=None, class_ids=None, out_file=None, ax=None):

    if patch_size is None:
        patch_size = cf.patch_size[:2]
    if class_ids is None:
        class_ids = np.ones((len(box_coords),), dtype='uint8')
    if scores is None:
        scores = np.ones((len(box_coords),), dtype='uint8')

    img = np.ones(patch_size)

    y1, x1, y2, x2 = box_coords[:,0], box_coords[:,1], box_coords[:,2], box_coords[:,3]
    width, height = x2-x1, y2-y1

    close = False
    if ax is None:
        fig = plt.figure(tight_layout=True, dpi=300)
        ax = fig.add_subplot(111)
        close = True
    title_fs = 56

    ax.set_facecolor((*cf.gray,0.15))
    ax.imshow(img, cmap='gray', vmin=0., vmax=1., alpha=0)
    #ax.axis('off')
    #suppress_axes_lines(ax)
    ax.set_xticks([])
    ax.set_yticks([])

    for bix, cl_id in enumerate(class_ids):
        label = cf.class_id2label[cl_id]
        text_x, text_y = x2[bix] -20, y1[bix] +5
        id_text = class_ids[bix] if cf.plot_class_ids else ""
        text_str = '{}{}{:.0f}'.format(id_text, " | ", scores[bix] * 100)
        text_settings = dict(facecolor=label.color, alpha=0.5, edgecolor='none', clip_on=True, pad=0)
        ax.text(text_x, text_y, text_str, color=cf.white, bbox=text_settings, fontsize=title_fs // 4)
        edgecolor = label.color

        bbox = mpatches.Rectangle((x1[bix], y1[bix]), width[bix], height[bix], linewidth=1., edgecolor=edgecolor, facecolor='none')
        ax.add_patch(bbox)

    if out_file is not None:
        plt.savefig(out_file)
    if close:
        plt.close()



if __name__=="__main__":
    cluster_exp_root = "/mnt/E132-Cluster-Projects"
    #dataset="prostate/"
    dataset = "lidc/"
    exp_name = "ms13_mrcnnal3d_rg_bs8_480k"
    #exp_dir = os.path.join("datasets", dataset, "experiments", exp_name)
    # exp_dir = os.path.join(cluster_exp_root, dataset, "experiments", exp_name)
    # log_dir = os.path.join(exp_dir, "logs")
    # sys.path.append(exp_dir)
    # from configs import Configs
    # cf = configs()
    #
    # #print("logdir", log_dir)
    # #out_dir = os.path.join(cf.source_dir, log_dir.replace("/", "_"))
    # #print("outdir", out_dir)
    # log_dir = os.path.join(cf.source_dir, log_dir)
    # plot_tboard_logs(cf, log_dir, tag_filters=["train/lesion_avp", "val/lesion_ap", "val/lesion_avp", "val/patient_lesion_avp"], smooth=2.2, out_dir=log_dir, # y_ranges=([0,900], [0,0.8]),
    #                 twin_axes=[1], y_labels=["counts",""], x_label="epoch")

    #plot_box_legend(cf, out_dir=exp_dir)

    
    
