"""
Created at 04/02/19 13:50
@author: gregor 
"""
import plotting as plg

import sys
import os
import pickle
import json, socket, subprocess, time, threading

import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
from matplotlib.lines import  Line2D

import utils.exp_utils as utils
import utils.model_utils as mutils
from predictor import Predictor
from evaluator import Evaluator


"""
Need to start this script as sudo for background logging thread to work (needs to set niceness<0)
"""


def measure_train_batch_loading(logger, batch_gen, iters=1, warm_up=20, is_val=False, out_dir=None):
    torch.cuda.empty_cache()
    timer_key = "val_fw" if is_val else "train_fw"
    for i in range(warm_up):
        batch = next(batch_gen)
        print("\rloaded warm-up batch {}/{}".format(i+1, warm_up), end="", flush=True)
    sysmetrics_start_ix = len(logger.sysmetrics.index)
    for i in range(iters):
        logger.time(timer_key)
        batch = next(batch_gen)
        print("\r{} batch {} loading took {:.3f}s.".format("val" if is_val else "train", i+1,
                                                           logger.time(timer_key)), end="", flush=True)
    print("Total avg fw {:.2f}s".format(logger.get_time(timer_key)/iters))
    if out_dir is not None:
        assert len(logger.sysmetrics[sysmetrics_start_ix:-1]) > 0, "train loading: empty df"
        logger.sysmetrics[sysmetrics_start_ix:-1].to_pickle(os.path.join(
            out_dir,"{}_loading.pickle".format("val" if is_val else "train")))
    return logger.sysmetrics[sysmetrics_start_ix:-1]


def measure_RPN(logger, net, batch, iters=1, warm_up=20, out_dir=None):
    torch.cuda.empty_cache()
    data = torch.from_numpy(batch["data"]).float().cuda()
    fpn_outs = net.fpn(data)
    rpn_feature_maps = [fpn_outs[i] for i in net.cf.pyramid_levels]

    for i in range(warm_up):
        layer_outputs = [net.rpn(p_feats) for p_feats in rpn_feature_maps]
        print("\rfinished warm-up batch {}/{}".format(i+1, warm_up), end="", flush=True)
    sysmetrics_start_ix = len(logger.sysmetrics.index)
    for i in range(iters):
        logger.time("RPN_fw")
        layer_outputs = [net.rpn(p_feats) for p_feats in rpn_feature_maps]
        print("\r{} batch took {:.3f}s.".format("RPN", logger.time("RPN_fw")), end="", flush=True)
    print("Total avg fw {:.2f}s".format(logger.get_time("RPN_fw")/iters))

    if out_dir is not None:
        assert len(logger.sysmetrics[sysmetrics_start_ix:-1])>0, "six {}, sysm ix {}".format(sysmetrics_start_ix, logger.sysmetrics.index)
        logger.sysmetrics[sysmetrics_start_ix:-1].to_pickle(os.path.join(out_dir,"RPN_msrmts.pickle"))
    return logger.sysmetrics[sysmetrics_start_ix:-1]

def measure_FPN(logger, net, batch, iters=1, warm_up=20, out_dir=None):
    torch.cuda.empty_cache()
    data = torch.from_numpy(batch["data"]).float().cuda()
    for i in range(warm_up):
        outputs = net.fpn(data)
        print("\rfinished warm-up batch {}/{}".format(i+1, warm_up), end="", flush=True)
    sysmetrics_start_ix = len(logger.sysmetrics.index)
    for i in range(iters):
        logger.time("FPN_fw")
        outputs = net.fpn(data)
        #print("in mean thread", logger.sysmetrics.index)
        print("\r{} batch took {:.3f}s.".format("FPN", logger.time("FPN_fw")), end="", flush=True)
    print("Total avg fw {:.2f}s".format(logger.get_time("FPN_fw")/iters))

    if out_dir is not None:
        assert len(logger.sysmetrics[sysmetrics_start_ix:-1])>0, "six {}, sysm ix {}".format(sysmetrics_start_ix, logger.sysmetrics.index)
        logger.sysmetrics[sysmetrics_start_ix:-1].to_pickle(os.path.join(out_dir,"FPN_msrmts.pickle"))
    return logger.sysmetrics[sysmetrics_start_ix:-1]

def measure_forward(logger, net, batch, iters=1, warm_up=20, out_dir=None):
    torch.cuda.empty_cache()
    data = torch.from_numpy(batch["data"]).float().cuda()
    for i in range(warm_up):
        outputs = net.forward(data)
        print("\rfinished warm-up batch {}/{}".format(i+1, warm_up), end="", flush=True)
    sysmetrics_start_ix = len(logger.sysmetrics.index)
    for i in range(iters):
        logger.time("net_fw")
        outputs = net.forward(data)
        print("\r{} batch took {:.3f}s.".format("forward", logger.time("net_fw")), end="", flush=True)
    print("Total avg fw {:.2f}s".format(logger.get_time("net_fw")/iters))
    if out_dir is not None:
        assert len(logger.sysmetrics[sysmetrics_start_ix:-1]) > 0, "fw: empty df"
        logger.sysmetrics[sysmetrics_start_ix:-1].to_pickle(os.path.join(out_dir,"fw_msrmts.pickle"))
    return logger.sysmetrics[sysmetrics_start_ix:-1].copy()

def measure_train_forward(logger, net, batch, iters=1, warm_up=20, is_val=False, out_dir=None):
    torch.cuda.empty_cache()
    timer_key = "val_fw" if is_val else "train_fw"
    optimizer = torch.optim.Adam(net.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay)
    for i in range(warm_up):
        results_dict = net.train_forward(batch)
        print("\rfinished warm-up batch {}/{}".format(i+1, warm_up), end="", flush=True)
    sysmetrics_start_ix = len(logger.sysmetrics.index)
    for i in range(iters):
        logger.time(timer_key)
        if not is_val:
            optimizer.zero_grad()
        results_dict = net.train_forward(batch, is_validation=is_val)
        #results_dict["torch_loss"] *= torch.rand(1).cuda()
        if not is_val:
            results_dict["torch_loss"].backward()
            optimizer.step()
        print("\r{} batch took {:.3f}s.".format("val" if is_val else "train", logger.time(timer_key)), end="", flush=True)
    print("Total avg fw {:.2f}s".format(logger.get_time(timer_key)/iters))
    if out_dir is not None:
        assert len(logger.sysmetrics[sysmetrics_start_ix:-1]) > 0, "train_fw: empty df"
        logger.sysmetrics[sysmetrics_start_ix:-1].to_pickle(os.path.join(
            out_dir,"{}_msrmts.pickle".format("val_fw" if is_val else "train_fwbw")))
    return logger.sysmetrics[sysmetrics_start_ix:-1].copy()

def measure_train_fw_incl_batch_gen(logger, net, batch_gen, iters=1, warm_up=20, is_val=False, out_dir=None):
    torch.cuda.empty_cache()
    timer_key = "val_fw" if is_val else "train_fw"
    for i in range(warm_up):
        batch = next(batch_gen)
        results_dict = net.train_forward(batch)
        print("\rfinished warm-up batch {}/{}".format(i+1, warm_up), end="", flush=True)
    sysmetrics_start_ix = len(logger.sysmetrics.index)
    for i in range(iters):
        logger.time(timer_key)
        batch = next(batch_gen)
        results_dict = net.train_forward(batch, is_validation=is_val)
        if not is_val:
            results_dict["torch_loss"].backward()
        print("\r{} batch took {:.3f}s.".format("val" if is_val else "train", logger.time(timer_key)), end="", flush=True)
    print("Total avg fw {:.2f}s".format(logger.get_time(timer_key)/iters))
    if out_dir is not None:
        assert len(logger.sysmetrics[sysmetrics_start_ix:-1]) > 0, "train_fw incl batch: empty df"
        logger.sysmetrics[sysmetrics_start_ix:-1].to_pickle(os.path.join(
            out_dir,"{}_incl_batch_msrmts.pickle".format("val_fw" if is_val else "train_fwbw")))
    return logger.sysmetrics[sysmetrics_start_ix:-1]



def measure_train_backward(cf, logger, net, batch, iters=1, warm_up=20, out_dir=None):
    torch.cuda.empty_cache()
    optimizer = torch.optim.Adam(net.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay)
    results_dict = net.train_forward(batch, is_validation=False)
    loss = results_dict["torch_loss"]
    for i in range(warm_up):
        loss.backward(retain_graph=True)
        print("\rfinished warm-up batch {}/{}".format(i + 1, warm_up), end="", flush=True)
    sysmetrics_start_ix = len(logger.sysmetrics.index)
    for i in range(iters):
        logger.time("train_bw")
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        print("\r{} bw batch {} took {:.3f}s.".format("train", i+1, logger.time("train_bw")), end="", flush=True)
    print("Total avg bw {:.2f}s".format(logger.get_time("train_bw") / iters))
    if out_dir is not None:
        assert len(logger.sysmetrics[sysmetrics_start_ix:-1]) > 0, "train_bw: empty df"
        logger.sysmetrics[sysmetrics_start_ix:-1].to_pickle(os.path.join(out_dir,"train_bw.pickle"))
    return logger.sysmetrics[sysmetrics_start_ix:-1]



def measure_test_forward(logger, net, batch, iters=1, return_masks=False):
    torch.cuda.empty_cache()
    for i in range(iters):
        logger.time("test_fw")
        results_dict = net.test_forward(batch, return_masks=return_masks)
        print("\rtest batch took {:.3f}s.".format(logger.time("test_fw")), end="", flush=True)
    print("Total avg test fw {:.2f}s".format(logger.get_time('test_fw')/iters))


def perform_measurements(args, iters=20):

    cf = utils.prep_exp(args.dataset_name, args.exp_dir, args.server_env, is_training=True, use_stored_settings=False)

    cf.exp_dir = args.exp_dir

    # pid = 1624
    # cf.fold = find_pid_in_splits(pid)
    cf.fold = 0
    cf.merge_2D_to_3D_preds = False
    cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(cf.fold))

    logger = utils.get_logger(cf.exp_dir, sysmetrics_interval=0.5)
    model = utils.import_module('model', cf.model_path)
    net = model.net(cf, logger).cuda()
    test_predictor = Predictor(cf, None, logger, mode='test')
    #cf.p_batchbalance = 0
    #cf.do_aug = False
    batch_gens = data_loader.get_train_generators(cf, logger)
    train_gen, val_gen = batch_gens['train'], batch_gens['val_sampling']
    test_gen = data_loader.get_test_generator(cf, logger)['test']
    weight_paths = [os.path.join(cf.fold_dir, '{}_best_params.pth'.format(rank)) for rank in
                    test_predictor.epoch_ranking]

    try:
        pids = test_gen.dataset_pids
    except:
        pids = test_gen.generator.dataset_pids
    print("pids in test set: ", pids)
    pid = pids[0]
    assert pid in pids
    pid = "285"

    model_name = cf.model

    results_dir = "/home/gregor/Documents/medicaldetectiontoolkit/code_optim/"+model_name
    os.makedirs(results_dir, exist_ok=True)
    print("Model: {}.".format(model_name))
    #gpu_logger = utils.Nvidia_GPU_Logger()
    #gpu_logger.start(interval=0.1)
    #measure_train_batch_loading(logger, train_gen, iters=iters, out_dir=results_dir)
    #measure_train_batch_loading(logger, val_gen, iters=iters, is_val=True, out_dir=results_dir)
    #measure_RPN(logger, net, next(train_gen), iters=iters,  out_dir=results_dir)
    #measure_FPN(logger, net, next(train_gen), iters=iters, out_dir=results_dir)
    #measure_forward(logger, net, next(train_gen), iters=iters, out_dir=results_dir)
    measure_train_forward(logger, net, next(train_gen), iters=iters, out_dir=results_dir) #[['global_step', 'gpu_utilization (%)']]
    #measure_train_forward(logger, net, next(val_gen), iters=iters, is_val=True, out_dir=results_dir)
    #measure_train_fw_incl_batch_gen(logger, net, train_gen, iters=iters, out_dir=results_dir)
    #measure_train_fw_incl_batch_gen(logger, net, val_gen, iters=iters, is_val=True, out_dir=results_dir)
    #measure_train_backward(cf, logger, net, next(train_gen), iters=iters, out_dir=results_dir)
    #measure_test_forward(logger, net, next(test_gen), iters=iters, return_masks=cf.return_masks_in_test)

    return results_dir, iters

def plot_folder(cf, ax, results_dir, iters, markers='o', offset=(+0.01, -4)):
    point_renaming = {"FPN_msrmts": ["FPN.forward", (offset[0], -4)], "fw_msrmts": "net.forward",
                      "train_bw": "backward+optimizer",
                      "train_fw_msrmts": "net.train_forward",
                      "train_fw_incl_batch": "train_fw+batch", "RPN_msrmts": "RPN.forward",
                      "train_fwbw_msrmts": ["train_fw+bw", (offset[0], +2)],
                      "val_fw_msrmts": ["val_fw", (offset[0], -4)],
                      "train_fwbw_incl_batch_msrmts": ["train_fw+bw+batchload", (offset[0], +2)],
                      "train_fwbw_incl_batch_aug_msrmts": ["train_fw+bw+batchload+aug", (-0.2, +2)],
                      "val_fw_incl_batch_msrmts": ["val_fw+batchload", (offset[0], -4)],
                      "val_loading": ["val_load", (-0.06, -4)],
                      "train_loading_wo_bal_fg_aug": ["train_load_w/o_bal,fg,aug", (offset[0], 2)],
                      "train_loading_wo_balancing": ["train_load_w/o_balancing", (-0.05, 2)],
                      "train_loading_wo_aug": ["train_load_w/o_aug", (offset[0], 2)],
                      "train_loading_wo_bal_fg": ["train_load_w/o_bal,fg", (offset[0], -4)],
                      "train_loading": ["train_load", (+0.01, -1.3)]
                      }
    dfs = OrderedDict()
    for file in os.listdir(results_dir):
        if os.path.splitext(file)[-1]==".pickle":
           dfs[file.split(os.sep)[-1].split(".")[0]] = pd.read_pickle(os.path.join(results_dir,file))


    for i, (name, df) in enumerate(dfs.items()):
        time = (df["rel_time"].iloc[-1] - df["rel_time"].iloc[0])/iters
        gpu_u = df["gpu_utilization (%)"].values.astype(int).mean()

        color = cf.color_palette[i%len(cf.color_palette)]
        ax.scatter(time, gpu_u, color=color, marker=markers)
        if name in point_renaming.keys():
            name = point_renaming[name]
            if isinstance(name, list):
                offset = name[1]
                name = name[0]
        ax.text(time+offset[0], gpu_u+offset[1], name, color=color)

def analyze_measurements(cf, results_dir, iters, title=""):
    fig, ax = plg.plt.subplots(1, 1)

    settings = [(results_dir, iters, 'o'), (os.path.join(results_dir, "200iters_pre_optim"), 200, 'v', (-0.08, 2)),
                (os.path.join(results_dir, "200iters_after_optim"), 200, 'o')]
    for args in settings:
        plot_folder(cf, ax, *args)
    labels = ["after optim", "pre optim"]
    handles = [Line2D([0], [0], marker=settings[i][2], label=labels[i], color="w", markerfacecolor=cf.black, markersize=10)
               for i in range(len(settings[:2]))]
    plg.plt.legend(handles=handles, loc="best")
    ax.set_xlim(0,ax.get_xlim()[1]*1.05)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mean GPU Utilization (%)")
    ax.set_xlabel("Runtime (s)")
    plg.plt.title(title+"GPU utilization vs Method Runtime\nMean Over {} Iterations".format(iters))

    major_ticks = np.arange(0, 101, 10)
    minor_ticks = np.arange(0, 101, 5)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)


    plg.plt.savefig(os.path.join(results_dir, "measurements.png"))



    return


if __name__=="__main__":
    class Args():
        def __init__(self):
            self.dataset_name = "datasets/prostate"
            self.exp_dir = "datasets/prostate/experiments/dev"
            self.server_env = False


    args = Args()

    sys.path.append(args.dataset_name)
    import data_loader
    from configs import Configs
    cf = configs(args.server_env)
    iters = 200
    results_dir, iters = perform_measurements(args, iters=iters)
    results_dir = "/home/gregor/Documents/medicaldetectiontoolkit/code_optim/" + cf.model
    analyze_measurements(cf, results_dir, iters=iters, title=cf.model+": ")


