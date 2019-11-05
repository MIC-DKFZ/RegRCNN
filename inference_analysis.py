"""for presentations etc"""

import plotting as plg

import sys
import os
import pickle

import numpy as np
import pandas as pd
import torch

import utils.exp_utils as utils
import utils.model_utils as mutils
from predictor import Predictor
from evaluator import Evaluator


def find_pid_in_splits(pid, exp_dir=None):
    if exp_dir is None:
        exp_dir = cf.exp_dir
    check_file = os.path.join(exp_dir, 'fold_ids.pickle')
    with open(check_file, 'rb') as handle:
        splits = pickle.load(handle)

    finds = []
    for i, split in enumerate(splits):
        if pid in split:
            finds.append(i)
            print("Pid {} found in split {}".format(pid, i))
    if not len(finds)==1:
        raise Exception("pid {} found in more than one split: {}".format(pid, finds))
    return finds[0]

def plot_train_forward(slices=None):
    with torch.no_grad():
        batch = next(val_gen)
        results_dict = net.train_forward(batch, is_validation=True) #seg preds are int preds already

        out_file = os.path.join(anal_dir, "straight_val_inference_fold_{}".format(str(cf.fold)))
        plg.view_batch(cf, batch, res_dict=results_dict, show_info=False, legend=True,
                                  out_file=out_file, slices=slices)

def plot_forward(pid, slices=None):
    with torch.no_grad():
        batch = batch_gen['test'].generate_train_batch(pid=pid)
        results_dict = net.test_forward(batch) #seg preds are only seg_logits! need to take argmax.

        if 'seg_preds' in results_dict.keys():
            results_dict['seg_preds'] = np.argmax(results_dict['seg_preds'], axis=1)[:,np.newaxis]

        out_file = os.path.join(anal_dir, "straight_inference_fold_{}_pid_{}".format(str(cf.fold), pid))
        plg.view_batch(cf, batch, res_dict=results_dict, show_info=False, legend=True, show_gt_labels=True,
                                  out_file=out_file, sample_picks=slices)


def plot_merged_boxes(results_list, pid, plot_mods=False, show_seg_ids="all", show_info=True, show_gt_boxes=True,
                      s_picks=None, vol_slice_picks=None, score_thres=None):
    """

    :param results_list: holds (results_dict, pid)
    :param pid:
    :return:
    """
    results_dict = [res_dict for (res_dict, pid_) in results_list if pid_==pid][0]
    #seg preds are discarded in predictor pipeline.
    #del results_dict['seg_preds']

    batch = batch_gen['test'].generate_train_batch(pid=pid)
    out_file = os.path.join(anal_dir, "merged_boxes_fold_{}_pid_{}_thres_{}.png".format(str(cf.fold), pid, str(score_thres).replace(".","_")))

    utils.save_obj({'res_dict':results_dict, 'batch':batch}, os.path.join(anal_dir, "bytes_merged_boxes_fold_{}_pid_{}".format(str(cf.fold), pid)))

    plg.view_batch(cf, batch, res_dict=results_dict, show_info=show_info, legend=False, sample_picks=s_picks,
                   show_seg_pred=True, show_seg_ids=show_seg_ids, show_gt_boxes=show_gt_boxes,
                   box_score_thres=score_thres, vol_slice_picks=vol_slice_picks, show_gt_labels=True,
                   plot_mods=plot_mods, out_file=out_file, has_colorchannels=cf.has_colorchannels, dpi=600)

    return





if __name__=="__main__":
    class Args():
        def __init__(self):
            #self.dataset_name = "datasets/prostate"
            self.dataset_name = "datasets/lidc"
            #self.exp_dir = "datasets/toy/experiments/mrcnnal2d_clkengal"  # detunet2d_di_bs16_ps512"
            #self.exp_dir = "/home/gregor/networkdrives/E132-Cluster-Projects/prostate/experiments/gs6071_retinau3d_cl_bs6"
            #self.exp_dir = "/home/gregor/networkdrives/E132-Cluster-Projects/prostate/experiments/gs6071_frcnn3d_cl_bs6"
            #self.exp_dir = "/home/gregor/networkdrives/E132-Cluster-Projects/prostate/experiments_t2/gs6071_mrcnn3d_cl_bs6_lessaug"
            #self.exp_dir = "/home/gregor/networkdrives/E132-Cluster-Projects/prostate/experiments/gs6071_detfpn3d_cl_bs6"
            #self.exp_dir = "/home/gregor/networkdrives/E132-Cluster-Projects/lidc_sa/experiments/ms12345_mrcnn3d_rgbin_bs8"
            self.exp_dir = '/home/gregor/Documents/medicaldetectiontoolkit/datasets/lidc/experiments/ms12345_mrcnn3d_rg_bs8'
            #self.exp_dir = '/home/gregor/Documents/medicaldetectiontoolkit/datasets/lidc/experiments/ms12345_mrcnn3d_rgbin_bs8'

            self.server_env = False
    args = Args()


    data_loader = utils.import_module('dl', os.path.join(args.dataset_name, "data_loader.py"))

    config_file = utils.import_module('cf', os.path.join(args.exp_dir, "configs.py"))
    cf = config_file.Configs()
    cf.exp_dir = args.exp_dir
    cf.test_dir = cf.exp_dir

    pid = '0811a'
    cf.fold = find_pid_in_splits(pid)
    #cf.fold = 0
    cf.merge_2D_to_3D_preds = False
    if cf.merge_2D_to_3D_preds:
        cf.dim==3
    cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(cf.fold))
    anal_dir = os.path.join(cf.exp_dir, "inference_analysis")

    logger = utils.get_logger(cf.exp_dir)
    model = utils.import_module('model', os.path.join(cf.exp_dir, "model.py"))
    torch.backends.cudnn.benchmark = cf.dim == 3
    net = model.net(cf, logger).cuda()
    test_predictor = Predictor(cf, None, logger, mode='test')
    test_evaluator = Evaluator(cf, logger, mode='test')
    #val_gen = data_loader.get_train_generators(cf, logger, data_statistics=False)['val_sampling']
    batch_gen = data_loader.get_test_generator(cf, logger)
    weight_paths = [os.path.join(cf.fold_dir, '{}_best_params.pth'.format(rank)) for rank in
                    test_predictor.epoch_ranking]
    try:
        pids = batch_gen["test"].dataset_pids
    except:
        pids = batch_gen["test"].generator.dataset_pids
    print("pids in test set: ",  pids)
    #pid = pids[0]
    #assert pid in pids

    # load already trained model weights
    rank = 0
    weight_path = weight_paths[rank]
    with torch.no_grad():
        pass
        net.load_state_dict(torch.load(weight_path))
        net.eval()
    # generate a batch from test set and show results
    if not os.path.isdir(anal_dir):
        os.mkdir(anal_dir)

    #plot_train_forward()
    #plot_forward(pids[0])
    #net.actual_dims()
    #batch_gen = data_loader.get_test_generator(cf, logger)
    merged_boxes_file = os.path.join(cf.fold_dir, "merged_box_results")
    try:
        results_list = utils.load_obj(merged_boxes_file+".pkl")
        print("loaded merged boxes from file.")
    except FileNotFoundError:
        results_list = test_predictor.load_saved_predictions()
        utils.save_obj(results_list, merged_boxes_file)


    cf.plot_class_ids = False
    for pid in [pid,]:#['0317a',]:#pids[2:8]:
        assert pid in [res[1] for res in results_list]
        plot_merged_boxes(results_list, pid=pid, show_info=True, show_gt_boxes=True, show_seg_ids="all", score_thres=0.13,
                          s_picks=None, vol_slice_picks=None, plot_mods=False)








