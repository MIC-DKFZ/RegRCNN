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

""" execution script. this where all routines come together and the only script you need to call.
    refer to parse args below to see options for execution.
"""

import plotting as plg

import os
import warnings
import argparse
import time

import torch

import utils.exp_utils as utils
from evaluator import Evaluator
from predictor import Predictor


for msg in ["Attempting to set identical bottom==top results",
            "This figure includes Axes that are not compatible with tight_layout",
            "Data has no positive values, and therefore cannot be log-scaled.",
            ".*invalid value encountered in true_divide.*"]:
    warnings.filterwarnings("ignore", msg)


def train(cf, logger):
    """
    performs the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs. logs to file and tensorboard.
    """
    logger.info('performing training in {}D over fold {} on experiment {} with model {}'.format(
        cf.dim, cf.fold, cf.exp_dir, cf.model))
    logger.time("train_val")

    # -------------- inits and settings -----------------
    net = model.net(cf, logger).cuda()
    if cf.optimizer == "ADAM":
        optimizer = torch.optim.Adam(net.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay)
    elif cf.optimizer == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay, momentum=0.3)
    if cf.dynamic_lr_scheduling:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=cf.scheduling_mode, factor=cf.lr_decay_factor,
                                                                    patience=cf.scheduling_patience)
    model_selector = utils.ModelSelector(cf, logger)

    starting_epoch = 1
    if cf.resume_from_checkpoint:
        starting_epoch = utils.load_checkpoint(cf.resume_from_checkpoint, net, optimizer)
        logger.info('resumed from checkpoint {} at epoch {}'.format(cf.resume_from_checkpoint, starting_epoch))

    # prepare monitoring
    monitor_metrics = utils.prepare_monitoring(cf)

    logger.info('loading dataset and initializing batch generators...')
    batch_gen = data_loader.get_train_generators(cf, logger)

    # -------------- training -----------------
    for epoch in range(starting_epoch, cf.num_epochs + 1):

        logger.info('starting training epoch {}/{}'.format(epoch, cf.num_epochs))
        logger.time("train_epoch")

        net.train()

        train_results_list = []
        train_evaluator = Evaluator(cf, logger, mode='train')

        for i in range(cf.num_train_batches):
            logger.time("train_batch_loadfw")
            batch = next(batch_gen['train'])
            batch_gen['train'].generator.stats['roi_counts'] += batch['roi_counts']
            batch_gen['train'].generator.stats['empty_samples_count'] += batch['empty_samples_count']

            logger.time("train_batch_loadfw")
            logger.time("train_batch_netfw")
            results_dict = net.train_forward(batch)
            logger.time("train_batch_netfw")
            logger.time("train_batch_bw")
            optimizer.zero_grad()
            results_dict['torch_loss'].backward()
            if cf.clip_norm:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cf.clip_norm, norm_type=2) # gradient clipping
            optimizer.step()
            train_results_list.append(({k:v for k,v in results_dict.items() if k != "seg_preds"}, batch["pid"])) # slim res dict
            if not cf.server_env:
                print("\rFinished training batch " +
                      "{}/{} in {:.1f}s ({:.2f}/{:.2f} forw load/net, {:.2f} backw).".format(i+1, cf.num_train_batches,
                                                                                             logger.get_time("train_batch_loadfw")+
                                                                                             logger.get_time("train_batch_netfw")
                                                                                             +logger.time("train_batch_bw"),
                                                                                             logger.get_time("train_batch_loadfw",reset=True),
                                                                                             logger.get_time("train_batch_netfw", reset=True),
                                                                                             logger.get_time("train_batch_bw", reset=True)), end="", flush=True)
        print()

        #--------------- train eval ----------------
        if (epoch-1)%cf.plot_frequency==0:
            # view an example batch
            logger.time("train_plot")
            plg.view_batch(cf, batch, results_dict, has_colorchannels=cf.has_colorchannels, show_gt_labels=True,
                           out_file=os.path.join(cf.plot_dir, 'batch_example_train_{}.png'.format(cf.fold)))
            logger.info("generated train-example plot in {:.2f}s".format(logger.time("train_plot")))


        logger.time("evals")
        _, monitor_metrics['train'] = train_evaluator.evaluate_predictions(train_results_list, monitor_metrics['train'])
        logger.time("evals")
        logger.time("train_epoch", toggle=False)
        del train_results_list

        #----------- validation ------------
        logger.info('starting validation in mode {}.'.format(cf.val_mode))
        logger.time("val_epoch")
        with torch.no_grad():
            net.eval()
            val_results_list = []
            val_evaluator = Evaluator(cf, logger, mode=cf.val_mode)
            val_predictor = Predictor(cf, net, logger, mode='val')

            for i in range(batch_gen['n_val']):
                logger.time("val_batch")
                batch = next(batch_gen[cf.val_mode])
                if cf.val_mode == 'val_patient':
                    results_dict = val_predictor.predict_patient(batch)
                elif cf.val_mode == 'val_sampling':
                    results_dict = net.train_forward(batch, is_validation=True)
                val_results_list.append([results_dict, batch["pid"]])
                if not cf.server_env:
                    print("\rFinished validation {} {}/{} in {:.1f}s.".format('patient' if cf.val_mode=='val_patient' else 'batch',
                                                                              i + 1, batch_gen['n_val'],
                                                                              logger.time("val_batch")), end="", flush=True)
            print()

            #------------ val eval -------------
            if (epoch - 1) % cf.plot_frequency == 0:
                logger.time("val_plot")
                plg.view_batch(cf, batch, results_dict, has_colorchannels=cf.has_colorchannels, show_gt_labels=True,
                               out_file=os.path.join(cf.plot_dir, 'batch_example_val_{}.png'.format(cf.fold)))
                logger.info("generated val plot in {:.2f}s".format(logger.time("val_plot")))

            logger.time("evals")
            _, monitor_metrics['val'] = val_evaluator.evaluate_predictions(val_results_list, monitor_metrics['val'])

            model_selector.run_model_selection(net, optimizer, monitor_metrics, epoch)
            del val_results_list
            #----------- monitoring -------------
            monitor_metrics.update({"lr": 
                {str(g) : group['lr'] for (g, group) in enumerate(optimizer.param_groups)}})
            logger.metrics2tboard(monitor_metrics, global_step=epoch)
            logger.time("evals")

            logger.info('finished epoch {}/{}, took {:.2f}s. train total: {:.2f}s, average: {:.2f}s. val total: {:.2f}s, average: {:.2f}s.'.format(
                epoch, cf.num_epochs, logger.get_time("train_epoch")+logger.time("val_epoch"), logger.get_time("train_epoch"),
                logger.get_time("train_epoch", reset=True)/cf.num_train_batches, logger.get_time("val_epoch"),
                logger.get_time("val_epoch", reset=True)/batch_gen["n_val"]))
            logger.info("time for evals: {:.2f}s".format(logger.get_time("evals", reset=True)))

        #-------------- scheduling -----------------
        if not cf.dynamic_lr_scheduling:
            for param_group in optimizer.param_groups:
                param_group['lr'] = cf.learning_rate[epoch-1]
        else:
            scheduler.step(monitor_metrics["val"][cf.scheduling_criterion][-1])

    logger.time("train_val")
    logger.info("Training and validating over {} epochs took {}".format(cf.num_epochs, logger.get_time("train_val", format="hms", reset=True)))
    batch_gen['train'].generator.print_stats(logger, plot=True)

def test(cf, logger, max_fold=None):
    """performs testing for a given fold (or held out set). saves stats in evaluator.
    """
    logger.time("test_fold")
    logger.info('starting testing model of fold {} in exp {}'.format(cf.fold, cf.exp_dir))
    net = model.net(cf, logger).cuda()
    batch_gen = data_loader.get_test_generator(cf, logger)

    test_predictor = Predictor(cf, net, logger, mode='test')
    test_results_list = test_predictor.predict_test_set(batch_gen, return_results = not hasattr(
        cf, "eval_test_separately") or not cf.eval_test_separately)

    if test_results_list is not None:
        test_evaluator = Evaluator(cf, logger, mode='test')
        test_evaluator.evaluate_predictions(test_results_list)
        test_evaluator.score_test_df(max_fold=max_fold)

    logger.info('Testing of fold {} took {}.'.format(cf.fold, logger.get_time("test_fold", reset=True, format="hms")))


if __name__ == '__main__':
    stime = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str,  default='train_test', help='one out of: create_exp, analysis, train, train_test, or test')
    parser.add_argument('-f', '--folds', nargs='+', type=int, default=None, help='None runs over all folds in CV. otherwise specify list of folds.')
    parser.add_argument('--exp_dir', type=str, default='/home/gregor/Documents/regrcnn/datasets/toy/experiments/dev',
                        help='path to experiment dir. will be created if non existent.')
    parser.add_argument('--server_env', default=False, action='store_true', help='change IO settings to deploy models on a cluster.')
    parser.add_argument('--data_dest', type=str, default=None, help="path to final data folder if different from config")
    parser.add_argument('--use_stored_settings', default=False, action='store_true',
                        help='load configs from existing exp_dir instead of source dir. always done for testing, '
                             'but can be set to true to do the same for training. useful in job scheduler environment, '
                             'where source code might change before the job actually runs.')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='path to checkpoint. if resuming from checkpoint, the desired fold still needs to be parsed via --folds.')
    parser.add_argument('--dataset_name', type=str, default='toy', help="path to the dataset-specific code in source_dir/datasets")
    parser.add_argument('-d', '--dev', default=False, action='store_true', help="development mode: shorten everything")

    args = parser.parse_args()
    args.dataset_name = os.path.join("datasets", args.dataset_name) if not "datasets" in args.dataset_name else args.dataset_name
    folds = args.folds
    resume_from_checkpoint = None if args.resume_from_checkpoint in ['None', 'none'] else args.resume_from_checkpoint

    if args.mode == 'create_exp':
        cf = utils.prep_exp(args.dataset_name, args.exp_dir, args.server_env, use_stored_settings=False)
        logger = utils.get_logger(cf.exp_dir, cf.server_env, -1)
        logger.info('created experiment directory at {}'.format(args.exp_dir))

    elif args.mode == 'train' or args.mode == 'train_test':
        cf = utils.prep_exp(args.dataset_name, args.exp_dir, args.server_env, args.use_stored_settings)
        if args.dev:
            folds = [0,1]
            cf.batch_size, cf.num_epochs, cf.min_save_thresh, cf.save_n_models = 3 if cf.dim==2 else 1, 1, 0, 1
            cf.num_train_batches, cf.num_val_batches, cf.max_val_patients = 5, 1, 1
            cf.test_n_epochs =  cf.save_n_models
            cf.max_test_patients = 1
            torch.backends.cudnn.benchmark = cf.dim==3
        else:
            torch.backends.cudnn.benchmark = cf.cuda_benchmark
        if args.data_dest is not None:
            cf.data_dest = args.data_dest
            
        logger = utils.get_logger(cf.exp_dir, cf.server_env, cf.sysmetrics_interval)
        data_loader = utils.import_module('data_loader', os.path.join(args.dataset_name, 'data_loader.py'))
        model = utils.import_module('model', cf.model_path)
        logger.info("loaded model from {}".format(cf.model_path))
        if folds is None:
            folds = range(cf.n_cv_splits)

        for fold in folds:
            """k-fold cross-validation: the dataset is split into k equally-sized folds, one used for validation,
            one for testing, the rest for training. This loop iterates k-times over the dataset, cyclically moving the
            splits. k==folds, fold in [0,folds) says which split is used for testing.
            """
            cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
            cf.fold, logger.fold = fold, fold
            cf.resume_from_checkpoint = resume_from_checkpoint
            if not os.path.exists(cf.fold_dir):
                os.mkdir(cf.fold_dir)
            train(cf, logger)
            cf.resume_from_checkpoint = None
            if args.mode == 'train_test':
                test(cf, logger)

    elif args.mode == 'test':
        cf = utils.prep_exp(args.dataset_name, args.exp_dir, args.server_env, use_stored_settings=True, is_training=False)
        if args.data_dest is not None:
            cf.data_dest = args.data_dest
        logger = utils.get_logger(cf.exp_dir, cf.server_env, cf.sysmetrics_interval)
        data_loader = utils.import_module('data_loader', os.path.join(args.dataset_name, 'data_loader.py'))
        model = utils.import_module('model', cf.model_path)
        logger.info("loaded model from {}".format(cf.model_path))

        fold_dirs = sorted([os.path.join(cf.exp_dir, f) for f in os.listdir(cf.exp_dir) if
                     os.path.isdir(os.path.join(cf.exp_dir, f)) and f.startswith("fold")])
        if folds is None:
            folds = range(cf.n_cv_splits)
        if args.dev:
            folds = folds[:2]
            cf.batch_size, cf.max_test_patients, cf.test_n_epochs = 1 if cf.dim==2 else 1, 2, 2
        else:
            torch.backends.cudnn.benchmark = cf.cuda_benchmark
        for fold in folds:
            cf.fold = fold
            cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(cf.fold))
            if cf.fold_dir in fold_dirs:
                test(cf, logger, max_fold=max([int(f[-1]) for f in fold_dirs]))
            else:
                logger.info("Skipping fold {} since no model parameters found.".format(fold))
    # load raw predictions saved by predictor during testing, run aggregation algorithms and evaluation.
    elif args.mode == 'analysis':
        """ analyse already saved predictions.
        """
        cf = utils.prep_exp(args.dataset_name, args.exp_dir, args.server_env, use_stored_settings=True, is_training=False)
        logger = utils.get_logger(cf.exp_dir, cf.server_env, -1)

        if cf.held_out_test_set and not cf.eval_test_fold_wise:
            predictor = Predictor(cf, net=None, logger=logger, mode='analysis')
            results_list = predictor.load_saved_predictions()
            logger.info('starting evaluation...')
            cf.fold = 0
            evaluator = Evaluator(cf, logger, mode='test')
            evaluator.evaluate_predictions(results_list)
            evaluator.score_test_df(max_fold=0)
        else:
            fold_dirs = sorted([os.path.join(cf.exp_dir, f) for f in os.listdir(cf.exp_dir) if
                         os.path.isdir(os.path.join(cf.exp_dir, f)) and f.startswith("fold")])
            if args.dev:
                fold_dirs = fold_dirs[:1]
            if folds is None:
                folds = range(cf.n_cv_splits)
            for fold in folds:
                cf.fold = fold
                cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(cf.fold))

                if cf.fold_dir in fold_dirs:
                    predictor = Predictor(cf, net=None, logger=logger, mode='analysis')
                    results_list = predictor.load_saved_predictions()
                    # results_list[x][1] is pid, results_list[x][0] is list of len samples-per-patient, each entry hlds
                    # list of boxes per that sample, i.e., len(results_list[x][y][0]) would be nr of boxes in sample y of patient x
                    logger.info('starting evaluation...')
                    evaluator = Evaluator(cf, logger, mode='test')
                    evaluator.evaluate_predictions(results_list)
                    max_fold = max([int(f[-1]) for f in fold_dirs])
                    evaluator.score_test_df(max_fold=max_fold)
                else:
                    logger.info("Skipping fold {} since no model parameters found.".format(fold))
    else:
        raise ValueError('mode "{}" specified in args is not implemented.'.format(args.mode))
        
    mins, secs = divmod((time.time() - stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    logger.info("{} total runtime: {}".format(os.path.split(__file__)[1], t))
    del logger
    torch.cuda.empty_cache()



