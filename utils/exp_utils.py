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
from typing import Union, Iterable, Tuple, Any
import sys
import os
import subprocess
from multiprocessing import Process
import threading
import pickle
import importlib.util
import psutil
import time
import nvidia_smi

import logging
from torch.utils.tensorboard import SummaryWriter

from collections import OrderedDict
import numpy as np
import pandas as pd
import torch


def import_module(name, path):
    """
    correct way of importing a module dynamically in python 3.
    :param name: name given to module instance.
    :param path: path to module.
    :return: module: returned module instance.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_obj(obj, name):
    """Pickle a python object."""
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def IO_safe(func, *args, _tries=5, _raise=True, **kwargs):
    """ Wrapper calling function func with arguments args and keyword arguments kwargs to catch input/output errors
        on cluster.
    :param func: function to execute (intended to be read/write operation to a problematic cluster drive, but can be
        any function).
    :param args: positional args of func.
    :param kwargs: kw args of func.
    :param _tries: how many attempts to make executing func.
    """
    for _try in range(_tries):
        try:
            return func(*args, **kwargs)
        except OSError as e:  # to catch cluster issues with network drives
            if _raise:
                raise e
            else:
                print("After attempting execution {} time{}, following error occurred:\n{}".format(_try + 1,
                                                                                                   "" if _try == 0 else "s",
                                                                                                   e))
                continue

def split_off_process(target, *args, daemon=False, **kwargs):
    """Start a process that won't block parent script.
    No join(), no return value. If daemon=False: before parent exits, it waits for this to finish.
    """
    p = Process(target=target, args=tuple(args), kwargs=kwargs, daemon=daemon)
    p.start()
    return p


def query_nvidia_gpu(device_id, d_keyword=None, no_units=False):
    """
    :param device_id:
    :param d_keyword: -d, --display argument (keyword(s) for selective display), all are selected if None
    :return: dict of gpu-info items
    """
    cmd = ['nvidia-smi', '-i', str(device_id), '-q']
    if d_keyword is not None:
        cmd += ['-d', d_keyword]
    outp = subprocess.check_output(cmd).strip().decode('utf-8').split("\n")
    outp = [x for x in outp if len(x) > 0]
    headers = [ix for ix, item in enumerate(outp) if len(item.split(":")) == 1] + [len(outp)]

    out_dict = {}
    for lix, hix in enumerate(headers[:-1]):
        head = outp[hix].strip().replace(" ", "_").lower()
        out_dict[head] = {}
        for lix2 in range(hix, headers[lix + 1]):
            try:
                key, val = [x.strip().lower() for x in outp[lix2].split(":")]
                if no_units:
                    val = val.split()[0]
                out_dict[head][key] = val
            except:
                pass

    return out_dict

class _AnsiColorizer(object):
    """
    A colorizer is an object that loosely wraps around a stream, allowing
    callers to write text to the stream in a particular color.

    Colorizer classes must implement C{supported()} and C{write(text, color)}.
    """
    _colors = dict(black=30, red=31, green=32, yellow=33,
                   blue=34, magenta=35, cyan=36, white=37, default=39)

    def __init__(self, stream):
        self.stream = stream

    @classmethod
    def supported(cls, stream=sys.stdout):
        """
        A class method that returns True if the current platform supports
        coloring terminal output using this method. Returns False otherwise.
        """
        if not stream.isatty():
            return False  # auto color only on TTYs
        try:
            import curses
        except ImportError:
            return False
        else:
            try:
                try:
                    return curses.tigetnum("colors") > 2
                except curses.error:
                    curses.setupterm()
                    return curses.tigetnum("colors") > 2
            except:
                raise
                # guess false in case of error
                return False

    def write(self, text, color):
        """
        Write the given text to the stream in the given color.

        @param text: Text to be written to the stream.

        @param color: A string label for a color. e.g. 'red', 'white'.
        """
        color = self._colors[color]
        self.stream.write('\x1b[%sm%s\x1b[0m' % (color, text))

class ColorHandler(logging.StreamHandler):

    def __init__(self, stream=sys.stdout):
        super(ColorHandler, self).__init__(_AnsiColorizer(stream))

    def emit(self, record):
        msg_colors = {
            logging.DEBUG: "green",
            logging.INFO: "default",
            logging.WARNING: "red",
            logging.ERROR: "red"
        }
        color = msg_colors.get(record.levelno, "blue")
        self.stream.write(record.msg + "\n", color)

class CombinedPrinter(object):
    """combined print function.
    prints to logger and/or file if given, to normal print if non given.

    """

    def __init__(self, logger=None, file=None):

        if logger is None and file is None:
            self.out = [print]
        elif logger is None:
            self.out = [file.write]
        elif file is None:
            self.out = [logger.info]
        else:
            self.out = [logger.info, file.write]

    def __call__(self, string):
        for fct in self.out:
            fct(string)

class Nvidia_GPU_Logger(object):
    def __init__(self):
        self.count = None

    def get_vals(self):

        # cmd = ['nvidia-settings', '-t', '-q', 'GPUUtilization']
        # gpu_util = subprocess.check_output(cmd).strip().decode('utf-8').split(",")
        # gpu_util = dict([f.strip().split("=") for f in gpu_util])
        # cmd[-1] = 'UsedDedicatedGPUMemory'
        # gpu_used_mem = subprocess.check_output(cmd).strip().decode('utf-8')


        nvidia_smi.nvmlInit()
        # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
        self.gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        util_res = nvidia_smi.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        #mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        # current_vals = {"gpu_mem_alloc": mem_res.used / (1024**2), "gpu_graphics_util": int(gpu_util['graphics']),
        #                 "gpu_mem_util": gpu_util['memory'], "time": time.time()}
        current_vals = {"gpu_graphics_util": float(util_res.gpu),
                        "time": time.time()}
        return current_vals

    def loop(self, interval):
        i = 0
        while True:
            current_vals = self.get_vals()
            self.log["time"].append(time.time())
            self.log["gpu_util"].append(current_vals["gpu_graphics_util"])
            if self.count is not None:
                i += 1
                if i == self.count:
                    exit(0)
            time.sleep(self.interval)

    def start(self, interval=1.):
        self.interval = interval
        self.start_time = time.time()
        self.log = {"time": [], "gpu_util": []}
        if self.interval is not None:
            thread = threading.Thread(target=self.loop)
            thread.daemon = True
            thread.start()

class CombinedLogger(object):
    """Combine console and tensorboard logger and record system metrics.
    """

    def __init__(self, name, log_dir, server_env=True, fold="all", sysmetrics_interval=2):
        self.pylogger = logging.getLogger(name)
        self.tboard = SummaryWriter(log_dir=os.path.join(log_dir, "tboard"))
        self.times = {}
        self.log_dir = log_dir
        self.fold = str(fold)
        self.server_env = server_env

        self.pylogger.setLevel(logging.DEBUG)
        self.log_file = os.path.join(log_dir, "fold_"+self.fold, 'exec.log')
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.pylogger.addHandler(logging.FileHandler(self.log_file))
        if not server_env:
            self.pylogger.addHandler(ColorHandler())
        else:
            self.pylogger.addHandler(logging.StreamHandler())
        self.pylogger.propagate = False

        # monitor system metrics (cpu, mem, ...)
        if not server_env and sysmetrics_interval > 0:
            self.sysmetrics = pd.DataFrame(
                columns=["global_step", "rel_time", r"CPU (%)", "mem_used (GB)", r"mem_used (%)",
                         r"swap_used (GB)", r"gpu_utilization (%)"], dtype="float16")
            for device in range(torch.cuda.device_count()):
                self.sysmetrics[
                    "mem_allocd (GB) by torch on {:10s}".format(torch.cuda.get_device_name(device))] = np.nan
                self.sysmetrics[
                    "mem_cached (GB) by torch on {:10s}".format(torch.cuda.get_device_name(device))] = np.nan
            self.sysmetrics_start(sysmetrics_interval)
            pass
        else:
            print("NOT logging sysmetrics")

    def __getattr__(self, attr):
        """delegate all undefined method requests to objects of
        this class in order pylogger, tboard (first find first serve).
        E.g., combinedlogger.add_scalars(...) should trigger self.tboard.add_scalars(...)
        """
        for obj in [self.pylogger, self.tboard]:
            if attr in dir(obj):
                return getattr(obj, attr)
        print("logger attr not found")
        #raise AttributeError("CombinedLogger has no attribute {}".format(attr))

    def set_logfile(self, fold=None, log_file=None):
        if fold is not None:
            self.fold = str(fold)
        if log_file is None:
            self.log_file = os.path.join(self.log_dir, "fold_"+self.fold, 'exec.log')
        else:
            self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        for hdlr in self.pylogger.handlers:
            hdlr.close()
        self.pylogger.handlers = []
        self.pylogger.addHandler(logging.FileHandler(self.log_file))
        if not self.server_env:
            self.pylogger.addHandler(ColorHandler())
        else:
            self.pylogger.addHandler(logging.StreamHandler())

    def time(self, name, toggle=None):
        """record time-spans as with a stopwatch.
        :param name:
        :param toggle: True^=On: start time recording, False^=Off: halt rec. if None determine from current status.
        :return: either start-time or last recorded interval
        """
        if toggle is None:
            if name in self.times.keys():
                toggle = not self.times[name]["toggle"]
            else:
                toggle = True

        if toggle:
            if not name in self.times.keys():
                self.times[name] = {"total": 0, "last": 0}
            elif self.times[name]["toggle"] == toggle:
                self.info("restarting running stopwatch")
            self.times[name]["last"] = time.time()
            self.times[name]["toggle"] = toggle
            return time.time()
        else:
            if toggle == self.times[name]["toggle"]:
                self.info("WARNING: tried to stop stopped stop watch: {}.".format(name))
            self.times[name]["last"] = time.time() - self.times[name]["last"]
            self.times[name]["total"] += self.times[name]["last"]
            self.times[name]["toggle"] = toggle
            return self.times[name]["last"]

    def get_time(self, name=None, kind="total", format=None, reset=False):
        """
        :param name:
        :param kind: 'total' or 'last'
        :param format: None for float, "hms"/"ms" for (hours), mins, secs as string
        :param reset: reset time after retrieving
        :return:
        """
        if name is None:
            times = self.times
            if reset:
                self.reset_time()
            return times

        else:
            if self.times[name]["toggle"]:
                self.time(name, toggle=False)
            time = self.times[name][kind]
            if format == "hms":
                m, s = divmod(time, 60)
                h, m = divmod(m, 60)
                time = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(m), int(s))
            elif format == "ms":
                m, s = divmod(time, 60)
                time = "{:02d}m:{:02d}s".format(int(m), int(s))
            if reset:
                self.reset_time(name)
            return time

    def reset_time(self, name=None):
        if name is None:
            self.times = {}
        else:
            del self.times[name]

    def sysmetrics_update(self, global_step=None):
        if global_step is None:
            global_step = time.strftime("%x_%X")
        mem = psutil.virtual_memory()
        mem_used = (mem.total - mem.available)
        gpu_vals = self.gpu_logger.get_vals()
        rel_time = time.time() - self.sysmetrics_start_time
        self.sysmetrics.loc[len(self.sysmetrics)] = [global_step, rel_time,
                                                     psutil.cpu_percent(), mem_used / 1024 ** 3,
                                                     mem_used / mem.total * 100,
                                                     psutil.swap_memory().used / 1024 ** 3,
                                                     int(gpu_vals['gpu_graphics_util']),
                                                     *[torch.cuda.memory_allocated(d) / 1024 ** 3 for d in
                                                       range(torch.cuda.device_count())],
                                                     *[torch.cuda.memory_cached(d) / 1024 ** 3 for d in
                                                       range(torch.cuda.device_count())]
                                                     ]
        return self.sysmetrics.loc[len(self.sysmetrics) - 1].to_dict()

    def sysmetrics2tboard(self, metrics=None, global_step=None, suptitle=None):
        tag = "per_time"
        if metrics is None:
            metrics = self.sysmetrics_update(global_step=global_step)
            tag = "per_epoch"

        if suptitle is not None:
            suptitle = str(suptitle)
        elif self.fold != "":
            suptitle = "Fold_" + str(self.fold)
        if suptitle is not None:
            self.tboard.add_scalars(suptitle + "/System_Metrics/" + tag,
                                    {k: v for (k, v) in metrics.items() if (k != "global_step"
                                                                            and k != "rel_time")}, global_step)

    def sysmetrics_loop(self):
        try:
            os.nice(-19)
            self.info("Logging system metrics with superior process priority.")
        except:
            self.info("Logging system metrics without superior process priority.")
        while True:
            metrics = self.sysmetrics_update()
            self.sysmetrics2tboard(metrics, global_step=metrics["rel_time"])
            # print("thread alive", self.thread.is_alive())
            time.sleep(self.sysmetrics_interval)

    def sysmetrics_start(self, interval):
        if interval is not None and interval > 0:
            self.sysmetrics_interval = interval
            self.gpu_logger = Nvidia_GPU_Logger()
            self.sysmetrics_start_time = time.time()
            self.sys_metrics_process = split_off_process(target=self.sysmetrics_loop, daemon=True)
            # self.thread = threading.Thread(target=self.sysmetrics_loop)
            # self.thread.daemon = True
            # self.thread.start()

    def sysmetrics_save(self, out_file):
        self.sysmetrics.to_pickle(out_file)

    def metrics2tboard(self, metrics, global_step=None, suptitle=None):
        """
        :param metrics: {'train': dataframe, 'val':df}, df as produced in
            evaluator.py.evaluate_predictions
        """
        # print("metrics", metrics)
        if global_step is None:
            global_step = len(metrics['train'][list(metrics['train'].keys())[0]]) - 1
        if suptitle is not None:
            suptitle = str(suptitle)
        else:
            suptitle = "Fold_" + str(self.fold)

        for key in ['train', 'val']:
            # series = {k:np.array(v[-1]) for (k,v) in metrics[key].items() if not np.isnan(v[-1]) and not 'Bin_Stats' in k}
            loss_series = {}
            unc_series = {}
            bin_stat_series = {}
            mon_met_series = {}
            for tag, val in metrics[key].items():
                val = val[-1]  # maybe remove list wrapping, recording in evaluator?
                if 'bin_stats' in tag.lower() and not np.isnan(val):
                    bin_stat_series["{}".format(tag.split("/")[-1])] = val
                elif 'uncertainty' in tag.lower() and not np.isnan(val):
                    unc_series["{}".format(tag)] = val
                elif 'loss' in tag.lower() and not np.isnan(val):
                    loss_series["{}".format(tag)] = val
                elif not np.isnan(val):
                    mon_met_series["{}".format(tag)] = val

            self.tboard.add_scalars(suptitle + "/Binary_Statistics/{}".format(key), bin_stat_series, global_step)
            self.tboard.add_scalars(suptitle + "/Uncertainties/{}".format(key), unc_series, global_step)
            self.tboard.add_scalars(suptitle + "/Losses/{}".format(key), loss_series, global_step)
            self.tboard.add_scalars(suptitle + "/Monitor_Metrics/{}".format(key), mon_met_series, global_step)
        self.tboard.add_scalars(suptitle + "/Learning_Rate", metrics["lr"], global_step)
        return

    def batchImgs2tboard(self, batch, results_dict, cmap, boxtype2color, img_bg=False, global_step=None):
        raise NotImplementedError("not up-to-date, problem with importing plotting-file, torchvision dependency.")
        if len(batch["seg"].shape) == 5:  # 3D imgs
            slice_ix = np.random.randint(batch["seg"].shape[-1])
            seg_gt = plg.to_rgb(batch['seg'][:, 0, :, :, slice_ix], cmap)
            seg_pred = plg.to_rgb(results_dict['seg_preds'][:, 0, :, :, slice_ix], cmap)

            mod_img = plg.mod_to_rgb(batch["data"][:, 0, :, :, slice_ix]) if img_bg else None

        elif len(batch["seg"].shape) == 4:
            seg_gt = plg.to_rgb(batch['seg'][:, 0, :, :], cmap)
            seg_pred = plg.to_rgb(results_dict['seg_preds'][:, 0, :, :], cmap)
            mod_img = plg.mod_to_rgb(batch["data"][:, 0]) if img_bg else None
        else:
            raise Exception("batch content has wrong format: {}".format(batch["seg"].shape))

        # from here on only works in 2D
        seg_gt = np.transpose(seg_gt, axes=(0, 3, 1, 2))  # previous shp: b,x,y,c
        seg_pred = np.transpose(seg_pred, axes=(0, 3, 1, 2))

        seg = np.concatenate((seg_gt, seg_pred), axis=0)
        # todo replace torchvision (tv) dependency
        seg = tv.utils.make_grid(torch.from_numpy(seg), nrow=2)
        self.tboard.add_image("Batch seg, 1st col: gt, 2nd: pred.", seg, global_step=global_step)

        if img_bg:
            bg_img = np.transpose(mod_img, axes=(0, 3, 1, 2))
        else:
            bg_img = seg_gt
        box_imgs = plg.draw_boxes_into_batch(bg_img, results_dict["boxes"], boxtype2color)
        box_imgs = tv.utils.make_grid(torch.from_numpy(box_imgs), nrow=4)
        self.tboard.add_image("Batch bboxes", box_imgs, global_step=global_step)

        return

    def __del__(self):  # otherwise might produce multiple prints e.g. in ipython console
        #self.sys_metrics_process.terminate()
        for hdlr in self.pylogger.handlers:
            hdlr.close()
        self.pylogger.handlers = []
        del self.pylogger
        self.tboard.flush()
        # close holds up main script exit. maybe revise this issue with a later pytorch version.
        #self.tboard.close()

def get_logger(exp_dir, server_env=False, sysmetrics_interval=2):
    log_dir = os.path.join(exp_dir, "logs")
    logger = CombinedLogger('Reg R-CNN', log_dir, server_env=server_env,
                            sysmetrics_interval=sysmetrics_interval)
    print("logging to {}".format(logger.log_file))
    return logger

def prepare_monitoring(cf):
    """
    creates dictionaries, where train/val metrics are stored.
    """
    metrics = {}
    # first entry for loss dict accounts for epoch starting at 1.
    metrics['train'] = OrderedDict()  # [(l_name, [np.nan]) for l_name in cf.losses_to_monitor] )
    metrics['val'] = OrderedDict()  # [(l_name, [np.nan]) for l_name in cf.losses_to_monitor] )
    metric_classes = []
    if 'rois' in cf.report_score_level:
        metric_classes.extend([v for k, v in cf.class_dict.items()])
        if hasattr(cf, "eval_bins_separately") and cf.eval_bins_separately:
            metric_classes.extend([v for k, v in cf.bin_dict.items()])
    if 'patient' in cf.report_score_level:
        metric_classes.extend(['patient_' + cf.class_dict[cf.patient_class_of_interest]])
        if hasattr(cf, "eval_bins_separately") and cf.eval_bins_separately:
            metric_classes.extend(['patient_' + cf.bin_dict[cf.patient_bin_of_interest]])
    for cl in metric_classes:
        for m in cf.metrics:
            metrics['train'][cl + '_' + m] = [np.nan]
            metrics['val'][cl + '_' + m] = [np.nan]

    return metrics


class ModelSelector:
    '''
    saves a checkpoint after each epoch as 'last_state' (can be loaded to continue interrupted training).
    saves the top-k (k=cf.save_n_models) ranked epochs. In inference, predictions of multiple epochs can be ensembled
    to improve performance.
    '''

    def __init__(self, cf, logger):

        self.cf = cf
        self.logger = logger

        self.model_index = pd.DataFrame(columns=["rank", "score", "criteria_values", "file_name"],
                                        index=pd.RangeIndex(self.cf.min_save_thresh, self.cf.num_epochs, name="epoch"))

    def run_model_selection(self, net, optimizer, monitor_metrics, epoch):
        """rank epoch via weighted mean from self.cf.model_selection_criteria: {criterion : weight}
        :param net:
        :param optimizer:
        :param monitor_metrics:
        :param epoch:
        :return:
        """
        crita = self.cf.model_selection_criteria  # shorter alias
        metrics =  monitor_metrics['val']

        epoch_score = np.sum([metrics[criterion][-1] * weight for criterion, weight in crita.items() if
                              not np.isnan(metrics[criterion][-1])])
        if not self.cf.resume:
            epoch_score_check = np.sum([metrics[criterion][epoch] * weight for criterion, weight in crita.items() if
                                  not np.isnan(metrics[criterion][epoch])])
            assert np.all(epoch_score == epoch_score_check)

        self.model_index.loc[epoch, ["score", "criteria_values"]] = epoch_score, {cr: metrics[cr][-1] for cr in crita.keys()}

        nonna_ics = self.model_index["score"].dropna(axis=0).index
        order = np.argsort(self.model_index.loc[nonna_ics, "score"].to_numpy(), kind="stable")[::-1]
        self.model_index.loc[nonna_ics, "rank"] = np.argsort(order) + 1 # no zero-indexing for ranks (best rank is 1).

        rank = int(self.model_index.loc[epoch, "rank"])
        if rank <= self.cf.save_n_models:
            name = '{}_best_params.pth'.format(epoch)
            if self.cf.server_env:
                IO_safe(torch.save, net.state_dict(), os.path.join(self.cf.fold_dir, name))
            else:
                torch.save(net.state_dict(), os.path.join(self.cf.fold_dir, name))
            self.model_index.loc[epoch, "file_name"] = name
            self.logger.info("saved current epoch {} at rank {}".format(epoch, rank))

            clean_up = self.model_index.dropna(axis=0, subset=["file_name"])
            clean_up = clean_up[clean_up["rank"] > self.cf.save_n_models]
            if clean_up.size > 0:
                file_name = clean_up["file_name"].to_numpy().item()
                subprocess.call("rm {}".format(os.path.join(self.cf.fold_dir, file_name)), shell=True)
                self.logger.info("removed outranked epoch {} at {}".format(clean_up.index.values.item(),
                                                                       os.path.join(self.cf.fold_dir, file_name)))
                self.model_index.loc[clean_up.index, "file_name"] = np.nan

        state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_index': self.model_index,
        }

        if self.cf.server_env:
            IO_safe(torch.save, state, os.path.join(self.cf.fold_dir, 'last_state.pth'))
        else:
            torch.save(state, os.path.join(self.cf.fold_dir, 'last_state.pth'))


def set_params_flag(module: torch.nn.Module, flag: Tuple[str, Any], check_overwrite: bool = True):
    """Set an attribute for all module parameters.

    :param flag: tuple (str attribute name : attr value)
    :param check_overwrite: if True, assert that attribute not already exists.

    """

    for param in module.parameters():
        if check_overwrite:
            assert not hasattr(param, flag[0]), \
                "param {} already has attr {} (w/ val {})".format(param, flag[0], getattr(param, flag[0]))
        setattr(param, flag[0], flag[1])
    return module

def parse_params_for_optim(net: torch.nn.Module, weight_decay: float = 0., exclude_from_wd: Iterable = ("norm",)):
    """Format network parameters for the optimizer.
    Convenience function to include options for group-specific settings like weight decay.
    :param net:
    :param weight_decay:
    :param exclude_from_wd: List of strings of parameter-group names to exclude from weight decay. Options: "norm", "bias".
    :return:
    """
    # pytorch implements parameter groups as dicts {'params': ...} and
    # weight decay as p.data.mul_(1 - group['lr'] * group['weight_decay'])
    norm_types = [torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                  torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
                  torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.SyncBatchNorm, torch.nn.LocalResponseNorm
                  ]
    level_map = {"bias": "weight",
                 "norm": "module"}
    type_map = {"norm": norm_types}

    exclude_from_wd = [str(name).lower() for name in exclude_from_wd]
    exclude_weight_names = [k for k, v in level_map.items() if k in exclude_from_wd and v == "weight"]
    exclude_module_types = tuple([type_ for k, v in level_map.items() if (k in exclude_from_wd and v == "module")
                                  for type_ in type_map[k]])

    if exclude_from_wd:
        print("excluding {} from weight decay.".format(exclude_from_wd))

    for module in net.modules():
        if isinstance(module, exclude_module_types):
            set_params_flag(module, ("no_wd", True))
    for param_name, param in net.named_parameters():
        if np.any([ename in param_name for ename in exclude_weight_names]):
            setattr(param, "no_wd", True)

    with_dec, no_dec = [], []
    for param in net.parameters():
        if hasattr(param, "no_wd") and param.no_wd == True:
            no_dec.append(param)
        else:
            with_dec.append(param)

    orig_ps = sum(p.numel() for p in net.parameters())
    with_ps = sum(p.numel() for p in with_dec)
    wo_ps = sum(p.numel() for p in no_dec)
    assert orig_ps == with_ps + wo_ps, "orig n parameters {} unequals sum of with wd {} and w/o wd {}."\
        .format(orig_ps, with_ps, wo_ps)
    groups = [{'params': gr, 'weight_decay': wd} for (gr, wd) in [(no_dec, 0.), (with_dec, weight_decay)] if len(gr)>0]
    return groups

def load_checkpoint(checkpoint_path, net, optimizer, model_selector):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model_selector.model_index = checkpoint["model_index"]
    return checkpoint['epoch'] + 1, net, optimizer, model_selector

def prep_exp(dataset_path, exp_path, server_env, use_stored_settings=True, is_training=True):
    """
    I/O handling, creating of experiment folder structure. Also creates a snapshot of configs/model scripts and copies them to the exp_dir.
    This way the exp_dir contains all info needed to conduct an experiment, independent to changes in actual source code. Thus, training/inference of this experiment can be started at anytime.
    Therefore, the model script is copied back to the source code dir as tmp_model (tmp_backbone).
    Provides robust structure for cloud deployment.
    :param dataset_path: path to source code for specific data set. (e.g. medicaldetectiontoolkit/lidc_exp)
    :param exp_path: path to experiment directory.
    :param server_env: boolean flag. pass to configs script for cloud deployment.
    :param use_stored_settings: boolean flag. When starting training: If True, starts training from snapshot in existing
        experiment directory, else creates experiment directory on the fly using configs/model scripts from source code.
    :param is_training: boolean flag. distinguishes train vs. inference mode.
    :return: configs object.
    """

    if is_training:

        if use_stored_settings:
            cf_file = import_module('cf', os.path.join(exp_path, 'configs.py'))
            cf = cf_file.Configs(server_env)
            # in this mode, previously saved model and backbone need to be found in exp dir.
            if not os.path.isfile(os.path.join(exp_path, 'model.py')) or \
                    not os.path.isfile(os.path.join(exp_path, 'backbone.py')):
                raise Exception(
                    "Selected use_stored_settings option but no model and/or backbone source files exist in exp dir.")
            cf.model_path = os.path.join(exp_path, 'model.py')
            cf.backbone_path = os.path.join(exp_path, 'backbone.py')
        else:  # this case overwrites settings files in exp dir, i.e., default_configs, configs, backbone, model
            os.makedirs(exp_path, exist_ok=True)
            # run training with source code info and copy snapshot of model to exp_dir for later testing (overwrite scripts if exp_dir already exists.)
            subprocess.call('cp {} {}'.format('default_configs.py', os.path.join(exp_path, 'default_configs.py')),
                            shell=True)
            subprocess.call(
                'cp {} {}'.format(os.path.join(dataset_path, 'configs.py'), os.path.join(exp_path, 'configs.py')),
                shell=True)
            cf_file = import_module('cf_file', os.path.join(dataset_path, 'configs.py'))
            cf = cf_file.Configs(server_env)
            subprocess.call('cp {} {}'.format(cf.model_path, os.path.join(exp_path, 'model.py')), shell=True)
            subprocess.call('cp {} {}'.format(cf.backbone_path, os.path.join(exp_path, 'backbone.py')), shell=True)
            if os.path.isfile(os.path.join(exp_path, "fold_ids.pickle")):
                subprocess.call('rm {}'.format(os.path.join(exp_path, "fold_ids.pickle")), shell=True)

    else:  # testing, use model and backbone stored in exp dir.
        cf_file = import_module('cf', os.path.join(exp_path, 'configs.py'))
        cf = cf_file.Configs(server_env)
        cf.model_path = os.path.join(exp_path, 'model.py')
        cf.backbone_path = os.path.join(exp_path, 'backbone.py')

    cf.exp_dir = exp_path
    cf.test_dir = os.path.join(cf.exp_dir, 'test')
    cf.plot_dir = os.path.join(cf.exp_dir, 'plots')
    if not os.path.exists(cf.test_dir):
        os.mkdir(cf.test_dir)
    if not os.path.exists(cf.plot_dir):
        os.mkdir(cf.plot_dir)
    cf.experiment_name = exp_path.split("/")[-1]
    cf.dataset_name = dataset_path
    cf.server_env = server_env
    cf.created_fold_id_pickle = False

    return cf
