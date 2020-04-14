"""
Created at 4/8/20 4:24 PM
@author: gregor 
"""

import os, time
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import argparse
from multiprocessing import Pool
from collections import OrderedDict

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

def load_data(cf):
    pp_data_path = cf.pp_data_path
    p_df = pd.read_pickle(os.path.join(pp_data_path, cf.input_df_name))

    class_targets = p_df['class_id'].tolist()
    pids = p_df.pid.tolist()
    imgs = [os.path.join(pp_data_path, '{}.npy'.format(pid)) for pid in pids]
    segs = [os.path.join(pp_data_path,'{}.npy'.format(pid)) for pid in pids]

    data = OrderedDict()
    for ix, pid in enumerate(pids):

        data[pid] = {'data': imgs[ix], 'seg': segs[ix], 'pid': pid, 'class_target': [class_targets[ix]]}

    return data


def plot_data_and_gt(cf, data, n_samples=14, out_dir=None):

    sample_keys = np.random.choice(list(data.keys()), size=n_samples)

    fig = plt.figure(figsize=(n_samples*2, 4))
    grid = gridspec.GridSpec(2,n_samples)

    for s_ix, skey in enumerate(sample_keys):
        sample = data[skey]
        img = np.load(sample["data"])[0]
        seg = np.load(sample["seg"])[1]
        gt_class = sample["class_target"]
        ax = fig.add_subplot(grid[0, s_ix])
        ax.imshow(img)
        ax.set_title("img")
        ax.axis("off")
        ax = fig.add_subplot(grid[1, s_ix])
        ax.imshow(seg)
        ax.set_title("seg. gt_class: {}".format(gt_class))
        ax.axis("off")


    if out_dir is not None:
        out_file = out_dir / "check_samples.png"
        plt.savefig(str(out_file), dpi=200, bbox_inches="tight")




if __name__ == '__main__':
    stime = time.time()
    import sys
    sys.path.append("../..")
    import utils.exp_utils as utils

    parser = argparse.ArgumentParser()
    args = parser.parse_args()


    cf_file = utils.import_module("cf", "configs.py")
    cf = cf_file.configs()

    data = load_data(cf)
    plot_data_and_gt(cf, data, out_dir=Path("/media/gregor/HDD1/experiments/mdt/toy_1x/data_check"))


    mins, secs = divmod((time.time() - stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))