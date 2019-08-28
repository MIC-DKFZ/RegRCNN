"""
Created on Fri Jun 22 14:47:15 2018

@author: gregor
"""
import plotting as plg

import warnings
import os
import sys
import subprocess
import time
import shutil
import psutil
import pickle
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import math
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from PIL import Image

from collections import namedtuple

import SimpleITK as sitk
from skimage import transform as tf
from sklearn.metrics import roc_curve, precision_recall_curve
import nvidia_smi

import evaluator

sys.path.append("datasets/prostate/")
from configs import Configs
cf = configs()
import utils.exp_utils as utils


blue = [61, 114, 242]
red = [215, 11, 2]
green= [139, 172, 48]
orange = [255, 125, 0]

#for c in cf.color_palette:
    #plt.imshow([[c]])
    #plt.show()


#filepath = "/home/gregor/Documents/data/prostate/data_070918/Master_00091_merged_seg.npy"
#nfilepath = "/home/gregor/Documents/data/prostate/data_130918_npz/Master_00091_merged_seg.npz"
#imgpath = "/home/gregor/Documents/data/prostate/data_070918/Master_00091_imgs.npy"
#nimgpath = "/home/gregor/Documents/data/prostate/data_130918_npz/Master_00091_imgs.npz"
#imgs = np.load(imgpath)
#img3d = imgs[0]
#seg3d = np.load(filepath)
##plg.view_slices(cf, img3d, seg3d, title="before registration", out_dir="prostate/experiments/dev/bef_reg.png")
#img3d = np.load(nimgpath)
#img3d = img3d[img3d.keys()[0]][0]
#seg3d = np.load(nfilepath)
##print("seg keys", seg3d.keys())
#seg3d = seg3d[seg3d.keys()[0]]
#print("types", type(img3d), type(seg3d))
#print("sahpes", img3d.shape, seg3d.shape)

#plg.view_slices(cf, img3d, seg3d, title="after registration", out_dir="prostate/experiments/dev/after_reg.png")


dic = {'a':{'targ':[1]}, 'b':{'targ':[2,3]}, 'c':{'targ':[]}, 'd':{'notarg':[4]}}

arr = np.random.randn(3,3,2)
arr = np.random.randint(0,4, size=4)
lst = [[-1], [-1], [-1], [2, 1]]

arr = np.array([[1,1,0],
       [0,0,0],
       [3,4,0],
       [1,2,3]])

mscores = np.array([[2]])

def agg_regress(x):
    print("print", type(x), x)

    norm = x.apply(np.linalg.norm)
    loc = np.argmax(norm.values)

    return list(np.array(x.iloc[loc]))


# df = pd.DataFrame({"pid":[1, 1, 2, 2],
#                    "regression":[[1.,1.,1.], [1.,2.,1.], [5., 4., 4.], [5., 3., 2.]],
#                    "score":[0.5, 0.1, 0.2, 0.4]})
#
# Label = namedtuple("Label", ['id', 'bin_values'])
#
# bin_labels = [ Label(0, (0,)), Label(1, (1,2,3)), Label(2, (4,5))]
#
#
# colors = {
#         "black" : (7, 54, 66),
#         "brblack" : ( 0, 43, 54),
#         "brgreen" : (88, 110, 117),
#         "brblue" : (131, 148, 150),
#         "cyan" : (42, 161, 152),
#         "white" : (253, 246, 227),
#         "yellow" : (181, 137,   0),
#         "orange" : (203, 75 , 22),
#         "red" : (220,  50 , 47),
#         "aubergine" : (108, 113, 196),
#         "magenta" : (211,  54, 130),
#         "blue" : (38, 139, 210 ),
#         "green" : (133, 153,   0),
#         }
#
# for cn, cv in colors.items():
#     cv = np.array(cv)/255.
#     print("self.{} = {}".format(cn, "("+",".join(["{:.3f}".format(v) for v in cv])+")"))



def func(a, b=0):
    print("a {}, b {}".format(a, b))

utils.IO_safe(func, 1, b=2)

pass
