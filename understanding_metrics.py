
"""
Created at 06/12/18 13:34
@author: gregor 
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve

import plotting as plg
import evaluator

sys.path.append("datasets/prostate/")
from configs import Configs

""" This is just a supplementary file which you may use to demonstrate or understand detection metrics.
"""


def get_det_types(df):
    det_types = []
    for ix, score in enumerate(df["pred_score"]):
        if score > 0 and df["class_label"][ix] == 1:
            det_types.append("det_tp")
        elif score > 0 and df["class_label"][ix] == 0:
            det_types.append("det_fp")
        elif score == 0 and df["class_label"][ix] == 1:
            det_types.append("det_fn")
        elif score == 0 and df["class_label"][ix] == 0:
            det_types.append("det_tn")
    return det_types


if __name__=="__main__":
    cf = Configs()

    working_dir = "/home/gregor/Documents/ramien/Thesis/UnderstandingMetrics"

    df = pd.DataFrame(columns=['pred_score', 'class_label', 'pred_class', 'det_type', 'match_iou'])

    df["pred_score"] = [0.3,  0.]
    df["class_label"] = [0,   1]
    #df["pred_class"] = [1]*len(df)
    det_types = get_det_types(df)

    df["det_type"] = det_types
    df["match_iou"] = [0.1]*len(df)

    prc_own = evaluator.compute_prc(df)
    all_stats = [{"prc":prc_own, 'roc':np.nan, 'name': "demon"}]
    plg.plot_stat_curves(cf, all_stats, os.path.join(working_dir, "understanding_ap_own"), fill=True)

    prc_sk = precision_recall_curve(df.class_label.tolist(), df.pred_score.tolist())
    all_stats = [{"prc":prc_sk, 'roc':np.nan, 'name': "demon"}]
    plg.plot_stat_curves(cf, all_stats, os.path.join(working_dir, "understanding_ap"), fill=True)

    ap = evaluator.get_roi_ap_from_df((df, 0.02, False))
    ap_sk = average_precision_score(df.class_label.tolist(), df.pred_score.tolist())
    print("roi_ap_from_df (own implement):",ap)
    print("aver_prec_sc (sklearn):",ap_sk)

    plg.plot_prediction_hist(cf, df, os.path.join(working_dir, "understanding_ap.png"), title="AP_own {:.2f}, AP_sklearn {:.2f}".format(ap, ap_sk))

