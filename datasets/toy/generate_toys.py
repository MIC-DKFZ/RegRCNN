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

""" Generate a data set of toy examples. Examples can be cylinders, spheres, blocks, diamonds.
    Distortions may be applied, e.g., noise to the radius ground truths.
    Settings are configured in configs file.
"""

import plotting as plg
import os
import shutil
import warnings
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import data_manager as dmanager


for msg in ["RuntimeWarning: divide by zero encountered in true_divide.*",]:
    warnings.filterwarnings("ignore", msg)


class ToyGenerator(object):
    """ Generator of toy data set.
        A train and a test split with certain nr of samples are created and saved to disk. Samples can contain varying
        number of objects. Objects have shapes cylinder or block (diamond, ellipsoid, torus not fully implemented).

        self.mp_args holds image split and id, objects are then randomly drawn into each image. Multi-processing is
        enabled for parallel creation of images, final .npy-files can then be converted to .npz.
    """
    def __init__(self, cf):
        """
        :param cf: configs file holding object specifications and output directories.
        """

        self.cf = cf

        self.n_train, self.n_test = cf.n_train_samples, cf.n_test_samples
        self.sample_size = cf.pre_crop_size
        self.dim = len(self.sample_size)
        self.class_radii = np.array([label.radius for label in self.cf.pp_classes if label.id!=0])
        self.class_id2label = {label.id: label for label in self.cf.pp_classes}

        self.mp_args = []
        # count sample ids consecutively over train, test splits within on dataset (one shape kind)
        self.last_s_id = 0
        for split in ["train", "test"]:
            self.set_splits_info(split)

    def set_splits_info(self, split):
        """ Set info for data set splits, i.e., directory and nr of samples.
        :param split: name of split, in {"train", "test"}.
        """
        out_dir = os.path.join(self.cf.pp_rootdir, split)
        os.makedirs(out_dir, exist_ok=True)

        n_samples = self.n_train if "train" in split else self.n_test
        req_exact_gt = "test" in split

        self.mp_args += [[out_dir, self.last_s_id+running_id, req_exact_gt] for running_id in range(n_samples)]
        self.last_s_id+= n_samples

    def generate_sample_radii(self, class_ids, shapes):

        # the radii set in labels are ranges to sample from in the form [(min_x,min_y,min_z), (max_x,max_y,max_z)]
        all_radii = []
        for ix, cl_radii in enumerate([self.class_radii[cl_id - 1].transpose() for cl_id in class_ids]):
            if "cylinder" in shapes[ix] or "block" in shapes[ix]:
                # maintain 2D aspect ratio
                sample_radii = [np.random.uniform(*cl_radii[0])] * 2
                assert len(sample_radii) == 2, "upper sr {}, cl_radii {}".format(sample_radii, cl_radii)
                if self.cf.pp_place_radii_mid_bin:
                    bef_conv_r = np.copy(sample_radii)
                    bin_id =  self.cf.rg_val_to_bin_id(bef_conv_r)
                    assert np.isscalar(bin_id)
                    sample_radii = self.cf.bin_id2rg_val[bin_id]*2
                    assert len(sample_radii) == 2, "mid before sr {}, sr {}, rgv2bid {}, cl_radii {},  bid2rgval {}".format(bef_conv_r, sample_radii, bin_id, cl_radii,
                                                                                                             self.cf.bin_id2rg_val[bin_id])
            else:
                raise NotImplementedError("requested object shape {}".format(shapes[ix]))
            if self.dim == 3:
                assert len(sample_radii) == 2, "lower sr {}, cl_radii {}".format(sample_radii, cl_radii)
                #sample_radii += [np.random.uniform(*cl_radii[2])]
                sample_radii = np.concatenate((sample_radii, np.random.uniform(*cl_radii[2], size=1)))
            all_radii.append(sample_radii)

        return all_radii

    def apply_gt_distort(self, class_id, radii, radii_divs, outer_min_radii=None, outer_max_radii=None):
        """ Apply a distortion to the ground truth (gt). This is motivated by investigating the effects of noisy labels.
            GTs that can be distorted are the object radii and ensuing GT quantities like segmentation and regression
            targets.
        :param class_id: class id of object.
        :param radii: radii of object. This is in the abstract sense, s.t. for a block-shaped object radii give the side
            lengths.
        :param radii_divs: radii divisors, i.e., fractions to take from radii to get inner radii of hole-shaped objects,
            like a torus.
        :param outer_min_radii: min radii assignable when distorting gt.
        :param outer_max_radii: max radii assignable when distorting gt.
        :return:
        """
        applied_gt_distort = False
        for ambig in self.class_id2label[class_id].gt_distortion:
            if self.cf.ambiguities[ambig][0] > np.random.rand():
                if ambig == "outer_radius":
                    radii = radii * abs(np.random.normal(1., self.cf.ambiguities["outer_radius"][1]))
                    applied_gt_distort = True
                if ambig == "radii_relations":
                    radii = radii * abs(np.random.normal(1.,self.cf.ambiguities["radii_relations"][1],size=len(radii)))
                    applied_gt_distort = True
                if ambig == "inner_radius":
                    radii_divs = radii_divs * abs(np.random.normal(1., self.cf.ambiguities["inner_radius"][1]))
                    applied_gt_distort = True
                if ambig == "radius_calib":
                    if self.cf.ambigs_sampling=="uniform":
                        radii = abs(np.random.uniform(outer_min_radii, outer_max_radii))
                    elif self.cf.ambigs_sampling=="gaussian":
                        distort = abs(np.random.normal(1, scale=self.cf.ambiguities["radius_calib"][1], size=None))
                        assert len(radii) == self.dim, "radii {}".format(radii)
                        radii *= [distort, distort, 1.] if self.cf.pp_only_distort_2d else distort
                    applied_gt_distort = True
        return radii, radii_divs, applied_gt_distort

    def draw_object(self, img, seg, undistorted_seg, ics, regress_targets, undistorted_rg_targets, applied_gt_distort,
                                 roi_ix, class_id, shape, radii, center):
        """ Draw a single object into the given image and add it to the corresponding ground truths.
        :param img: image (volume) to hold the object.
        :param seg: pixel-wise labelling of the image, possibly distorted if gt distortions are applied.
        :param undistorted_seg: certainly undistorted, i.e., exact segmentation of object.
        :param ics: indices which mark the positions within the image.
        :param regress_targets: regression targets (e.g., 2D radii of object), evtly distorted.
        :param undistorted_rg_targets: undistorted regression targets.
        :param applied_gt_distort: boolean, whether or not gt distortion was applied.
        :param roi_ix: running index of object in whole image.
        :param class_id: class id of object.
        :param shape: shape of object (e.g., whether to draw a cylinder, or block, or ...).
        :param radii: radii of object (in an abstract sense, i.e., radii are side lengths in case of block shape).
        :param center: center of object in image coordinates.
        :return: img, seg, undistorted_seg, regress_targets, undistorted_rg_targets, applied_gt_distort, which are now
            extended are amended to reflect the new object.
        """

        radii_blur = hasattr(self.cf, "ambiguities") and hasattr(self.class_id2label[class_id],
                                                                 "gt_distortion") and 'radius_calib' in \
                     self.class_id2label[class_id].gt_distortion

        if radii_blur:
            blur_width = self.cf.ambiguities['radius_calib'][1]
            if self.cf.ambigs_sampling == "uniform":
                blur_width *= np.sqrt(12)
            if self.cf.pp_only_distort_2d:
                outer_max_radii = np.concatenate((radii[:2] + blur_width * radii[:2], [radii[2]]))
                outer_min_radii = np.concatenate((radii[:2] - blur_width * radii[:2], [radii[2]]))
                #print("belt width ", outer_max_radii - outer_min_radii)
            else:
                outer_max_radii = radii + blur_width * radii
                outer_min_radii = radii - blur_width * radii
        else:
            outer_max_radii, outer_min_radii = radii, radii

        if "ellipsoid" in shape or "torus" in shape:
            # sphere equation: (x-h)**2 + (y-k)**2 - (z-l)**2 = r**2
            # ellipsoid equation: ((x-h)/a)**2+((y-k)/b)**2+((z-l)/c)**2 <= 1; a, b, c the "radii"/ half-length of principal axes
            obj = ((ics - center) / radii) ** 2
        elif "diamond" in shape:
            # diamond equation: (|x-h|)/a+(|y-k|)/b+(|z-l|)/c <= 1
            obj = abs(ics - center) / radii
        elif "cylinder" in shape:
            # cylinder equation:((x-h)/a)**2 + ((y-k)/b)**2 <= 1 while |z-l| <= c
            obj = ((ics - center).astype("float64") / radii) ** 2
            # set z values s.t. z slices outside range are sorted out
            obj[:, -1] = np.where(abs((ics - center)[:, -1]) <= radii[2], 0., 1.1)
            if radii_blur:
                inner_obj = ((ics - center).astype("float64") / outer_min_radii) ** 2
                inner_obj[:, -1] = np.where(abs((ics - center)[:, -1]) <= outer_min_radii[2], 0., 1.1)
                outer_obj = ((ics - center).astype("float64") / outer_max_radii) ** 2
                outer_obj[:, -1] = np.where(abs((ics - center)[:, -1]) <= outer_max_radii[2], 0., 1.1)
                # radial dists: sqrt( (x-h)**2 + (y-k)**2 + (z-l)**2 )
                obj_radial_dists = np.sqrt(np.sum((ics - center).astype("float64")**2, axis=1))
        elif "block" in shape:
            # block equation: (|x-h|)/a+(|y-k|)/b <= 1 while  |z-l| <= c
            obj = abs(ics - center) / radii
            obj[:, -1] = np.where(abs((ics - center)[:, -1]) <= radii[2], 0., 1.1)
            if radii_blur:
                inner_obj = abs(ics - center) / outer_min_radii
                inner_obj[:, -1] = np.where(abs((ics - center)[:, -1]) <= outer_min_radii[2], 0., 1.1)
                outer_obj = abs(ics - center) / outer_max_radii
                outer_obj[:, -1] = np.where(abs((ics - center)[:, -1]) <= outer_max_radii[2], 0., 1.1)
                obj_radial_dists = np.sum(abs(ics - center), axis=1).astype("float64")
        else:
            raise Exception("Invalid object shape '{}'".format(shape))

        # create the "original" GT, i.e., the actually true object and draw it into undistorted seg.
        obj = (np.sum(obj, axis=1) <= 1)
        obj = obj.reshape(seg[0].shape)
        slices_to_discard = np.where(np.count_nonzero(np.count_nonzero(obj, axis=0), axis=0) <= self.cf.min_2d_radius)[0]
        obj[..., slices_to_discard] = 0
        undistorted_radii = np.copy(radii)
        undistorted_seg[class_id][obj] = roi_ix + 1
        obj = obj.astype('float64')

        if radii_blur:
            inner_obj = np.sum(inner_obj, axis=1) <= 1
            outer_obj = (np.sum(outer_obj, axis=1) <= 1) & ~inner_obj
            obj_radial_dists[outer_obj] = obj_radial_dists[outer_obj] / max(obj_radial_dists[outer_obj])
            intensity_slope = self.cf.pp_blur_min_intensity - 1.
            # intensity(r) = (i(r_max)-i(0))/r_max * r + i(0), where i(0)==1.
            obj_radial_dists[outer_obj] = obj_radial_dists[outer_obj] * intensity_slope + 1.
            inner_obj = inner_obj.astype('float64')
            #outer_obj, obj_radial_dists = outer_obj.reshape(seg[0].shape), obj_radial_dists.reshape(seg[0].shape)
            inner_obj += np.where(outer_obj, obj_radial_dists, 0.)
            obj = inner_obj.reshape(seg[0].shape)
        if not np.any(obj):
            print("An object was completely discarded due to min 2d radius requirement, discarded slices: {}.".format(
                slices_to_discard))
        # draw the evtly blurred obj into image.
        img += obj * (class_id + 1.)

        if hasattr(self.cf, "ambiguities") and hasattr(self.class_id2label[class_id], "gt_distortion"):
            radii_divs = [None]  # dummy since not implemented yet
            radii, radii_divs, applied_gt_distort = self.apply_gt_distort(class_id, radii, radii_divs,
                                                                          outer_min_radii, outer_max_radii)
            if applied_gt_distort:
                if "ellipsoid" in shape or "torus" in shape:
                    obj = ((ics - center) / radii) ** 2
                elif 'diamond' in shape:
                    obj = abs(ics - center) / radii
                elif "cylinder" in shape:
                    obj = ((ics - center) / radii) ** 2
                    obj[:, -1] = np.where(abs((ics - center)[:, -1]) <= radii[2], 0., 1.1)
                elif "block" in shape:
                    obj = abs(ics - center) / radii
                    obj[:, -1] = np.where(abs((ics - center)[:, -1]) <= radii[2], 0., 1.1)
                obj = (np.sum(obj, axis=1) <= 1).reshape(seg[0].shape)
                obj[..., slices_to_discard] = False

        if self.class_id2label[class_id].regression == "radii":
            regress_targets.append(radii)
            undistorted_rg_targets.append(undistorted_radii)
        elif self.class_id2label[class_id].regression == "radii_2d":
            regress_targets.append(radii[:2])
            undistorted_rg_targets.append(undistorted_radii[:2])
        elif self.class_id2label[class_id].regression == "radius_2d":
            regress_targets.append(radii[:1])
            undistorted_rg_targets.append(undistorted_radii[:1])
        else:
            regress_targets.append(self.class_id2label[class_id].regression)
            undistorted_rg_targets.append(self.class_id2label[class_id].regression)

        seg[class_id][obj.astype('bool')] = roi_ix + 1

        return  img, seg, undistorted_seg, regress_targets, undistorted_rg_targets, applied_gt_distort

    def create_sample(self, args):
        """ Create a single sample and save to file. One sample is one image (volume) containing none, one, or multiple
            objects.
        :param args: out_dir: directory where to save sample, s_id: id of the sample.
        :return: specs that identify this single created image
        """
        out_dir, s_id, req_exact_gt = args

        print('processing {} {}'.format(out_dir, s_id))
        img = np.random.normal(loc=0.0, scale=self.cf.noise_scale, size=self.sample_size)
        img[img<0.] = 0.
        # one-hot-encoded seg
        seg = np.zeros((self.cf.num_classes+1, *self.sample_size)).astype('uint8')
        undistorted_seg = np.copy(seg)
        applied_gt_distort = False

        if hasattr(self.cf, "pp_empty_samples_ratio") and self.cf.pp_empty_samples_ratio >= np.random.rand():
            # generate fully empty sample
            class_ids, regress_targets, undistorted_rg_targets = [], [], []
        else:
            class_choices = np.repeat(np.arange(1, self.cf.num_classes+1), self.cf.max_instances_per_class)
            n_insts = np.random.randint(1, self.cf.max_instances_per_sample + 1)
            class_ids = np.random.choice(class_choices, size=n_insts, replace=False)
            shapes = np.array([self.class_id2label[cl_id].shape for cl_id in class_ids])
            all_radii = self.generate_sample_radii(class_ids, shapes)

            # reorder s.t. larger objects are drawn first (in order to not fully cover smaller objects)
            order = np.argsort(-1*np.prod(all_radii,axis=1))
            class_ids = class_ids[order]; all_radii = np.array(all_radii)[order]; shapes = shapes[order]

            regress_targets, undistorted_rg_targets = [], []
            # indices ics equal positions within img/volume
            ics = np.argwhere(np.ones(seg[0].shape))
            for roi_ix, class_id in enumerate(class_ids):
                radii = all_radii[roi_ix]
                # enforce distance between object center and image edge relative to radii.
                margin_r_divisor = (2, 2, 4)
                center = [np.random.randint(radii[dim] / margin_r_divisor[dim], img.shape[dim] -
                                            radii[dim] / margin_r_divisor[dim]) for dim in range(len(img.shape))]

                img, seg, undistorted_seg, regress_targets, undistorted_rg_targets, applied_gt_distort = \
                    self.draw_object(img, seg, undistorted_seg, ics, regress_targets, undistorted_rg_targets, applied_gt_distort,
                                 roi_ix, class_id, shapes[roi_ix], radii, center)

        fg_slices = np.where(np.sum(np.sum(np.sum(seg,axis=0), axis=0), axis=0))[0]
        if self.cf.pp_create_ohe_seg:
            img = img[np.newaxis]
        else:
            # choosing rois to keep by smaller radius==higher prio needs to be ensured during roi generation,
            # smaller objects need to be drawn later (==higher roi id)
            seg = seg.max(axis=0)
            seg_ids = np.unique(seg)
            if len(seg_ids) != len(class_ids) + 1:
                # in this case an object was completely covered by a succeeding object
                print("skipping corrupt sample")
                print("seg ids {}, class_ids {}".format(seg_ids, class_ids))
                return None
            if not applied_gt_distort:
                assert np.all(np.flatnonzero(img>0) == np.flatnonzero(seg>0))
                assert np.all(np.array(regress_targets).flatten()==np.array(undistorted_rg_targets).flatten())

        # save the img
        out_path = os.path.join(out_dir, '{}.npy'.format(s_id))
        np.save(out_path, img.astype('float16'))

        # exact GT
        if req_exact_gt:
            if not self.cf.pp_create_ohe_seg:
                undistorted_seg = undistorted_seg.max(axis=0)
            np.save(os.path.join(out_dir, '{}_exact_seg.npy'.format(s_id)), undistorted_seg)
        else:
            # if hasattr(self.cf, 'ambiguities') and \
            #     np.any([hasattr(label, "gt_distortion") and len(label.gt_distortion)>0 for label in self.class_id2label.values()]):
            # save (evtly) distorted GT
            np.save(os.path.join(out_dir, '{}_seg.npy'.format(s_id)), seg)


        return [out_dir, out_path, class_ids, regress_targets, fg_slices, undistorted_rg_targets, str(s_id)]

    def create_sets(self, processes=os.cpu_count()):
        """ Create whole training and test set, save to files under given directory cf.out_dir.
        :param processes: nr of parallel processes.
        """


        print('starting creation of {} images.'.format(len(self.mp_args)))
        shutil.copyfile("configs.py", os.path.join(self.cf.pp_rootdir, 'applied_configs.py'))
        pool = Pool(processes=processes)
        try:
            imgs_info = pool.map(self.create_sample, self.mp_args)
        except AttributeError as e:
            raise AttributeError("{}\nAre configs tasks = ['class', 'regression'] (both)?".format(e))
        imgs_info = [img for img in imgs_info if img is not None]
        pool.close()
        pool.join()
        print("created a total of {} samples.".format(len(imgs_info)))

        self.df = pd.DataFrame.from_records(imgs_info, columns=['out_dir', 'path', 'class_ids', 'regression_vectors',
                                                                'fg_slices', 'undistorted_rg_vectors', 'pid'])

        for out_dir, group_df in self.df.groupby("out_dir"):
            group_df.to_pickle(os.path.join(out_dir, 'info_df.pickle'))


    def convert_copy_npz(self):
        """ Convert a copy of generated .npy-files to npz and save in .npz-directory given in configs.
        """
        if hasattr(self.cf, "pp_npz_dir") and self.cf.pp_npz_dir:
            for out_dir, group_df in self.df.groupby("out_dir"):
                rel_dir = os.path.relpath(out_dir, self.cf.pp_rootdir).split(os.sep)
                npz_out_dir = os.path.join(self.cf.pp_npz_dir, str(os.sep).join(rel_dir))
                print("npz out dir: ", npz_out_dir)
                os.makedirs(npz_out_dir, exist_ok=True)
                group_df.to_pickle(os.path.join(npz_out_dir, 'info_df.pickle'))
                dmanager.pack_dataset(out_dir, npz_out_dir, recursive=True, verbose=False)
        else:
            print("Did not convert .npy-files to .npz because npz directory not set in configs.")


if __name__ == '__main__':
    import configs as cf
    cf = cf.Configs()
    total_stime = time.time()

    toy_gen = ToyGenerator(cf)
    toy_gen.create_sets()
    toy_gen.convert_copy_npz()


    mins, secs = divmod((time.time() - total_stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))
