import sys
sys.path.append('../') #works on cluster indep from where sbatch job is started
import plotting as plg

import warnings
import os
import time
import pickle


import numpy as np
import pandas as pd
from PIL import Image as pil

import torch
import torch.utils.data

# batch generator tools from https://github.com/MIC-DKFZ/batchgenerators
from batchgenerators.transforms.spatial_transforms import MirrorTransform as Mirror
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform
from batchgenerators.transforms.color_transforms import GammaTransform
#from batchgenerators.transforms.utility_transforms import ConvertSegToBoundingBoxCoordinates


sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import utils.exp_utils as utils
import utils.dataloader_utils as dutils
from utils.dataloader_utils import ConvertSegToBoundingBoxCoordinates

from configs import Configs
cf= configs()


warnings.filterwarnings("ignore", message="This figure includes Axes.*")


def load_obj(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

def save_to_npy(arr_out, array):
    np.save(arr_out+".npy", array) 
    print("Saved binary .npy-file to {}".format(arr_out))
    return arr_out+".npy"

def shape_small_first(shape):
    if len(shape)<=2: #no changing dimensions if channel-dim is missing
        return shape
    smallest_dim = np.argmin(shape)
    if smallest_dim!=0: #assume that smallest dim is color channel
        new_shape = np.array(shape) #to support mask indexing
        new_shape = (new_shape[smallest_dim],
                    *new_shape[(np.arange(len(shape),dtype=int)!=smallest_dim)])
        return new_shape
    else:
        return shape
       
class Dataset(dutils.Dataset):
    def __init__(self,  cf, logger=None, subset_ids=None, data_sourcedir=None):
        super(Dataset, self).__init__(cf, data_sourcedir=data_sourcedir)

        info_dict = load_obj(cf.info_dict_path)

        if subset_ids is not None:
            img_ids = subset_ids
            if logger is None:
                print('subset: selected {} instances from df'.format(len(pids)))
            else:
                logger.info('subset: selected {} instances from df'.format(len(pids)))
        else:
            img_ids = list(info_dict.keys())

        #evtly copy data from data_rootdir to data_dir
        if cf.server_env and not hasattr(cf, "data_dir"):
            file_subset = [info_dict[img_id]['img'][:-3]+"*" for img_id in img_ids]
            file_subset+= [info_dict[img_id]['seg'][:-3]+"*" for img_id in img_ids]
            file_subset+= [cf.info_dict_path]
            self.copy_data(cf, file_subset=file_subset)
            cf.data_dir = self.data_dir

        img_paths = [os.path.join(self.data_dir, info_dict[img_id]['img']) for img_id in img_ids]
        seg_paths = [os.path.join(self.data_dir, info_dict[img_id]['seg']) for img_id in img_ids]

        # load all subject files
        self.data = {}
        for i, img_id in enumerate(img_ids):
            subj_data = {'img_id':img_id}
            subj_data['img'] = img_paths[i]
            subj_data['seg'] = seg_paths[i]
            if 'class' in self.cf.prediction_tasks:
                subj_data['class_targets'] = np.array(info_dict[img_id]['roi_classes'])
            else:
                subj_data['class_targets'] = np.ones_like(np.array(info_dict[img_id]['roi_classes']))
    
            self.data[img_id] = subj_data

        cf.roi_items = cf.observables_rois[:]
        cf.roi_items += ['class_targets']
        if 'regression' in cf.prediction_tasks:
            cf.roi_items += ['regression_targets']

        self.set_ids = list(self.data.keys())
        
        self.df = None

class BatchGenerator(dutils.BatchGenerator):
    """
    create the training/validation batch generator. Randomly sample batch_size patients
    from the data set, (draw a random slice if 2D), pad-crop them to equal sizes and merge to an array.
    :param data: data dictionary as provided by 'load_dataset'
    :param img_modalities: list of strings ['adc', 'b1500'] from config
    :param batch_size: number of patients to sample for the batch
    :param pre_crop_size: equal size for merging the patients to a single array (before the final random-crop in data aug.)
    :return dictionary containing the batch data / seg / pids as lists; the augmenter will later concatenate them into an array.
    """
    def __init__(self, cf, data, n_batches=None, sample_pids_w_replace=True):
        super(BatchGenerator, self).__init__(cf, data, n_batches)
        self.dataset_length = len(self._data)
        self.cf = cf

        self.sample_pids_w_replace = sample_pids_w_replace
        self.eligible_pids = list(self._data.keys())

        self.chans = cf.channels if cf.channels is not None else np.index_exp[:]
        assert hasattr(self.chans, "__iter__"), "self.chans has to be list-like to maintain dims when slicing"

        self.p_fg = 0.5
        self.empty_samples_max_ratio = 0.33
        self.random_count = int(cf.batch_random_ratio * cf.batch_size)

        self.balance_target_distribution(plot=sample_pids_w_replace)
        self.stats = {"roi_counts" : np.zeros((len(self.unique_ts),), dtype='uint32'), "empty_samples_count" : 0}

    def generate_train_batch(self):
        #everything done in here is per batch
        #print statements in here get confusing due to multithreading
        if self.sample_pids_w_replace:
            # fully random patients
            batch_patient_ids = list(np.random.choice(self.dataset_pids, size=self.random_count, replace=False))
            # target-balanced patients
            batch_patient_ids += list(np.random.choice(
                self.dataset_pids, size=self.batch_size - self.random_count, replace=False, p=self.p_probs))
        else:
            batch_patient_ids = np.random.choice(self.eligible_pids, size=self.batch_size, replace=False)
        if self.sample_pids_w_replace == False:
            self.eligible_pids = [pid for pid in self.eligible_pids if pid not in batch_patient_ids]
            if len(self.eligible_pids) < self.batch_size:
                self.eligible_pids = self.dataset_pids
        
        batch_data, batch_segs, batch_class_targets = [], [], []
        # record roi count of classes in batch
        batch_roi_counts, empty_samples_count = np.zeros((self.cf.num_classes,), dtype='uint32'), 0

        for sample in range(self.batch_size):

            patient = self._data[batch_patient_ids[sample]]
            
            data = np.load(patient["img"], mmap_mode="r")
            seg = np.load(patient['seg'], mmap_mode="r")
            
            (c,y,x) = data.shape
            spatial_shp = data[0].shape
            assert spatial_shp==seg.shape, "spatial shape incongruence betw. data {} and seg {}".format(spatial_shp, seg.shape)

            if np.any([spatial_shp[ix] < self.cf.pre_crop_size[ix] for ix in range(len(spatial_shp))]):
                new_shape = [np.max([spatial_shp[ix], self.cf.pre_crop_size[ix]]) for ix in range(len(spatial_shp))]
                data = dutils.pad_nd_image(data, (len(data), *new_shape))
                seg = dutils.pad_nd_image(seg, new_shape)
            
            #eventual cropping to pre_crop_size: with prob self.p_fg sample pixel from random ROI and shift center,
            #if possible, to that pixel, so that img still contains ROI after pre-cropping
            dim_cropflags = [spatial_shp[i] > self.cf.pre_crop_size[i] for i in range(len(spatial_shp))]
            if np.any(dim_cropflags):
                #sample crop center regardless of ROIs, not guaranteed to be empty
                def get_cropped_centercoords(dim):                        
                    return np.random.randint(low=self.cf.pre_crop_size[dim]//2,
                                             high=spatial_shp[dim] - self.cf.pre_crop_size[dim]//2)
                    
                sample_seg_center = {}
                for dim in np.where(dim_cropflags)[0]:
                    sample_seg_center[dim] = get_cropped_centercoords(dim)
                    min_ = int(sample_seg_center[dim] - self.cf.pre_crop_size[dim]//2)
                    max_ = int(sample_seg_center[dim] + self.cf.pre_crop_size[dim]//2)
                    data = np.take(data, indices=range(min_, max_), axis=dim+1) #+1 for channeldim
                    seg = np.take(seg, indices=range(min_, max_), axis=dim)
                    
            batch_data.append(data)
            batch_segs.append(seg[np.newaxis])
                
            batch_class_targets.append(patient['class_targets'])

            for cl in range(self.cf.num_classes):
                batch_roi_counts[cl] += np.count_nonzero(patient['class_targets'][np.unique(seg[seg>0]) - 1] == cl)
            if not np.any(seg):
                empty_samples_count += 1
        
        batch = {'data': np.array(batch_data).astype('float32'), 'seg': np.array(batch_segs).astype('uint8'),
                 'pid': batch_patient_ids, 'class_targets': np.array(batch_class_targets),
                 'roi_counts': batch_roi_counts, 'empty_samples_count': empty_samples_count}
        return batch

class PatientBatchIterator(dutils.PatientBatchIterator):
    """
    creates a val/test generator. Step through the dataset and return dictionaries per patient.
    For Patching, shifts all patches into batch dimension. batch_tiling_forward will take care of exceeding batch dimensions.

    This iterator/these batches are not intended to go through MTaugmenter afterwards
    """

    def __init__(self, cf, data):
        super(PatientBatchIterator, self).__init__(cf, data)

        self.patch_size = cf.patch_size

        self.patient_ix = 0  # running index over all patients in set

    def generate_train_batch(self, pid=None):

        if self.patient_ix == len(self.dataset_pids):
            self.patient_ix = 0
        if pid is None:
            pid = self.dataset_pids[self.patient_ix]  # + self.thread_id
        patient = self._data[pid]
        batch_class_targets = np.array([patient['class_targets']])

        data = np.load(patient["img"], mmap_mode="r")[np.newaxis]
        seg = np.load(patient['seg'], mmap_mode="r")[np.newaxis, np.newaxis]
        (b, c, y, x) = data.shape
        spatial_shp = data.shape[2:]
        assert spatial_shp == seg.shape[2:], "spatial shape incongruence betw. data {} and seg {}".format(spatial_shp,
                                                                                                      seg.shape)
        if np.any([spatial_shp[ix] < self.cf.pre_crop_size[ix] for ix in range(len(spatial_shp))]):
            new_shape = [np.max([spatial_shp[ix], self.cf.pre_crop_size[ix]]) for ix in range(len(spatial_shp))]
            data = dutils.pad_nd_image(data, (len(data), *new_shape))
            seg = dutils.pad_nd_image(seg, new_shape)

        batch = {'data': data, 'seg': seg, 'class_targets': batch_class_targets}
        converter = ConvertSegToBoundingBoxCoordinates(self.cf.dim, self.cf.roi_items, False, self.cf.class_specific_seg)
        batch = converter(**batch)
        batch.update({'patient_bb_target': batch['bb_target'],
                      'patient_class_targets': batch['class_targets'],
                      'original_img_shape': data.shape,
                      'pid': np.array([pid] * len(data))})

        # eventual tiling into patches
        spatial_shp = batch["data"].shape[2:]
        if np.any([spatial_shp[ix] > self.patch_size[ix] for ix in range(len(spatial_shp))]):
            patient_batch = batch
            print("patientiterator produced patched batch!")
            patch_crop_coords_list = dutils.get_patch_crop_coords(data[0], self.patch_size)
            new_img_batch, new_seg_batch = [], []

            for c in patch_crop_coords_list:
                new_img_batch.append(data[:, c[0]:c[1], c[2]:c[3]])
                seg_patch = seg[:, c[0]:c[1], c[2]: c[3]]
                new_seg_batch.append(seg_patch)

            shps = []
            for arr in new_img_batch:
                shps.append(arr.shape)

            data = np.array(new_img_batch)  # (patches, c, x, y, z)
            seg = np.array(new_seg_batch)
            batch_class_targets = np.repeat(batch_class_targets, len(patch_crop_coords_list), axis=0)

            patch_batch = {'data': data.astype('float32'), 'seg': seg.astype('uint8'),
                           'class_targets': batch_class_targets,
                           'pid': np.array([pid] * data.shape[0])}
            patch_batch['patch_crop_coords'] = np.array(patch_crop_coords_list)
            patch_batch['patient_bb_target'] = patient_batch['patient_bb_target']
            patch_batch['patient_class_targets'] = patient_batch['patient_class_targets']
            patch_batch['patient_data'] = patient_batch['data']
            patch_batch['patient_seg'] = patient_batch['seg']
            patch_batch['original_img_shape'] = patient_batch['original_img_shape']

            converter = ConvertSegToBoundingBoxCoordinates(self.cf.dim, self.cf.roi_items, False, self.cf.class_specific_seg)
            patch_batch = converter(**patch_batch)
            batch = patch_batch

        self.patient_ix += 1
        if self.patient_ix == len(self.dataset_pids):
            self.patient_ix = 0

        return batch

def create_data_gen_pipeline(cf, patient_data, do_aug=True, sample_pids_w_replace=True):
    """
    create mutli-threaded train/val/test batch generation and augmentation pipeline.
    :param patient_data: dictionary containing one dictionary per patient in the train/test subset
    :param test_pids: (optional) list of test patient ids, calls the test generator.
    :param do_aug: (optional) whether to perform data augmentation (training) or not (validation/testing)
    :return: multithreaded_generator
    """
    data_gen = BatchGenerator(cf, patient_data, sample_pids_w_replace=sample_pids_w_replace)

    my_transforms = []
    if do_aug:
        if cf.da_kwargs["mirror"]:
            mirror_transform = Mirror(axes=cf.da_kwargs['mirror_axes'])
            my_transforms.append(mirror_transform)
        spatial_transform = SpatialTransform(patch_size=cf.patch_size[:cf.dim],
                                         patch_center_dist_from_border=cf.da_kwargs['rand_crop_dist'][:2],
                                         do_elastic_deform=cf.da_kwargs['do_elastic_deform'],
                                         alpha=cf.da_kwargs['alpha'], sigma=cf.da_kwargs['sigma'],
                                         do_rotation=cf.da_kwargs['do_rotation'], angle_x=cf.da_kwargs['angle_x'],
                                         angle_y=cf.da_kwargs['angle_y'], angle_z=cf.da_kwargs['angle_z'],
                                         do_scale=cf.da_kwargs['do_scale'], scale=cf.da_kwargs['scale'],
                                         random_crop=cf.da_kwargs['random_crop'],
                                         border_mode_data=cf.da_kwargs['border_mode_data'])
        my_transforms.append(spatial_transform)
        gamma_transform = GammaTransform(gamma_range=cf.da_kwargs["gamma_range"], invert_image=False,
                                         per_channel=False, retain_stats=False)
        my_transforms.append(gamma_transform)
    
    else:
        my_transforms.append(CenterCropTransform(crop_size=cf.patch_size[:cf.dim]))

    if cf.create_bounding_box_targets:
        my_transforms.append(ConvertSegToBoundingBoxCoordinates(cf.dim, cf.roi_items, False, cf.class_specific_seg))
        #batch receives entry 'bb_target' w bbox coordinates as [y1,x1,y2,x2,z1,z2].
    #my_transforms.append(ConvertSegToOnehotTransform(classes=range(cf.num_seg_classes)))
    all_transforms = Compose(my_transforms)
    #MTAugmenter creates iterator from data iterator data_gen after applying the composed transform all_transforms
    multithreaded_generator = MultiThreadedAugmenter(data_gen, all_transforms, num_processes=cf.n_workers,
                                                     seeds=np.random.randint(0,cf.n_workers*2,size=cf.n_workers))
    return multithreaded_generator


def get_train_generators(cf, logger, data_statistics=True):
    """
    wrapper function for creating the training batch generator pipeline. returns the train/val generators
    need to select cv folds on patient level, but be able to include both breasts of each patient.
    """
    dataset = Dataset(cf)
    dataset.init_FoldGenerator(cf.seed, cf.n_cv_splits)
    dataset.generate_splits(check_file=os.path.join(cf.exp_dir, 'fold_ids.pickle'))
    set_splits = dataset.fg.splits

    test_ids, val_ids = set_splits.pop(cf.fold), set_splits.pop(cf.fold - 1)
    train_ids = np.concatenate(set_splits, axis=0)

    if cf.hold_out_test_set:
        train_ids = np.concatenate((train_ids, test_ids), axis=0)
        test_ids = []

    train_data = {k: v for (k, v) in dataset.data.items() if k in train_ids}
    val_data = {k: v for (k, v) in dataset.data.items() if k in val_ids}

    logger.info("data set loaded with: {} train / {} val / {} test patients".format(len(train_ids), len(val_ids),
                                                                                    len(test_ids)))
    if data_statistics:
        dataset.calc_statistics(subsets={"train": train_ids, "val": val_ids, "test": test_ids},
                                plot_dir=os.path.join(cf.plot_dir, "data_stats_fold_"+str(cf.fold)))

    batch_gen = {}
    batch_gen['train'] = create_data_gen_pipeline(cf, train_data, do_aug=True)
    batch_gen[cf.val_mode] = create_data_gen_pipeline(cf, val_data, do_aug=False, sample_pids_w_replace=False)
    batch_gen['n_val'] = cf.num_val_batches if cf.num_val_batches!="all" else len(val_data)
        
    return batch_gen

def get_test_generator(cf, logger):
    """
    if get_test_generators is called multiple times in server env, every time of 
    Dataset initiation rsync will check for copying the data; this should be okay
    since rsync will not copy if files already exist in destination.
    """

    if cf.hold_out_test_set:
        sourcedir = cf.test_data_sourcedir
        test_ids = None
    else:
        sourcedir = None
        with open(os.path.join(cf.exp_dir, 'fold_ids.pickle'), 'rb') as handle:
            set_splits = pickle.load(handle)
        test_ids = set_splits[cf.fold]


    test_set = Dataset(cf, test_ids, data_sourcedir=sourcedir)
    logger.info("data set loaded with: {} test patients".format(len(test_set.set_ids)))
    batch_gen = {}
    batch_gen['test'] = PatientBatchIterator(cf, test_set.data)
    batch_gen['n_test'] = len(test_set.set_ids) if cf.max_test_patients=="all" else min(cf.max_test_patients, len(test_set.set_ids))
    
    return batch_gen   

def main():
    total_stime = time.time()
    times = {}
    
    CUDA = torch.cuda.is_available()
    print("CUDA available: ", CUDA)


    #cf.server_env = True
    #cf.data_dir = "experiments/dev_data"

    cf.exp_dir = "experiments/dev/"
    cf.plot_dir = cf.exp_dir+"plots"
    os.makedirs(cf.exp_dir, exist_ok=True)
    cf.fold = 0
    logger = utils.get_logger(cf.exp_dir)

    gens = get_train_generators(cf, logger)
    train_loader = gens['train']
    
    #for i in range(train_loader.dataset_length):
    #    print("batch", i)
    stime = time.time()
    ex_batch = next(train_loader)
   # plg.view_batch(cf, ex_batch, out_file="experiments/dev/dev_extrainbatch.png", has_colorchannels=True, isRGB=True)
    times["train_batch"] = time.time()-stime

    
    val_loader = gens['val_sampling']
    stime = time.time()
    ex_batch = next(val_loader)
    times["val_batch"] = time.time()-stime
    stime = time.time()
    plg.view_batch(cf, ex_batch, out_file="experiments/dev/dev_exvalbatch.png", has_colorchannels=True, isRGB=True, show_gt_boxes=False)
    times["val_plot"] = time.time()-stime
    
    test_loader = get_test_generator(cf, logger)["test"]
    stime = time.time()
    ex_batch = next(test_loader)
    times["test_batch"] = time.time()-stime
    #plg.view_batch(cf, ex_batch, out_file="experiments/dev/dev_expatientbatch.png", has_colorchannels=True, isRGB=True)
    
    print(ex_batch["data"].shape)


    print("Times recorded throughout:")
    for (k,v) in times.items():
        print(k, "{:.2f}".format(v))
    
    mins, secs = divmod((time.time() - total_stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs)) 
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))
    

        
if __name__=="__main__":
    start_time = time.time()
    
    main()
    
    print("Program runtime in s: ", '{:.2f}'.format(time.time()-start_time))