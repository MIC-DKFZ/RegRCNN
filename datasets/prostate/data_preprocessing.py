__author__ = "Simon Kohl, Gregor Ramien"


# subject-wise extractor that does not depend on Prisma/Radval and that checks for geometry miss-alignments
# (corrects them if applicable), images and masks should be stored separately, each in its own memmap
# at run-time, the data-loaders will assemble dicts using the histo csvs
import os
import sys
from multiprocessing import Pool
import warnings
import time
import shutil

import pandas as pd
import numpy as np
import pickle

import SimpleITK as sitk
from scipy.ndimage.measurements import center_of_mass

sys.path.append("../")
import plotting as plg
import data_manager as dmanager

def save_obj(obj, name):
    """Pickle a python object."""
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_array(path):
    """Load an image as a numpy array."""
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img)

def id_to_spec(id, base_spec):
    """Construct subject specifier from base string and an integer subject number."""
    num_zeros = 5 - len(str(id))
    assert num_zeros>=0, "id_to_spec: patient id too long to fit into 5 figures"
    return base_spec + '_' + ('').join(['0'] * num_zeros) + str(id)

def spec_to_id(spec):
    """Get subject id from string"""
    return int(spec[-5:])

def has_equal_geometry(img1, img2, precision=0.001):
    """Check whether geometries of 2 images match within a given precision."""
    equal = True

    # assert equal image extentions
    delta = [abs((img1.GetSize()[i] - img2.GetSize()[i])) < precision for i in range(3)]
    if not np.all(delta):
        equal = False

    # assert equal origins
    delta = [abs((img1.GetOrigin()[i] - img2.GetOrigin()[i])) < precision for i in range(3)]
    if not np.all(delta):
        equal = False

    # assert equal spacings
    delta = [abs((img1.GetSpacing()[i] - img2.GetSpacing()[i])) < precision for i in range(3)]
    if not np.all(delta):
        equal = False

    return equal

def resample_to_reference(ref_img, img, interpolation):
    """
    Resample an sitk image to a reference image, the size, spacing,
    origin and direction of the reference image will be used
    :param ref_img:
    :param img:
    :param interpolation:
    :return: interpolated SITK image
    """
    if interpolation == 'nearest':
        interpolator = sitk.sitkNearestNeighbor #these are just integers
    elif interpolation == 'linear':
        interpolator = sitk.sitkLinear
    elif interpolation == 'bspline':
        # basis spline of order 3
        interpolator = sitk.sitkBSpline
    else:
        raise NotImplementedError('Interpolation of type {} not implemented!'.format(interpolation))

    img = sitk.Cast(img, sitk.sitkFloat64)

    rif = sitk.ResampleImageFilter()
    # set the output size, origin, spacing and direction to that of the provided image
    rif.SetReferenceImage(ref_img) 
    rif.SetInterpolator(interpolator)

    return rif.Execute(img)

def rescale(img, scaling, interpolation=sitk.sitkBSpline, out_fpath=None):
    """
    :param scaling: tuple (z_scale, y_scale, x_scale) of scaling factors
    :param out_fpath: filepath (incl filename), if set will write .nrrd (uncompressed)
        to that location
    
    sitk/nrrd images spacing: imgs are treated as physical objects. When resampling,
    a given image is re-evaluated (resampled) at given gridpoints, the physical 
    properties of the image don't change. Hence, if the resampling-grid has a smaller
    spacing than the original image(grid), the image is sampled more often than before.
    Since every sampling produces one pixel, the resampled image will have more pixels
    (when sampled at undefined points of the image grid, the sample values will be
    interpolated). I.e., for an upsampling of an image, we need to set a smaller
    spacing for the resampling grid and a larger (pixel)size for the resampled image.
    """
    (z,y,x) = scaling
    
    old_size = np.array(img.GetSize())
    old_spacing = np.array(img.GetSpacing())
    

    new_size = (int(old_size[0]*x), int(old_size[1]*y), int(old_size[2]*z))
    new_spacing = old_spacing * (old_size/ new_size)
    
    rif = sitk.ResampleImageFilter()
    
    rif.SetReferenceImage(img)
    rif.SetInterpolator(interpolation)
    rif.SetOutputSpacing(new_spacing)
    rif.SetSize(new_size)
    
    new_img = rif.Execute(img)
    
    if not out_fpath is None:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(out_fpath)
        writer.SetUseCompression(True)
        writer.Execute(new_img)
    
    return new_img

def get_valid_z_range(arr):
    """
    check which z-slices of an image array aren't constant
    :param arr:
    :return: min and max valid slice found; under the assumption that invalid 
        slices occur never inbetween valid slices
    """

    valid_z_slices = []
    for z in range(arr.shape[0]):
        if np.var(arr[z]) != 0:
            valid_z_slices.append(z)
    return valid_z_slices[0], valid_z_slices[-1] 

def convert_to_arrays(data):
    """convert to numpy arrays.
        sitk.Images have shape (x,y,z), but GetArrayFromImage returns shape (z,y,x)
    """
    for mod in data['img'].keys():
        data['img'][mod] = sitk.GetArrayFromImage(data['img'][mod]).astype(np.float32)

    for mask in data['anatomical_masks'].keys():
        data['anatomical_masks'][mask] = sitk.GetArrayFromImage(data['anatomical_masks'][mask]).astype(np.uint8)

    for mask in data['lesions'].keys():
        data['lesions'][mask] = sitk.GetArrayFromImage(data['lesions'][mask]).astype(np.uint8)
    return data

def merge_crossmod_masks(data, rename_tags, mode="union"):
    """if data has multiple ground truths (e.g. after registration), merge 
        masks by mode. class labels (leason gleason) are assumed to be naturally registered (no ambiguity)
    :param rename_tags: usually from prepro_cf['rename_tags']
    :param mode: 'union' or name of mod ('adc', 't2') to consider only one gt
    """   

    if 'adc' in data['img'].keys() and 't2' in data['img'].keys():
        if mode=='union':
            #print("Merging gts of T2, ADC mods. Assuming data is registered!")
            tags = list(data["anatomical_masks"].keys())
            for tag in tags:
                tags.remove(tag)
                merge_with = [mtag for mtag in tags\
                              if mtag.lower().split("_")[2]==tag.lower().split("_")[2]]
                assert len(merge_with)==1, "attempted to merge {} ground truths".format(len(merge_with))
                merge_with = merge_with[0]
                tags.remove(merge_with)
                #masks are binary
                #will throw error if masks dont have same shape
                data["anatomical_masks"][tag] = np.logical_or(data["anatomical_masks"][tag].astype(np.uint8),
                    data["anatomical_masks"].pop(merge_with).astype(np.uint8)).astype(np.uint8)

            tags = list(data["lesions"].keys())
            for tag in tags:
                tags.remove(tag)
                merge_with = [mtag for mtag in tags\
                              if mtag.lower().split("_")[2]==tag.lower().split("_")[2]]
                assert len(merge_with)==1, "attempted to merge {} ground truths".format(len(merge_with))
                merge_with = merge_with[0]
                tags.remove(merge_with)
                data["lesions"][tag] = np.logical_or(data["lesions"][tag],
                    data["lesions"].pop(merge_with)).astype(np.uint8)

        elif mode=='adc' or mode=='t2':
            data["anatomical_masks"] = {tag:v for tag,v in data["anatomical_masks"].items() if
                                        tag.lower().split("_")[1]==mode}
            data["lesions"] = {tag: v for tag, v in data["lesions"].items() if tag.lower().split("_")[1] == mode}

        else:
            raise Exception("cross-mod gt merge mode {} not implemented".format(mode))

    for tag in list(data["anatomical_masks"]):
        data["anatomical_masks"][rename_tags[tag]] = data["anatomical_masks"].pop(tag)
        #del data["anatomical_masks"][tag]
    for tag in list(data["lesions"]):
        new_tag = "seg_REG_"+"".join(tag.split("_")[2:])
        data["lesions"][new_tag] = data["lesions"].pop(tag)
        data["lesion_gleasons"][new_tag] = data["lesion_gleasons"].pop(tag)

    return data

def crop_3D(data, pre_crop_size, center_of_mass_crop=True):
    pre_crop_size = np.array(pre_crop_size)
    # restrain z-ranges to where ADC has valid entries
    if 'adc' in data['img'].keys():
        ref_mod = 'adc'
        comp_mod = 't2'
    else:
        ref_mod = 't2'
        comp_mod = 'adc'
    min_z, max_z = get_valid_z_range(data['img'][ref_mod])    
    if comp_mod in data['img'].keys():
        assert (min_z, max_z) == get_valid_z_range(data['img'][comp_mod]), "adc, t2 different valid z range"

    if center_of_mass_crop:
        # cut the arrays to the desired x_y_crop_size around the center-of-mass of the PRO segmentation
        pro_com = center_of_mass(data['anatomical_masks']['pro'])
        center = [int(np.round(i, 0)) for i in pro_com]
    else:
        center = [data['img'][ref_mod].shape[i] // 2 for i in range(3)]
    
    
    l = pre_crop_size // 2
    #z_low, z_up = max(min_z, center[0] - l[0]), min(max_z + 1, center[0] + l[0])
    z_low, z_up = center[0] - l[0], center[0] + l[0]
    while z_low<min_z or z_up>max_z+1:
        if z_low<min_z:
            z_low += 1
            z_up += 1
            if z_up>max_z+1:
                warnings.warn("could not crop patient {}'s z-dim to demanded size.".format(data['Original_ID']))
        if z_up>max_z+1:
            z_low -= 1
            z_up -= 1
            if z_low<min_z:
                warnings.warn("could not crop patient {}'s z-dim to demanded size.".format(data['Original_ID']))

    #ensure too small image/ too large pcropsize don't lead to error
    d = np.array((z_low, center[1]-l[1], center[2]-l[2]))
    assert np.all(d>=0),\
        "Precropsize too large for image dimensions by {} pixels in patient {}".format(d, data['Original_ID'])

    for mod in data['img'].keys():
        data['img'][mod] = data['img'][mod][z_low:z_up, center[1]-l[1]: center[1] + l[1], center[2]-l[2]: center[2]+l[2]]
    vals_lst = list(data['img'].values())
    assert np.all([mod.shape==vals_lst[0].shape for mod in vals_lst]),\
    "produced modalities for same subject with different shapes"
    
    for mask in data['anatomical_masks'].keys():
        data['anatomical_masks'][mask] = data['anatomical_masks'][mask] \
            [z_low:z_up, center[1]-l[1]: center[1]+l[1], center[2]-l[2]: center[2]+l[2]]
            
    for mask in data['lesions'].keys():
        data['lesions'][mask] = data['lesions'][mask] \
            [z_low:z_up, center[1]-l[1]: center[1]+l[1], center[2]-l[2]: center[2]+l[2]]
    return data

def add_transitional_zone_mask(data):
    if 'pro' in data['anatomical_masks'] and 'pz' in data['anatomical_masks']:
        intersection = data['anatomical_masks']['pro'] & data['anatomical_masks']['pz']
        data['anatomical_masks']['tz'] = data['anatomical_masks']['pro'] - intersection
    return data

def generate_labels(data, seg_labels, class_labels, gleason_map, observables_rois):
    """merge individual binary labels to an integer label mask and create class labels from Gleason score.
        if seg_labels has seg_label 'roi': seg label will be roi count.
    """
    anatomical_masks2label = [l for l in data['anatomical_masks'].keys() if l in seg_labels.keys()]
    
    data['seg'] = np.zeros(shape=data['anatomical_masks']['pro'].shape, dtype=np.uint8)
    data['roi_classes'] = []
    #data['roi_observables']: dict, each entry is one list of length final roi_count in this patient
    data['roi_observables'] = {obs:[] for obs in observables_rois}
    roi_count = 0

    for mask in anatomical_masks2label:
        ixs = np.where(data['anatomical_masks'][mask])
        roi_class = class_labels[mask]
        if len(ixs)>0 and roi_class!=-1:
            roi_count+=1
            label = seg_labels[mask]
            if label=='roi':
                label = roi_count
            data['seg'][ixs] = label
            data['roi_classes'].append(roi_class)
            for obs in observables_rois:
                obs_val = data[obs][mask] if mask in data[obs].keys() else None
                data['roi_observables'][obs].append(obs_val)
        #print("appended mask lab", class_labels[mask])
      
    if "lesions" in seg_labels.keys():   
        for lesion_key, lesion_mask in data['lesions'].items():
            ixs = np.where(lesion_mask)
            roi_class = class_labels['lesions']
            if roi_class == "gleason":
                roi_class = gleason_map(data['lesion_gleasons'][lesion_key])
                # roi_class =  data['lesion_gleasons'][lesion_key]
            if len(ixs)>0 and roi_class!=-1:
                roi_count+=1
                label = seg_labels['lesions']
                if label=='roi':
                    label = roi_count
                data['seg'][ixs] = label
                #segs have form: slices x h x w, i.e., one channel per z-slice, each lesion has its own label
                data['roi_classes'].append(roi_class)
                for obs in observables_rois:
                    obs_val = data[obs][lesion_key] if lesion_key in data[obs].keys() else None
                    data['roi_observables'][obs].append(obs_val)

                # data['lesion_gleasons'][label] = data['lesion_gleasons'].pop(lesion_key)
    for obs in data['roi_observables'].keys():
        del data[obs]
    return data

def normalize_image(data, normalization_dict):
    """normalize the full image."""
    percentiles = normalization_dict['percentiles']
    for mod in data['img'].keys():
        p = np.percentile(data['img'][mod], percentiles[0])
        q = np.percentile(data['img'][mod], percentiles[1])
        masked_img = data['img'][mod][(data['img'][mod] > p) & (data['img'][mod] < q)]
        data['img'][mod] = (data['img'][mod] - np.median(masked_img)) / np.std(masked_img)
    return data

def concat_mods(data, mods2concat):
    """concat modalities on new first channel
    """
    concat_on_channel = [] #holds tmp data to be concatenated on the same channel
    for mod in mods2concat:
        mod_img = data['img'][mod][np.newaxis]
        concat_on_channel.append(mod_img)
    data['img'] = np.concatenate(concat_on_channel, axis=0)
    
    return data

def swap_yx(data, apply_flag):
    """swap x and y axes in img and seg
    """
    if apply_flag:
        data["img"] = np.swapaxes(data["img"], -1,-2)
        data["seg"] = np.swapaxes(data["seg"], -1,-2)

    return data

def get_fg_z_indices(seg):
    """return z-indices of array at which the x-y-arrays have labels!=0, 0 is background
    """
    fg_slices = np.argwhere(seg.astype(int))[:,0]
    fg_slices = np.unique(fg_slices)
    return fg_slices


class Preprocessor():

    def __init__(self, config):

        self._config_path = config.config_path
        self.full_cf = config
        self._cf = config.prepro

    def get_excluded_master_ids(self):
        """Get the Master IDs that are excluded from their corresponding Prisma/Radval/Master IDs."""

        excluded_prisma = self._cf['excluded_prisma_subjects']
        excluded_radval = self._cf['excluded_radval_subjects']
        excluded_master = self._cf['excluded_master_subjects']
        histo = self._histo_patient_based

        excluded_master_ids = []

        if len(excluded_prisma) > 0:
            for prisma_id in excluded_prisma:
                master_spec = histo['Master_ID'][histo['Original_ID'] == id_to_spec(prisma_id, 'Prisma')].values[0]
                excluded_master_ids.append(spec_to_id(master_spec))

        if len(excluded_radval) > 0:
            for radval_id in excluded_radval:
                master_spec = histo['Master_ID'][histo['Original_ID'] == id_to_spec(radval_id, 'Radiology')].values[0]
                excluded_master_ids.append(spec_to_id(master_spec))

        excluded_master_ids += excluded_master

        return excluded_master_ids


    def prepare_filenames(self):
        """check whether histology-backed subjects and lesions are available in the data and
        yield dict of subject file-paths."""

        # assemble list of histology-backed subject ids and check that corresponding images are available
        self._histo_lesion_based = pd.read_csv(os.path.join(self._cf['histo_dir'], self._cf['histo_lesion_based']))
        self._histo_patient_based = pd.read_csv(os.path.join(self._cf['histo_dir'], self._cf['histo_patient_based']))

        excluded_master_ids = self.get_excluded_master_ids()
        self._subj_ids = np.unique(self._histo_lesion_based[self._cf['histo_id_column_name']].values)
        self._subj_ids = [s for s in self._subj_ids.tolist() if
                          s not in excluded_master_ids]

        # get subject directory paths from
        img_paths = os.listdir(self._cf['data_dir'])
        self._img_paths = [p for p in img_paths if 'Master' in p and len(p) == len('Master') + 6]

        # check that all images of subjects with histology are available
        available_subj_ids = np.array([spec_to_id(s) for s in self._img_paths])
        self._missing_image_ids = np.setdiff1d(self._subj_ids, available_subj_ids)

        assert len(self._missing_image_ids)== 0,\
                'Images of subjs {} are not available.'.format(self._missing_image_ids)

        # make dict holding relevant paths to data of each subject
        self._paths_by_subject = {}
        for s in self._subj_ids:
            self._paths_by_subject[s] = self.load_subject_paths(s)
        

    def load_subject_paths(self, subject_id):
        """Make dict holding relevant paths to data of a given subject."""
        dir_spec = self._cf['dir_spec']
        s_dict = {}

        # iterate images
        images_paths = {}
        for kind, filename in self._cf['images'].items():
            filename += self._cf['img_postfix']+self._cf['overall_postfix']
            images_paths[kind] = os.path.join(self._cf['data_dir'], id_to_spec(subject_id, dir_spec), filename)
        s_dict['images'] = images_paths

        # iterate anatomical structures
        anatomical_masks_paths = {}
        for tag in self._cf['anatomical_masks']:
            filename = tag + self._cf['overall_postfix']
            anatomical_masks_paths[tag] = os.path.join(self._cf['data_dir'], id_to_spec(subject_id, dir_spec), filename)
        s_dict['anatomical_masks'] = anatomical_masks_paths

        # iterate lesions
        lesion_names = []
        if 'adc' in self._cf['images']:
            lesion_names.extend(self._histo_lesion_based[self._histo_lesion_based[self._cf['histo_id_column_name']]\
                                                    == subject_id]['segmentationsNameADC'].dropna())
        if 't2' in self._cf['images']:
            lesion_names.extend(self._histo_lesion_based[self._histo_lesion_based[self._cf['histo_id_column_name']]\
                                                    == subject_id]['segmentationsNameT2'].dropna())
        lesion_paths = {}
        for l in lesion_names:
            lesion_path = os.path.join(self._cf['data_dir'], id_to_spec(subject_id, dir_spec),
                                       l+self._cf['lesion_postfix']+self._cf['overall_postfix'])
            assert os.path.isfile(lesion_path), 'Lesion mask not found under {}!'.format(lesion_path)

            lesion_paths[l] = lesion_path

        s_dict['lesions'] = lesion_paths
        return s_dict


    def load_subject_data(self, subject_id):
        """load img data, masks, histo data for a single subject."""
        subj_paths = self._paths_by_subject[subject_id]
        data = {}

        # iterate images
        data['img'] = {}
        for mod in subj_paths['images']:
            data['img'][mod] = sitk.ReadImage(subj_paths['images'][mod])

        # iterate anatomical masks
        data['anatomical_masks'] = {} 
        for tag in subj_paths['anatomical_masks']:
            data['anatomical_masks'][tag] = sitk.ReadImage(subj_paths['anatomical_masks'][tag])

        # iterate lesions, include gleason score
        data['lesions'] = {}
        data['lesion_gleasons'] = {}
        idcol = self._cf['histo_id_column_name']
        subj_histo = self._histo_lesion_based[self._histo_lesion_based[idcol]==subject_id]
        for l in subj_paths['lesions']:
            #print("subjpaths lesions l ", l)
            data['lesions'][l] = sitk.ReadImage(subj_paths['lesions'][l])

            try:
                gleason = subj_histo[subj_histo["segmentationsNameADC"]==l]["Gleason"].tolist()[0]
            except IndexError:
                gleason = subj_histo[subj_histo["segmentationsNameT2"]==l]["Gleason"].tolist()[0]

            data['lesion_gleasons'][l] = gleason
        
        # add other subj-specific histo and id data
        idcol = self._cf['histo_pb_id_column_name']
        subj_histo = self._histo_patient_based[self._histo_patient_based[idcol]==subject_id]
        for d in self._cf['observables_patient']:
            data[d] = subj_histo[d].values
        
        return data

    def analyze_subject_data(self, data):
        """record post-alignment geometries."""

        ref_mods = data['img'].keys()
        geos = {}
        for ref_mod in ref_mods:
            geos[ref_mod] = {'size': data['img'][ref_mod].GetSize(), 'origin': data['img'][ref_mod].GetOrigin(),
                   'spacing': data['img'][ref_mod].GetSpacing()}

        return geos

    def process_subject_data(self, data):
        """evtly rescale images, check for geometry miss-alignments and perform crop."""
        
        if not self._cf['mod_scaling'] == (1,1,1):
            for img_name in data['img']:
                res_img = rescale(data["img"][img_name], self._cf['mod_scaling'])
                data['img'][img_name] = res_img

        #----check geometry alignment between masks and image---
        for tag in self._cf['anatomical_masks']:
            if tag.lower().startswith("seg_adc"):
                ref_mod = 'adc'
            elif tag.lower().startswith("seg_t2"):
                ref_mod = 't2'
            if not has_equal_geometry(data['img'][ref_mod], data['anatomical_masks'][tag]):
                #print("bef", np.unique(sitk.GetArrayFromImage(data['anatomical_masks'][tag])))
                #print('Geometry mismatch: {}, {} is resampled to its image geometry!'.format(data["Original_ID"], tag))
                data['anatomical_masks'][tag] =\
                    resample_to_reference(data['img'][ref_mod], data['anatomical_masks'][tag],
                                          interpolation=self._cf['interpolation'])
                #print("aft", np.unique(sitk.GetArrayFromImage(data['anatomical_masks'][tag])))

        for tag in data['lesions'].keys():
            if tag.lower().startswith("seg_adc"):
                ref_mod = 'adc'
            elif tag.lower().startswith("seg_t2"):
                ref_mod = 't2'
            if not has_equal_geometry(data['img'][ref_mod], data['lesions'][tag]):
                #print('Geometry mismatch: {}, {} is resampled to its image geometry!'.format(data["Original_ID"], tag))
                #print("pre-sampling data type: {}".format(data['lesions'][tag]))
                data['lesions'][tag] = resample_to_reference(data['img'][ref_mod], data['lesions'][tag],
                                                              interpolation=self._cf['interpolation'])


        data = convert_to_arrays(data)
        data = merge_crossmod_masks(data, self._cf['rename_tags'], mode=self._cf['merge_mode'])
        data = crop_3D(data, self._cf['pre_crop_size'], self._cf['center_of_mass_crop'])
        data = add_transitional_zone_mask(data)
        data = generate_labels(data, self._cf['seg_labels'], self._cf['class_labels'], self._cf['gleason_map'],
                               self._cf['observables_rois'])
        data = normalize_image(data, self._cf['normalization'])
        data = concat_mods(data, self._cf['modalities2concat'])
        data = swap_yx(data, self._cf["swap_yx_to_xy"])
        
        data['fg_slices'] = get_fg_z_indices(data['seg'])
        
        return data

    def write_subject_arrays(self, data, subject_spec):
        """Write arrays to disk and save file names in dict."""

        out_dir = self._cf['output_directory']
        os.makedirs(out_dir, exist_ok=True) #might throw error if restrictive permissions

        out_dict = {}

        # image(s)
        name = subject_spec + '_imgs.npy'
        np.save(os.path.join(out_dir, name), data['img'])
        out_dict['img'] = name
        
        # merged labels
        name = subject_spec + '_merged_seg.npy'
        np.save(os.path.join(out_dir, name), data['seg'])
        out_dict['seg'] = name

        # anatomical masks separately
        #for mask in list(data['anatomical_masks'].keys()) + (['tz'] if 'tz' in data.keys() else []):
        #    name = subject_spec + '_{}.npy'.format(mask)
        #    np.save(os.path.join(out_dir, name), data['anatomical_masks'][mask])
        #    out_dict[mask] = name

        # lesion masks and lesion classes separately
        #out_dict['lesion_gleasons'] = {}
        #for mask in data['lesions'].keys():
        #    name = subject_spec + '_{}.npy'.format(mask)
        #    np.save(os.path.join(out_dir, name), data['lesions'][mask])
        #    out_dict[mask] = name
        #    out_dict['lesion_gleasons'][int(mask[-1])] = data['lesion_gleasons'][int(mask[-1])]
            
        # roi classes
        out_dict['roi_classes'] = data['roi_classes']

        
        # fg_slices info
        out_dict['fg_slices'] = data['fg_slices']
        
        # other observables
        for obs in self._cf['observables_patient']:
            out_dict[obs] = data[obs]
        for obs in data['roi_observables'].keys():
            out_dict[obs] = data['roi_observables'][obs]
        #print("subj outdict ", out_dict.keys())
        return out_dict

    def subject_iteration(self, subj_id): #single iteration, wrapped for pooling
        data = self.load_subject_data(subj_id)
        data = self.process_subject_data(data)
        subj_out_dict = self.write_subject_arrays(data, id_to_spec(subj_id, self._cf['dir_spec']))
        
        print('Processed subject {}.'.format(id_to_spec(subj_id, self._cf['dir_spec'])))
        
        return (subj_id, subj_out_dict)
        
    def iterate_subjects(self, ids_subset=None, processes=6):
        """process all subjects."""
        
        if ids_subset is None:
            ids_subset = self._subj_ids
        else:
            ids_subset = np.array(ids_subset)
            id_check = np.array([id in self._subj_ids for id in ids_subset])
            assert np.all(id_check), "pids {} not in eligible pids".format(ids_subset[np.invert(id_check)])

        p = Pool(processes)
        subj_out_dicts = p.map(self.subject_iteration, ids_subset)
        """note on Pool.map: only takes one arg, pickles the function for execution -->
        cannot write to variables defined outside local scope --> cannot write to
        self.variables, therefore need to return single subj_out_dicts and join after;
        however p.map can access object methods via self.method().
        Is a bit complicated, but speedup is huge.
        """
        p.close()
        p.join()
        assert len(subj_out_dicts)==len(ids_subset), "produced less subject dicts than demanded"
        self._info_dict = {id:dic for (id, dic) in subj_out_dicts}
        
        return

    def subject_analysis(self, subj_id):  # single iteration, wrapped for pooling
        data = self.load_subject_data(subj_id)
        analysis = self.analyze_subject_data(data)

        print('Analyzed subject {}.'.format(id_to_spec(subj_id, self._cf['dir_spec'])))

        return (subj_id, analysis)

    def analyze_subjects(self, ids_subset=None, processes=os.cpu_count()):
        """process all subjects."""

        if ids_subset is None:
            ids_subset = self._subj_ids
        else:
            ids_subset = np.array(ids_subset)
            id_check = np.array([id in self._subj_ids for id in ids_subset])
            assert np.all(id_check), "pids {} not in eligible pids".format(ids_subset[np.invert(id_check)])

        p = Pool(processes)
        subj_analyses = p.map(self.subject_analysis, ids_subset)
        """note on Pool.map: only takes one arg, pickles the function for execution -->
        cannot write to variables defined outside local scope --> cannot write to
        self.variables, therefore need to return single subj_out_dicts and join after;
        however p.map can access object methods via self.method().
        Is a bit complicated, but speedup is huge.
        """
        p.close()
        p.join()

        df = pd.DataFrame(columns=['id', 'mod', 'size', 'origin', 'spacing'])
        for subj_id, analysis in subj_analyses:
            for mod, geo in analysis.items():
                df.loc[len(df)] = [subj_id, mod, np.array(geo['size']), np.array(geo['origin']), np.array(geo['spacing'])]

        os.makedirs(self._cf['output_directory'], exist_ok=True)
        df.to_csv(os.path.join(self._cf['output_directory'], "analysis_df"))

        print("\nOver all mods")
        print("Size mean {}\u00B1{}".format(df['size'].mean(), np.std(df['size'].values)))
        print("Origin mean {}\u00B1{}".format(df['origin'].mean(), np.std(df['origin'].values)))
        print("Spacing mean {}\u00B1{}".format(df['spacing'].mean(), np.std(df['spacing'].values)))
        print("-----------------------------------------\n")

        for mod in df['mod'].unique():
            print("\nModality: {}".format(mod))
            mod_df = df[df['mod']==mod]
            print("Size mean {}\u00B1{}".format(mod_df['size'].mean(), np.std(mod_df['size'].values)))
            print("Origin mean {}\u00B1{}".format(mod_df['origin'].mean(), np.std(mod_df['origin'].values)))
            print("Spacing mean {}\u00B1{}".format(mod_df['spacing'].mean(), np.std(mod_df['spacing'].values)))
            print("-----------------------------------------\n")
        return


    def dump_class_labels(self, out_dir):
        """save used GS mapping and class labels to file.
            will likely not work if non-lesion classes (anatomy) are contained
        """
        #if "gleason_thresh" in self._cf.keys():
        possible_gs = {gs for p_dict in self._info_dict.values() for gs in p_dict['lesion_gleasons']}
        gs_mapping_inv = [(self._cf["gleason_map"](gs)+1, gs) for gs in possible_gs]
        #elif "gleason_mapping" in self._cf.keys():
            #gs_mapping_inv = [(val + 1, key) for (key, val) in self._cf["gleason_mapping"].items() if val != -1]
        classes = {pair[0] for pair in gs_mapping_inv}
        groups = [[pair[1] for pair in gs_mapping_inv if pair[0]==cl] for cl in classes]
        gr_names = [ "GS{}-{}".format(min(gr), max(gr)) if len(gr)>1 else "GS"+str(*gr) for gr in groups ]
        if "color_palette" in self._cf.keys():
            class_labels = {cl: {"gleasons": groups[ix], "name": gr_names[ix], "color": self._cf["color_palette"][ix]}
                            for ix, cl in enumerate(classes) }
        else:
            class_labels = {cl: {"gleasons": groups[ix], "name": gr_names[ix], "color": self.full_cf.color_palette[ix]}
                            for ix, cl in enumerate(classes)}

        save_obj(class_labels, os.path.join(out_dir,"pp_class_labels"))



    def save_and_finish(self):
        """copy config and used code to out_dir."""

        out_dir = self._cf['output_directory']

        # save script
        current_script = os.path.realpath(__file__)
        shutil.copyfile(current_script, os.path.join(out_dir, 'applied_preprocessing.py'))

        # save config
        if self._config_path[-1] == 'c':
            self._config_path = self._config_path[:-1]
        shutil.copyfile(self._config_path, os.path.join(out_dir, 'applied_config.py'))
        
        #copy histo data to local dir
        lbased = self._cf['histo_lesion_based']
        pbased = self._cf['histo_patient_based']
        os.makedirs(self._cf['histo_dir_out'], exist_ok=True)
        shutil.copyfile(self._cf['histo_dir']+lbased, self._cf['histo_dir_out']+lbased)
        shutil.copyfile(self._cf['histo_dir']+pbased, self._cf['histo_dir_out']+pbased)
       
        # save info dict
        #print("info dict ", self._info_dict)
        save_obj(self._info_dict, self._cf['info_dict_path'][:-4])
        self.dump_class_labels(out_dir)

        return
    
    def convert_copy_npz(self):
        if not self._cf["npz_dir"]:
            return
        print("npz dir", self._cf['npz_dir'])
        os.makedirs(self._cf['npz_dir'], exist_ok=True)
        save_obj(self._info_dict, os.path.join(self._cf['npz_dir'], 
                                               self._cf['info_dict_path'].split("/")[-1][:-4]))
        lbased = self._cf['histo_lesion_based']
        pbased = self._cf['histo_patient_based']
        histo_out = os.path.join(self._cf['npz_dir'], "histos/")
        print("histo dir", histo_out)
        os.makedirs(histo_out, exist_ok=True)
        shutil.copyfile(self._cf['histo_dir']+lbased, histo_out+lbased)
        shutil.copyfile(self._cf['histo_dir']+pbased, histo_out+pbased)
        shutil.copyfile(os.path.join(self._cf['output_directory'], 'applied_config.py'),
                        os.path.join(self._cf['npz_dir'], 'applied_config.py'))
        shutil.copyfile(os.path.join(self._cf['output_directory'], 'applied_preprocessing.py'),
                        os.path.join(self._cf['npz_dir'], 'applied_preprocessing.py'))
        shutil.copyfile(os.path.join(self._cf['output_directory'], 'pp_class_labels.pkl'),
                        os.path.join(self._cf['npz_dir'], 'pp_class_labels.pkl'))
        
        dmanager.pack_dataset(self._cf["output_directory"], self._cf["npz_dir"], recursive=True)
        
        
        


if __name__ == "__main__":

    stime = time.time()
    
    from configs import Configs
    cf = configs()
    
    
    pp = Preprocessor(config=cf)
    pp.prepare_filenames()
    #pp.analyze_subjects(ids_subset=None)#[1,2,3])
    pp.iterate_subjects(ids_subset=None, processes=os.cpu_count())
    pp.save_and_finish()
    pp.convert_copy_npz()
    
   
    #patient_id = 17
    #data = pp.load_subject_data(patient_id)
    #data = pp.process_subject_data(data)
    
    #img = data['img']
    #print("img shape ", img.shape)
    #print("seg shape ",  data['seg'].shape)
    #label_remap = {0:0}
    #label_remap.update({roi_id : 1 for roi_id in range(1,5)})
    #plg.view_slices(cf, img[0], data['seg'], instance_labels=True,
    #                out_dir="experiments/dev/ex_slices.png")
    
    mins, secs = divmod((time.time() - stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs)) 
    print("Prepro program runtime: {}".format(t))
