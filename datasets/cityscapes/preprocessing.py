import sys
import os
from multiprocessing import Pool
import time
import pickle

import numpy as np

from PIL import Image as pil
from matplotlib import pyplot as plt

sys.path.append("../")
import data_manager as dmanager

from configs import Configs
cf = configs()


"""
"""

def load_obj(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

def save_obj(obj, path):
    """Pickle a python object."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def merge_labelids(target, cf=cf):
    """relabel preprocessing id to training id according to config.labels
    :param target: np.array hxw holding the annotation (labelids at pixel positions)
    :cf: The configurations file
    """
    for i in range(target.shape[0]):  #Iterate over height.
        for j in range(target.shape[1]): #Iterate over width
            target[i][j] = cf.ppId2id[int(target[i][j])]
            
    return target

def generate_detection_labels(target, cf=cf):
    """labels suitable to be used with batchgenerators.ConvertSegToBoundingBoxCoordinates.
    Flaw: cannot handle more than 2 segmentation classes (fg/bg).
    --> seg-info is lost, but not interested in seg rn anyway.
    :param target: expected as instanceIds img
         The pixel values encode both, class and the individual instance.
         The integer part of a division by 1000 of each ID provides the class ID,
         as described in labels.py. The remainder is the instance ID. If a certain
         annotation describes multiple instances, then the pixels have the regular
         ID of that class.
    """

    unique_IDs = np.unique(target)
    roi_classes = []
    
    objs_in_img = 0
    for i, instanceID in enumerate(unique_IDs):
        if instanceID > max(list(cf.ppId2id.keys())):
            instance_classID = instanceID // 1000
        else:
            # this is the group case (only class id assigned, no instance id)
            instance_classID = instanceID 
            if cf.ppId2id[instance_classID]!=0:
                #discard this whole sample since it has group instead of 
                #single instance annotations for a non-bg class
                return None, None
        
        if cf.ppId2id[instance_classID]!=0:
            #only pick reasonable objects, exclude road, sky, etc.
            roi_classes.append(cf.ppId2id[instance_classID])
            objs_in_img+=1 #since 0 is bg
            target[target==instanceID] = objs_in_img
        else:
            target[target==instanceID] = 0

    return target, roi_classes

class Preprocessor():
    
    def __init__(self, cf, cities):
        
        self._cf = cf.prepro
        
        self.rootpath = cf.prepro['data_dir']
        self.set_splits = self._cf["set_splits"]
        self.cities = cities
        self.datapath = cf.datapath
        self.targetspath = cf.targetspath
        self.targettype = cf.prepro["targettype"]
        
        self.img_t_size = cf.prepro["img_target_size"]
        self.target_t_size = self.img_t_size
        
        self.rootpath_out = cf.prepro["output_directory"]
        
        self.info_dict = {}
        """info_dict: will hold {img_identifier: img_dict} with
            img_dict = {id: img_identifier, img:img_path, seg:seg_path,
            roi_classes:roiclasses}
        """
     
    def load_from_path_to_path(self, set_split, max_num=None):
        """composes data and corresponding labels paths (to .png-files).
        
        assumes data tree structure:   datapath-|-->city1-->img1.png,img2.png,...
                                                |-->city2-->img1.png, ...
        """
        data = []
        labels = []
        num=0
        for city in self.cities[set_split]:
            path = os.path.join(self.rootpath, self.datapath, set_split, city)
            lpath = os.path.join(self.rootpath,self.targetspath,set_split, city)

            files_in_dir = os.listdir(path)        
            for file in files_in_dir:
                split = os.path.splitext(file)
                if split[1].lower() == ".png":
                    num+=1
                    filetag = file[:-(len(self.datapath)+3)]
                    data.append(os.path.join(path,file))
                    labels.append(os.path.join(lpath,filetag+self.targettype+".png"))
                    
                    if num==max_num:
                        break
            if num==max_num:
                break
      
        return data, labels 
        
    def prep_img(self, args):
        """suited for multithreading.
        :param args: (img_path, targ_path)
        """           

        img_path, trg_path = args[0], args[1]
        
        img_rel_path = img_path[len(self.rootpath):]
        trg_rel_path = trg_path[len(self.rootpath):]
        
        _path, img_name = os.path.split(img_path)        
        img_identifier = "".join(img_name.split("_")[:3])
        img_info_dict = {} #entry of img_identifier in full info_dict
        
        img, target = pil.open(img_path), pil.open(trg_path)
        img, target = img.resize(self.img_t_size[::-1]), target.resize(self.target_t_size[::-1])
        img, target = np.array(img), np.array(target) #shapes y,x(,c)
        img         = np.transpose(img, axes=(2,0,1)) #shapes (c,)y,x
        
        target, roi_classes = generate_detection_labels(target)
        if target is None:
            return (img_identifier, target)
        img_info_dict["roi_classes"] = roi_classes

        path = os.path.join(self.rootpath_out,*img_rel_path.split(os.path.sep)[:-1])
        os.makedirs(path, exist_ok=True)

        img_path = os.path.join(self.rootpath_out, img_rel_path[:-3]+"npy")

        #img.save(img_path)
        img_info_dict["img"] = img_rel_path[:-3]+"npy"
        np.save(img_path, img)
        
        path = os.path.join(self.rootpath_out,*trg_rel_path.split(os.path.sep)[:-1])
        os.makedirs(path, exist_ok=True)
        t_path = os.path.join(self.rootpath_out, trg_rel_path)[:-3]+"npy"
        #target.save(t_path)
        img_info_dict["seg"] = trg_rel_path[:-3]+"npy"
        np.save(t_path, target)
            
        print("\rSaved npy images and targets of shapes {}, {} to files\n {},\n {}". \
                format(img.shape, target.shape, img_path, t_path), flush=True, end="")
        
        return (img_identifier, img_info_dict)
    
    def prep_imgs(self, max_num=None, processes=4):
        self.info_dict = {}
        self.discarded = []
        os.makedirs(self.rootpath_out, exist_ok=True)
        for set_split in self.set_splits:
            data, targets = self.load_from_path_to_path(set_split, max_num=max_num)
            
            print(next(zip(data, targets)))
            p = Pool(processes)
            
            img_info_dicts = p.map(self.prep_img, zip(data, targets))

            p.close()
            p.join()

            self.info_dict.update({id_:dict_ for (id_,dict_) in img_info_dicts if dict_ is not None})
            self.discarded += [id_ for (id_, dict_) in img_info_dicts if dict_ is None]
            #list of samples discarded due to group instead of single instance annotation
        
    def finish(self):
        total_items = len(self.info_dict)+len(self.discarded)
        
        print("\n\nSamples discarded: {}/{}={:.1f}%, identifiers:".format(len(self.discarded),
              total_items, len(self.discarded)/total_items*100))
        for id_ in self.discarded:
            print(id_)
            
        save_obj(self.info_dict, self._cf["info_dict_path"])


    def convert_copy_npz(self):
        if not self._cf["npz_dir"]:
            return
        print("converting & copying to npz dir", self._cf['npz_dir'])
        os.makedirs(self._cf['npz_dir'], exist_ok=True)
        save_obj(self.info_dict, os.path.join(self._cf['npz_dir'], 
                                               self._cf['info_dict_path'].split("/")[-1]))
        
        dmanager.pack_dataset(self._cf["output_directory"], self._cf["npz_dir"], recursive=True, verbose=False)


    def verification(self, max_num=None):
        print("\n\n\nVerification\n")
        for i, k in enumerate(self.info_dict):
            if max_num is not None and i==max_num:
                break
            
            subject = self.info_dict[k]
            
            seg = np.load(os.path.join(self.rootpath_out, subject["seg"]))
            
            #print("seg values", np.unique(seg))
            print("nr of objects", len(subject["roi_classes"]))
            print("nr of objects should equal highest seg value, fulfilled?",
                  np.max(seg)==len(subject["roi_classes"]))
            #print("roi_classes", subject["roi_classes"])
            
            img = np.transpose(np.load(os.path.join(self.rootpath_out, subject["img"])), axes=(1,2,0))
            print("img shp", img.shape)
            plt.imshow(img)         
                            
        
def main():
    #cf.set_splits = ["train"]
    #cities = {'train':['dusseldorf'], 'val':['frankfurt']} #cf.cities
    cities= cf.cities
    
    pp = Preprocessor(cf, cities)
    pp.prep_imgs(max_num=None, processes=8)
    pp.finish()

    #pp.convert_copy_npz()

    pp.verification(1)
    
    
    
    
    
    
    return

if __name__=="__main__":
    stime = time.time()
    
    main()
    
    mins, secs = divmod((time.time() - stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs)) 
    print("Prepro program runtime: {}".format(t))
