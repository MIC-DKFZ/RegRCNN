
import os
import warnings
import time
import subprocess
from multiprocessing import Pool

import argparse
import shutil

import numpy as np


def get_identifiers(folder, ending=".npz"):
    identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith(ending)]
    return identifiers

def convert_to_npz(kwargs):
    """
    :param kwargs: npy-file:path or name:namestr and destination:path
    """
    assert "identifier" in kwargs.keys(), "you need to define at least a npyfile-identifier"
    identifier = kwargs["identifier"]
    if "folder" in kwargs.keys():
        folder = kwargs["folder"]
    else:
        folder = ""
    dest = kwargs["destination"]
    if "name" in kwargs.keys():
        name = kwargs["name"]
    else:
        name = identifier
    
    npy_file = os.path.join(folder, identifier+".npy")   
    data = np.load(npy_file)
    np.savez_compressed(os.path.join(dest, identifier + ".npz"), **{name:data})
    if "verbose" in kwargs.keys() and kwargs["verbose"]:
        print("converted file {} to npz".format(npy_file))

def pack_dataset(folder, destination=None, recursive=False, processes=None, verbose=True):
    """call convert_to_npz parallely with "processes" processes on all npys in folder.
    does not actually pack more than one file into an archive...
    """
    if processes is None:
        processes = os.cpu_count()
    
    p = Pool(processes)
    
    if destination is None:
        destination = folder
    if recursive:
        folders = [root for (root, dir, file) in os.walk(folder)]
    else:
        folders = [folder]
    
    for fldr in folders:
        identifiers = get_identifiers(fldr, ".npy")
        if recursive:
            cur_dest = os.path.join(destination, os.path.relpath(fldr, folder))
        else:
            cur_dest = destination
        if not os.path.isdir(cur_dest):
            os.mkdir(cur_dest)
                   
        kwargs  = [{"folder":fldr, "identifier":ident, "destination":cur_dest, "verbose":verbose} for ident in identifiers]
        p.map(convert_to_npz, kwargs) 
        print("converted folder {}.".format(fldr))
    p.close()
    p.join()


def convert_to_npy(kwargs):
    identifier = kwargs["identifier"]
    folder = kwargs["folder"]
    delete = kwargs["delete"]
    npz_file = os.path.join(folder,identifier+".npz")
    
    if os.path.isfile(npz_file[:-4] + ".npy"):
        print("{}.npy already exists, not overwriting.".format(npz_file[:-4]))
    else:
        data = np.load(npz_file)[identifier] # should be the only entry anyway
        np.save(npz_file[:-4] + ".npy", data)
        print("converted {} from npz to npy".format(npz_file[:-4]))
        if delete:
            os.remove(npz_file)

def unpack_dataset(folder, recursive=False, delete=True, processes=None):
    if processes is None:
        processes = os.cpu_count()
    
    p = Pool(processes)
    
    if recursive:
        folders = [root for (root, dir, file) in os.walk(folder)]
    else:
        folders = [folder]

    for fldr in folders:
        identifiers = get_identifiers(fldr)
        kwargs = [{"folder":fldr, "identifier":ident, "delete":delete} for ident in identifiers]
        p.map(convert_to_npy, kwargs)
        print("unpacked folder ", fldr)
    p.close()
    p.join()

def delete_npy(folder, recursive=False): #not used    
    identifiers = get_identifiers(folder)
    npy_files = [os.path.join(folder, i+".npy") for i in identifiers]
    #should not be necessary since get_iden already returns only existing files:
    npy_files = [i for i in npy_files if os.path.isfile(i)] 
    for n in npy_files:
        os.remove(n)

def copy(args, file_subset=None, verbose=True):
    r"""copy and evtly unpack dataset (convert npz->npy) or pack dataset (npy->npz).
    :param file_subset: copy only files whose names are in file_subset
    """
    
    source_path = args.source
    dest_path = args.destination
    assert dest_path is not None, "you need to define a copy destination"
    start_time = time.time()
    print("Destination: ", dest_path)

    rsync_opts = "-v " if verbose else ""
    if args.recursive:
        rsync_opts += r"-a --include '**/'"
    if args.cp_only_npz:
        rsync_opts+= r" --include '*.npz'"  #to copy only npz files

    try:
        rsync_opts_all = rsync_opts
        if file_subset is not None: #ranks higher than only-npz criterium
        #rsync include/exclude doesnt work with absolute paths for the files!! :/:/
            for file in file_subset:
                if os.path.isabs(file):
                    file = os.path.relpath(file, source_path)
                rsync_opts_all +=r" --include '{}'".format(file)
        if args.cp_only_npz or file_subset is not None:
            rsync_opts_all += r" --exclude '*'" #has to be added after all --includes
        subprocess.call('rsync {} {} {}'.format(rsync_opts_all,
                        source_path, dest_path), shell=True)
    except OSError: #in case argument list too long due to file subset
        warnings.warn("OSError when trying to copy file_subset at once. Copying single files instead.")
        if file_subset is not None:
            for file in file_subset:
                rsync_opts_file = rsync_opts+" --include '{}' --exclude '*'".format(file)
                subprocess.call('rsync {} {} {}'.format(rsync_opts_file,
                                source_path, dest_path), shell=True)
        else:
            if args.cp_only_npz:
                rsync_opts += r" --exclude '*'"
            subprocess.call('rsync {} {} {}'.format(rsync_opts,
                        source_path, dest_path), shell=True)
    #one would only need the part in exception catcher, but try part might be faster if feasible
            
    if not args.keep_packed:
        unpack_dataset(dest_path, recursive=args.recursive, delete=args.del_after_unpack, processes=args.threads)
    #elif pack_copied_dataset:
    #    pack_dataset(dest_path, recursive=args.recursive)
    mins, secs = divmod((time.time() - start_time), 60)
    t = "{:d}m:{:02d}s".format(int(mins), int(secs))  
    print("copying and unpacking data set took: {}".format(t))
    try:
        copied_files = [file for (root, dir, file) in os.walk(dest_path)]
        print("nr of files in destination: {}".format(len(copied_files)))
    except FileNotFoundError: #fails if destination is on a server
        pass


if __name__=="__main__":
    """ usage: create folder containing converted npys (i.e., npzs) and all other data that needs to be copied,
        copy the folder, evtly unpack to npy again.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help="convert, copy, or delete. convert: npy->npz. delete: rmtree dest.")
    parser.add_argument('-s', '--source', type=str, help="Source path to folder containing data.")
    parser.add_argument('-d', '--destination', type=str, default=None, help="Destination path")
    parser.add_argument('--keep_packed', action='store_true', help="after copying, do not convert to npy.")
    #parser.add_argument('--pack_after_copy', action='store_true', help="after copying, convert npy to npz.")
    parser.add_argument('-r', '--recursive', action='store_true')
    parser.add_argument('--cp_only_npz', action='store_true', help="whether to copy only .npz-files")
    parser.add_argument('--del_after_unpack', action='store_true', help="whether to delete npz after unpacking them")
    parser.add_argument('--threads', type=int, default=None, help="how many cpu threads to use for conversions")
    
    args = parser.parse_args()
    mode = args.mode

    if mode == "convert":
        #convert from npy to npz
        pack_dataset(args.source, destination=args.destination, recursive=args.recursive, processes=args.threads)
    elif mode == 'copy':
        copy(args)
    elif mode == 'delete':
        shutil.rmtree(args.destination) 
    else:
        'cluster_data_manager: chosen mode not implemented.'
