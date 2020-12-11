import os
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np

def get_data(config):
    return_A, return_B = [], []
    print("Constructing data list for A: {}, B: {} ...".format(config.domain_A, config.domain_B))
    for _path, _folders, _files in tqdm(os.walk(config.data_path)):
        if config.domain_A in _path:
            if len(_files)>0:
                _files = [os.path.join(_path, i) for i in os.listdir(_path) if config.extension.lower() in i.lower()]
                return_A += _files
        if config.domain_B in _path:
            if len(_files)>0:
                _files = [os.path.join(_path, i) for i in os.listdir(_path) if config.extension.lower() in i.lower()]
                return_B += _files
    return return_A, return_B

def preprocessing(datapaths, config):
    npy_list = []
    if config.extension.lower() in ['png', 'jpg', 'jpeg', 'dcm']:
        for datapath in datapaths:
            npy = sitk.GetArrayFromImage(sitk.ReadImage(datapath)).squeeze()
            if len(npy.shape)==2:
                npy = np.expand_dims(npy, axis=0)
            npy_list.append(npy)
    if config.extension.lower() == 'npy':
         for datapath in datapaths:
            npy = np.load(datapath).squeeze()
            if len(npy.shape)==2:
                npy = np.expand_dims(npy, axis=0)
            npy_list.append(npy)
    npy_list = np.array(npy_list)
    npy_list[npy_list<-1024] = -1024
    npy_list[npy_list> 3072] = 3072
    if config.normalize == 'minmax':
        npy_list -= npy_list.min()
        npy_list = npy_list / npy_list.max()
    elif config.normalize == 'tanh':
        npy_list -= npy_list.min()
        npy_list = npy_list / npy_list.max()
    elif config.normalize == 'CT':
        npy_list = npy_list + 1024
        npy_list = npy_list / 4095
    else:
        npy_list -= npy_list.min()
    return npy_list

