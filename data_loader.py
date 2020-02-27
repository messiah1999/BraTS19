import numpy as np
import os
from termcolor import colored
from tabulate import tabulate

from tensorpack.utils import logger
from tensorpack.utils.timer import timed_operation
from tensorpack.utils.argtools import log_once

import random
import pickle
import glob
from tqdm import tqdm
import config
import cv2
import skimage
import nibabel
from utils import crop_brain_region

class BRATS_SEG(object):
    def __init__(self, basedir, mode):
        self.basedir = os.path.join(basedir, mode)
        self.mode = mode
    
    def load_5fold(self):
        with open(config.CROSS_VALIDATION_PATH, 'rb') as f:
            data = pickle.load(f)
        imgs = data["fold{}".format(config.FOLD)][self.mode]
        patient_ids = [x.split("/")[-1] for x in imgs]
        ret = []
        for idx, file_name in enumerate(imgs):
            data = {}
            data['image_data'] = {}
            data['file_name'] = file_name
            data['id'] = patient_ids[idx]
            mod = glob.glob(file_name+"/*.nii*")
            assert len(mod) >= 4  
            for m in mod:
                if 'seg' in m:
                    data['gt'] = m
                else:
                    _m = m.split("/")[-1].split(".")[0].split("_")[-1]
                    data['image_data'][_m] = m
            if 'gt' in data:
                data['preprocessed'] = crop_brain_region(data['image_data'], data['gt'])
                del data['image_data']
                del data['gt']
            ret.append(data)
        return ret

    def load_3d(self):
        print("Data Folder: ", self.basedir)
        modalities = ['flair', 't1ce', 't1.', 't2']
        if 'training' in self.basedir:
            img_HGG = glob.glob(self.basedir+"/HGG/*")
            img_LGG = glob.glob(self.basedir+"/LGG/*")
            imgs = img_HGG + img_LGG
        else:
            imgs = glob.glob(self.basedir+"/*")
        imgs = [x for x in imgs if 'survival_evaluation.csv' not in x]
        
        patient_ids = [x.split("/")[-1] for x in imgs]
        ret = []
        print("Preprocessing Data ...")
        for idx, file_name in tqdm(enumerate(imgs), total=len(imgs)):
            data = {}
            data['image_data'] = {}
            data['file_name'] = file_name
            data['id'] = patient_ids[idx]
            mod = glob.glob(file_name+"/*.nii*")
            assert len(mod) >= 4, '{}'.format(file_name)
            for m in mod:
                if 'seg' in m:
                    data['gt'] = m
                else:
                    _m = m.split("/")[-1].split(".")[0].split("_")[-1]
                    data['image_data'][_m] = m
            
            if 'gt' in data:
                if not config.NO_CACHE and not 'training' in self.basedir:
                    data['preprocessed'] = crop_brain_region(data['image_data'], data['gt'])
                    del data['image_data']
                    del data['gt']
            else:
                data['preprocessed'] = crop_brain_region(data['image_data'], None, with_gt=False)
                del data['image_data']

            ret.append(data)
        return ret

    @staticmethod
    def load_from_file(basedir, names):
        brats = BRATS_SEG(basedir, names)
        return  brats.load_5fold()

    @staticmethod
    def load_many(basedir,names, add_gt=True, add_mask=False):
        if not isinstance(names, (list, tuple)):
            names = [names]
        ret = []
        for n in names:
            brats = BRATS_SEG(basedir, n)
            ret.extend(brats.load_3d())
        return ret

if __name__ == '__main__':
    brats2018 = BRATS_SEG("/BRATS2018/", "training")
    brats2018 = brats2018.load_3d()
    print(len(brats2018))
    print(brats2018[0])
