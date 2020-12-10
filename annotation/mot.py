import os
import json
import xml.etree.cElementTree as ET
from glob import glob
from tqdm import tqdm

import cv2

from .base import TXTFile
from .appbase import AppBase
from .image import ImageFile
from .shape import Rectangle

class MotGT(AppBase, TXTFile):
    def __init__(self, filepath, imgfolder=None, all_labels=None, check_exist=True, verbose=False, **kwargs):
        super().__init__(filepath, check_exist)
        self.imgfolder = imgfolder
        self.verbose = verbose
        self.all_labels = all_labels
    
    def parse(self):
        self.frame_dict = {}
        with tqdm(self.lines, leave=False, desc='parsing') as pbar:
            for l in pbar:
                fidx,identity,x1,y1,w,h,ignore,lid,occ = l.replace(' ','').split(',')
                label = self.all_labels[int(lid)] if self.all_labels else str(lid)
                rect = Rectangle(x1,y1,w,h, format='xywh', label=label)
                fidx = int(fidx)
                identity = int(identity)
                
                rect.id = identity
                
                if fidx not in self.frame_dict:
                    self.frame_dict[fidx] = {'shapes': [rect]}
                else:
                    self.frame_dict[fidx]['shapes'].append(rect)
            
        self._parse_imgfile()
        
    def _parse_imgfile(self):
        for f in glob(os.path.join(self.imgfolder, '*.jpg')):
            fname = os.path.splitext(os.path.basename(f))[0]
            fidx = int(fname)
            if fidx in self.frame_dict:
                if not self.frame_dict[fidx].get('imgfp'):
                    self.frame_dict[fidx]['imgfp'] = f
                elif self.verbose:
                    print(f'duplicated file/frame_index: [{fidx}] {f}')
            elif self.verbose:
                print(f'missing annotation at frame_index [{fidx}]')
        
        if self.verbose:
            for fidx,val in self.frame_dict.items():
                if not val.get('imgfp'):
                    print(f'missing file at frame_index [{fidx}]')
    
    def from_(self, img_path, shapes):
        raise NotImplementedError

    def save(self, dst=None):
        raise NotImplementedError


class MotDET(MotGT):
    def __init__(self, filepath, one_label, imgfolder=None, check_exist=True, verbose=False, **kwargs):
        super().__init__(filepath, imgfolder=imgfolder, all_labels=None, check_exist=check_exist, verbose=verbose)
        self.one_label = one_label
    
    def parse(self):
        self.frame_dict = {}
        with tqdm(self.lines, leave=False, desc='parsing') as pbar:
            for l in pbar:
                fidx,identity,x1,y1,w,h,conf,_,_,_ = l.replace(' ','').split(',')
                rect = Rectangle(x1,y1,w,h, format='xywh', label=self.one_label)
                fidx = int(fidx)
                identity = int(identity)
                conf = float(conf)
                
                rect.id = identity
                rect.conf = conf
                
                if fidx not in self.frame_dict:
                    self.frame_dict[fidx] = {'shapes': [rect]}
                else:
                    self.frame_dict[fidx]['shapes'].append(rect)
            
        self._parse_imgfile()