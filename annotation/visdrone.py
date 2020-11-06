import os
from tqdm import tqdm

from .base import TXTFile
from .appbase import AppBase
from .shape import Rectangle
from .labelimg import LabelImgXML

class VisDET(AppBase):
    def __init__(self, filepath, imgpath=None, check_exist=True, all_labels=None):
        super().__init__(filepath, check_exist=check_exist)
        self.imgpath = imgpath
        self.all_labels = all_labels
    
    def parse(self):
        with open(self.filepath, 'r', encoding="utf8", errors='ignore') as f:
            self.data = f.readlines()
        
        self.__rects = []
        self._sh_dict = {'rectangle': self.__rects}
        for l in self.data:
            x1,y1,w,h,conf,lid,_,_ = l.replace(' ', '').split(',')
            
            if self.all_labels:
                label = self.all_labels[int(lid)]
            else:
                label = str(lid)
            
            
            rect = Rectangle(x1,y1,w,h, format='xywh', label=label)
            self.__rects.append(rect)
    
    def from_(self, img_path, shapes):
        raise NotImplementedError
    
    def save(self, dst=None):
        if dst is None:
            dst = self.filepath
        
        with open(dst, 'w', encoding='utf8') as f:
            f.writelines(self.data)
    
    def to_labelImg(self, dst=None):
        if dst is None: dst = os.path.splitext(self.imgpath)[0] + '.xml'
        xf = LabelImgXML(dst, check_exist=False)
        xf.from_(self.imgpath, self.__rects)
        return xf
    
class VisMOT(TXTFile):
    def __init__(self, filepath, imgfolder, check_exist=True, all_labels=None):
        super().__init__(filepath, check_exist=check_exist)
        self.imgfolder = imgfolder
        self.all_labels = all_labels
    
    def parse(self):
        self.imgfile_dict = {} # {frame_idx: img_filepath}
        for f in os.listdir(self.imgfolder):
            fname,ext = os.path.splitext(f)
            if ext.lower() in ['.png','.jpg','.jpeg']:
                self.imgfile_dict[int(fname)] = os.path.join(self.imgfolder, f)
        
        self.shape_dict = {}
        with tqdm(self.lines, leave=False, desc='parsing') as pbar:
            for l in pbar:
                fidx,identity,x1,y1,w,h,conf,lid,_,_ = l.replace(' ','').split(',')
                label = self.all_labels[int(lid)] if self.all_labels else str(lid)
                rect = Rectangle(x1,y1,w,h, format='xywh', label=label)
                
                fidx = int(fidx)
                if fidx in self.shape_dict:
                    self.shape_dict[fidx].append(rect)
                else:
                    self.shape_dict[fidx] = [rect]
