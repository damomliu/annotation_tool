import os

from .appbase import AppBase
from .image import ImageFile
from .shape import Rectangle, Polygon
from .labelimg import LabelImgXML

class JdeTXT(AppBase):
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
            lid,identity,xc,yc,w,h = l.replace('  ', ' ').split(' ')
            
            w = float(w) * self.imgw
            h = float(h) * self.imgh
            xc = float(xc) * self.imgw
            yc = float(yc) * self.imgh
            x1 = xc - 0.5*w
            y1 = yc - 0.5*h
            
            if self.all_labels:
                label = self.all_labels[lid]
            else:
                label = str(lid)
            
            rect = Rectangle(x1,y1,w,h, format='xywh', label=identity)
            rect.id = identity
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
    