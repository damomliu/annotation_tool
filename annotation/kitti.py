import os

from .appbase import AppBase
from .shape import Rectangle
from .labelimg import LabelImgXML

class KittiTXT(AppBase):
    def __init__(self, filepath, check_exist=True):
        super().__init__(filepath, check_exist=check_exist)
    
    def parse(self):
        self.__rects = []
        self._sh_dict = {'rectangle': self.__rects}
        with open(self.filepath,'r') as f:
            self.data = f.readlines()
            
        for line in self.data:
            linesplit = line.split(' ')
            rect = Rectangle(*linesplit[4:8], format='xyxy', label=linesplit[0])
            self.__rects.append(rect)
        
    def from_(self, img_path, shapes):
        raise NotImplementedError
    
    def save(self, dst=None):
        raise NotImplementedError
    
    def to_labelImg(self, imgpath, dst=None):
        if dst is None:
            fname = os.path.splitext(imgpath)[0]
            dst = fname+'.xml'

        xf = LabelImgXML(dst, check_exist=False)
        xf.from_(imgpath, shapes=self.shapes)
        
        return xf