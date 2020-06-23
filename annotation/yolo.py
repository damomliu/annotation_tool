import cv2
from imgaug.augmentables.bbs import BoundingBoxesOnImage

from .appbase import AppBase
from .image import ImageFile
from .shape import Rectangle, Polygon

class YoloTXT(AppBase):
    def __init__(self, filepath, imgpath=None, check_exist=True, labels=None):
        super().__init__(filepath, check_exist=check_exist)
        
        if imgpath:
            self.imgpath = imgpath
            self._imgfile = ImageFile(self.imgpath, check_exist=check_exist)
        
        self._labels = labels
        if self.labels:
            self.lidx = {l:i for i,l in enumerate(self.labels)}
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def imgw(self):
        return self.imgfile.w

    @property
    def imgh(self):
        return self.imgfile.h
    
    def parse(self):
        with open(self.filepath, 'r', encoding="utf8", errors='ignore') as f:
            self.data = f.readlines()
        
        self.__rects = []
        self._sh_dict = {'rectangle': self.__rects}
        for l in self.data:
            lid,xc,yc,w,h = l.split(' ')
            
            w = float(w) * self.imgw
            h = float(h) * self.imgh
            xc = float(xc) * self.imgw
            yc = float(yc) * self.imgh
            x1 = xc - 0.5*w
            y1 = yc - 0.5*h
            
            if self.labels:
                label = self.labels[lid]
            else:
                label = str(lid)
            
            rect = Rectangle(x1,y1,w,h, format='xywh', label=label)
            self.__rects.append(rect)
    
    def __parse(self):
        if 'data' not in self.__dict__: self.parse()
    
    def from_(self, img_path, shapes):
        raise NotImplementedError
    
    def from_array(self, rgb, shapes, **kwargs):
        """
            rgb : ndarray with shape (w,h,c)
        """
        h,w = rgb.shape[:2]
        
        self.data = []
        for sh in shapes:
            if isinstance(sh, (Rectangle, Polygon)):
                if isinstance(sh, Polygon): sh = sh.as_rectangle
                xc = (sh.x1 + sh.x2) /2 /w
                yc = (sh.y1 + sh.y2) /2 /h
                rectw = abs(sh.x2 - sh.x1) /w
                recth = abs(sh.y2 - sh.y1) /h
                lid = self.lidx[sh.label]
                
                line = f'{lid} {xc} {yc} {rectw} {recth}\n'
                self.data.append(line)
                
    def save(self, dst=None):
        if dst is None:
            dst = self.filepath
        
        with open(dst, 'w', encoding='utf8') as f:
            f.writelines(self.data)
    
    @property
    def iaa(self):
        boxes = [box.iaa for box in self.shape_dict['rectangle']]
        imgshape = (self.imgh, self.imgw)
        shapes_on_image = {
            'image': self.imgfile.rgb,
            'bounding_boxes': BoundingBoxesOnImage(boxes, imgshape),
        }
        return shapes_on_image