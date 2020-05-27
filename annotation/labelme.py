import os
import json
import xml.etree.cElementTree as ET
from xml.dom import minidom
import codecs
import cv2

from .base import AppBase
from .image import ImageFile
from .shape import Rectangle, Point

__version__ = '4.2.7'

class LabelmeJSON(AppBase):
    def __init__(self, filepath, check_exist=True):
        super().__init__(filepath, check_exist=check_exist)
    
    def parse(self):
        with open(self.filepath, 'r') as f:
            self.data = json.load(f)
        
        self._sh_dict = {}
        for sh in self.data['shapes']:
            sh_type = sh.get('shape_type')
            pts = sh.get('points')
            label = sh.get('label')
            if sh_type=='rectangle':
                newshape = Rectangle(label,
                                     *(pts[0] + pts[1]),
                                     format='xyxy')
            elif sh_type=='point':
                newshape = Point(label, *pts[0])
            else:
                print(f'!!WARNING!! unknown shape_type in [{self.filepath}]')
                continue
            
            newshape.group_id = sh.get('group_id')
            newshape.flags = sh.get('flags')
            
            if sh_type in self._sh_dict:
                self._sh_dict[sh_type].append(newshape)
            else:
                self._sh_dict[sh_type] = [newshape]
    
    def from_(self, img_path, shapes, flags=None):
        img = ImageFile(img_path)
        _shapes = [sh.labelme() for sh in shapes]
        
        self.data = dict(
            version=__version__,
            flags=flags if flags else {},
            shapes=_shapes,
            imagePath=img.filename,
            imageData=img.imageData,
            imageHeight=img.height,
            imageWidth=img.width,
        )
    
    def save(self, dst=None):
        if dst is None:
            dst = self.filepath
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f)
