import os
import json
import xml.etree.cElementTree as ET
import cv2

from .base import AppBase
from .image import ImageFile
from .shape import Rectangle

class LabelImgXML(AppBase):
    def __init__(self, filepath, check_exist=True):
        super().__init__(filepath, check_exist)
    
    def parse(self):
        tree = ET.parse(self.filepath)
        root = tree.getroot()
        
        self.__rects = []
        self._sh_dict = {'rectangle': self.__rects}
        for obj in root.findall('object'):
            label = obj.find('name').text
            pts = [int(obj.find('bndbox').find(pt).text) for pt in ['xmin','ymin','xmax','ymax']]
            rect = Rectangle(label, *pts, format='xyxy')
            # rect.pose = obj.find('pose').text
            # rect.truncated = int(obj.find('truncated').text)
            # rect.difficult = int(obj.find('difficult').text)
            
            self.__rects.append(rect)
    
    def from_(self, img_path, shapes):
        img = ImageFile(img_path)
    
    def save(self):
        raise NotImplementedError
        