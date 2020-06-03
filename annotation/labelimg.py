import os
import json
import xml.etree.cElementTree as ET
from xml.dom import minidom
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
    
    def from_(self, img_path, shapes=None, database='unknown', **kwargs):
        img = ImageFile(img_path)
        if shapes is None: shapes = self.shapes
        
        annotation = ET.Element('annotation')
    
        ET.SubElement(annotation, "folder").text = img.foldername
        ET.SubElement(annotation, 'filename').text = img.filename
        ET.SubElement(annotation, 'path').text = img.filepath
        
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = database
        
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(img.w)
        ET.SubElement(size, "height").text = str(img.h)
        ET.SubElement(size, "depth").text = str(img.c)
        
        for k,v in kwargs.items():
            ET.SubElement(annotation, str(k)).text = str(v)
        
        for sh in shapes:
            annotation.append(sh.labelimg())
        
        self.data = annotation
    
    def save(self, dst=None):
        if dst is None:
            dst = self.filepath
        
        _rough_string = ET.tostring(self.data, encoding='utf-8')
        _reparsed = minidom.parseString(_rough_string)
        with open(dst, 'w', encoding='utf-8') as f:
            _reparsed.writexml(f, encoding='utf-8', addindent='    ', newl='\n')
        