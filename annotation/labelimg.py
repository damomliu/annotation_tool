import os
import xml.etree.cElementTree as ET

from .base import TXTFile
from .image import ImageFile
from .shape import Rectangle

class LabelImgXML(TXTFile):
    def __init__(self, filepath, check_exist=True):
        super().__init__(filepath, check_exist)
    
    @property
    def shape_dict(self):
        # shape_dict = {'shape_type': ['list','of','shapes]}
        if '_LabelImgXML__sh_dict' not in self.__dict__:
            self.parse()
        return self.__sh_dict
    
    @property
    def shapes(self):
        return [shape for sh_list in self.shape_dict.values() for shape in sh_list]
    
    def parse(self):
        tree = ET.parse(self.filepath)
        root = tree.getroot()
        
        self.__rects = []
        self.__sh_dict = {'rectangle': self.__rects}
        for obj in root.findall('object'):
            label = obj.find('name').text
            pts = [int(obj.find('bndbox').find(pt).text) for pt in ['xmin','ymin','xmax','ymax']]
            rect = Rectangle(label, *pts, format='xyxy')
            # rect.pose = obj.find('pose').text
            # rect.truncated = int(obj.find('truncated').text)
            # rect.difficult = int(obj.find('difficult').text)
            
            self.__rects.append(rect)
    
    def to_labelme(self):
        raise NotImplementedError

class LabelImgPair():
    def __init__(self, img_path, xml_path=None, check_exist=True):
        self.img = ImageFile(img_path, check_exist)
        if xml_path is None:
            fname = os.path.splitext(img_path)[0]
            xml_path = fname + '.xml'
        self.xml = LabelImgXML(xml_path, check_exist)

class LabelImgPair_(ImageFile, LabelImgXML):
    def __init__(self, img_path, xml_path=None, check_exist=True):
        ImageFile.__init__(self, img_path, check_exist)
        if xml_path is None:
            fname = os.path.splitext(img_path)[0]
            xml_path = fname + '.xml'
        LabelImgXML.__init__(self, xml_path, check_exist)
