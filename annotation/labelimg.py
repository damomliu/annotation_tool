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
    
    def to_labelme(self, img_path, json_path=None):
        img = ImageFile(img_path)
        if json_path is None:
            fname = os.path.splitext(img_path)[0]
            json_path = fname + '.json'
            
    def from_(self):
        raise NotImplementedError
    
    def save(self):
        raise NotImplementedError
        

class LabelImgPair():
    def __init__(self, img_path, xml_path=None, check_exist=True):
        self.img = ImageFile(img_path, check_exist)
        if xml_path is None:
            fname = os.path.splitext(img_path)[0]
            xml_path = fname + '.xml'
        self.xml = LabelImgXML(xml_path, check_exist)
    
    def to_labelme_json(self, json_path=None):
        if json_path is None:
            fname,_ = os.path.splitext(self.img.filepath)
            json_path = fname + '.json'
        
        

class LabelImgPair_(ImageFile, LabelImgXML):
    def __init__(self, img_path, xml_path=None, check_exist=True):
        ImageFile.__init__(self, img_path, check_exist)
        if xml_path is None:
            fname = os.path.splitext(img_path)[0]
            xml_path = fname + '.xml'
        LabelImgXML.__init__(self, xml_path, check_exist)

def json_generator(file, output_path, labels, xmin, ymin, xmax, ymax):
    """
        originally contributed by Johnny
        adapted OOP-wise
    """ 
    data = {}
    shapes = []
    
    img = cv2.imread(file)
    height, width, depth = img.shape
    
    basename = os.path.basename(file)
    output_json = os.path.join(output_path, basename)[:-3] + "json"
    
    with open(file, mode='rb') as d_file:
        img_str = base64.b64encode(d_file.read()).decode('utf-8')

    for i in range(len(labels)):
        shape_data = {}
        shape_data["label"] = labels[i]
        shape_data["points"] = [[xmin[i], ymin[i]], [xmax[i], ymax[i]]]
        shape_data["group_id"] = None
        shape_data["shape_type"] = "rectangle"
        shape_data["flags"] = {}
        shapes.append(shape_data)
        
    data["version"] = "4.2.10"
    data["flag"] = {}
    data["shapes"] = shapes
    data["lineColor"] = [0, 255, 0, 128]
    data["fillColor"] = [255, 0, 0, 128]
    data["imagePath"] = basename
    data["imageData"] = img_str
    data["imageHeight"] = height
    data["imageWidth"] = width
    
    jsObj = json.dumps(data, indent=2)  
    fileObject = open(output_json, 'w')  
    fileObject.write(jsObj)  
    fileObject.close()