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
    
    def from_(self, img_path, shapes=None, flags=None):
        img = ImageFile(img_path)
        if shapes is None:
            shapes = self._sh_dict
        shapes = [sh.labelme() for sh in shapes]
        
        self.data = dict(
            version=__version__,
            flags=flags if flags else {},
            shapes=shapes,
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

class LabelmePair():
    def __init__(self, img_path, json_path=None, check_exist=True):
        self.img = ImageFile(img_path, check_exist)
        if json_path is None:
            fname = os.path.splitext(img_path)[0]
            json_path = fname + '.json'
        self.json = LabelmeJSON(json_path, check_exist)
    
    def to_labelimg(self, xml_path=None):
        if xml_path is None:
            fname,_ = os.path.splitext(self.img.filepath)
            xml_path = fname + '.xml'
        
        labels, xmins,ymins, xmaxs,ymaxs = [],[],[],[],[]
        for sh in self.shape_dict['rectangle']:
            xmins.append(sh.x1)
            ymins.append(sh.y1)
            xmaxs.append(sh.x2)
            ymaxs.append(sh.y2)
            labels.append(sh.label)
        
        xml_generator(self.img.filepath, xml_path, labels, xmins, ymins, xmaxs, ymaxs)
    
    def to_labelimg_pair(self, xml_path=None):
        if xml_path is None:
            fname,_ = os.path.splitext(self.img.filepath)
            xml_path = fname + '.xml'
        
        
class LabelmePair_(ImageFile, LabelmeJSON):
    def __init__(self, img_path, json_path=None, check_exist=True):
        ImageFile.__init__(self, img_path, check_exist)
        if json_path is None:
            fname = os.path.splitext(img_path)[0]
            json_path = fname + '.json'
        LabelmeJSON.__init__(self, json_path, check_exist)
    
    def to_labelimg(self, xml_path=None):
        super().to_labelimg()

def xml_generator(file, output_xml, labels, xmin, ymin, xmax, ymax):
    """
        originally contributed by Johnny
        adapted OOP-wise
    """ 
    img = cv2.imread(file)
    height, width, depth = img.shape
    
    basename = os.path.basename(file)
    # output_xml = os.path.join(output_path, basename)[:-3] + "xml"
    
    # start to generate xml
    annotation = ET.Element('annotation')
    
    ET.SubElement(annotation, "folder").text = os.path.dirname(file)
    ET.SubElement(annotation, 'filename').text = os.path.basename(file)
    ET.SubElement(annotation, 'path').text = os.path.abspath(file)
    
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"
    
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(annotation, "segmented").text = "0"
    
    for i in range(len(labels)):
        obj_box = ET.SubElement(annotation, "object")
        ET.SubElement(obj_box, "name").text = labels[i]
        ET.SubElement(obj_box, "pose").text = "Unspecified"
        ET.SubElement(obj_box, "truncated").text = "0"
        ET.SubElement(obj_box, "difficult").text = "0"

        bndbox = ET.SubElement(obj_box, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(xmin[i]))
        ET.SubElement(bndbox, "ymin").text = str(int(ymin[i]))
        ET.SubElement(bndbox, "xmax").text = str(int(xmax[i]))
        ET.SubElement(bndbox, "ymax").text = str(int(ymax[i]))

    tree = ET.ElementTree(annotation)
    tree.write(output_xml)
    
    #format xml file
    dom = minidom.parse(output_xml)
    f = codecs.open(output_xml, 'w', 'utf-8') 
    dom.writexml(f, addindent='  ', newl='\n', encoding = 'utf-8')  
    f.close()