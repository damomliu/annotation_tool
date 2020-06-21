import os
import json
import xml.etree.cElementTree as ET
from xml.dom import minidom
import codecs

import cv2
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug.augmentables.polys import PolygonsOnImage

from .appbase import AppBase
from .image import ImageFile
from .shape import Rectangle, Point, Polygon
from .labelimg import LabelImgXML

__version__ = '4.2.7'

class LabelmeJSON(AppBase):
    def __init__(self, filepath, check_exist=True):
        super().__init__(filepath, check_exist=check_exist)
    
    def parse(self):
        with open(self.filepath, 'r', encoding="utf8", errors='ignore') as f:
            self.data = json.load(f)
        
        self._sh_dict = {'point':[],
                         'rectangle':[],
                         'polygon':[]}
        for sh in self.data['shapes']:
            sh_type = sh.get('shape_type')
            pts = sh.get('points')
            label = sh.get('label')
            if sh_type=='rectangle':
                newshape = Rectangle(*(pts[0] + pts[1]),
                                     label=label,
                                     format='xyxy')
            elif sh_type=='point':
                newshape = Point(*pts[0], label=label)
            elif sh_type=='polygon':
                pts_flat = []
                for pt in pts:
                    pts_flat.extend(pt)
                newshape = Polygon(*pts_flat, label=label)
            else:
                raise TypeError
            
            newshape.group_id = sh.get('group_id')
            newshape.flags = sh.get('flags')
            
            if sh_type in self._sh_dict:
                self._sh_dict[sh_type].append(newshape)
            else:
                self._sh_dict[sh_type] = [newshape]
    
    def __parse(self):
        if 'data' not in self.__dict__: self.parse()
    
    @property
    def imgpath(self):
        self.__parse()
        imgname = self.data.get('imagePath')
        imgfolder = os.path.dirname(self.filepath)
        return os.path.join(imgfolder, imgname)
    
    @property
    def imgw(self):
        self.__parse()
        return self.data.get('imageWidth')
    
    @property
    def imgh(self):
        self.__parse()
        return self.data.get('imageHeight')
    
    # def __get_imgfile(self):
    #     self.__imgfile = ImageFile(self.imgpath)
    
    # @property
    # def imgfile(self):
    #     if f'_{self.__class__.__name__}__imgfile' not in self.__dict__: self.__get_imgfile()
    #     return self.__imgfile 
    
    def from_(self, img_path, shapes=None, flags=None):
        img = ImageFile(img_path)
        if shapes is None: shapes = self.shapes
        
        _shapes = [sh.labelme() for sh in shapes]
        
        self.data = dict(
            version=__version__,
            flags=flags if flags else {},
            shapes=_shapes,
            imagePath=img.filename,
            imageData=img.imageData,
            imageHeight=img.h,
            imageWidth=img.w,
        )
    
    def from_array(self, rgb, imgpath, shapes, flags=None):
        _shapes = [sh.labelme() for sh in shapes]
        self.data = dict(
            version=__version__,
            flags=flags if flags else {},
            shapes=_shapes,
            imagePath=os.path.basename(imgpath),
            imageData=None,
            imageHeight=rgb.shape[0],
            imageWidth=rgb.shape[1],
        )
    
    def save(self, dst=None):
        if dst is None:
            dst = self.filepath
        with open(dst, 'w', encoding='utf8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def to_labelImg(self, poly2rect=False, poly2rect_labels=None, xml_path=None):
        rects = [sh for sh in self.shape_dict['rectangle']]
        
        if poly2rect_labels is None:
            poly2rect_labels = self.labels
        
        if poly2rect:
            for poly in self.shape_dict['polygon']:
                if poly.label in poly2rect_labels:
                    rects.append(poly.as_rectangle)
                else:
                    print(f'polygon label [{poly.label}] will not be converted, filename = [{self.filepath}]')
        
        if xml_path is None:
            xml_path = self.filepath.replace('.json', '.xml')
        xml = LabelImgXML(xml_path, check_exist=False)
        xml.from_(img_path=self.imgpath, shapes=rects)
        
        return xml
    
    @property
    def iaa(self):
        
        pts = [pt.iaa for pt in self.shape_dict['point']]
        boxes = [box.iaa for box in self.shape_dict['rectangle']]
        polys = [poly.iaa for poly in self.shape_dict['polygon']]
        
        imgshape = (self.imgh, self.imgw)
        shapes_on_image = {
            'image': self.imgfile.rgb,
            'keypoints': KeypointsOnImage(pts, imgshape),
            'bounding_boxes': BoundingBoxesOnImage(boxes, imgshape),
            'polygons': PolygonsOnImage(polys, imgshape),
        }
        
        return shapes_on_image
    