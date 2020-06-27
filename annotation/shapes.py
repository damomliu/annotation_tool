import numpy as np
from imgaug.augmentables.kps import Keypoint
from imgaug.augmentables.bbs import BoundingBox
from imgaug.augmentables.polys import Polygon as iaaPolygon
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug.augmentables.polys import PolygonsOnImage

from .shape import Point,Rectangle,Polygon

class ShapesOnImage:
    def __init__(self, from_iaa=None, shapes=None, image=None):
        """
        [parameters]
        - from_iaa: (1) :tuple: the returned values of iaa.augmenter.augment()
                    (2) :dict:  with keys "keypoints" / "image" / ...
                       
        """
        assert shapes is not None or from_iaa is not None, 'Either of "shapes" / "from_iaa" must be given'
        if from_iaa:
            if isinstance(from_iaa, (tuple,list)):
                self.image = from_iaa[0]
                self.__init_soi()
                for rst in from_iaa[1:]:
                    self.extend(rst.items)
            elif isinstance(from_iaa, dict):
                self.image = from_iaa.get('image')
                self.__init_soi()
                for k,rst in from_iaa.items():
                    if k!='image':
                        self.extend(rst.items)
            else:
                raise TypeError

        else:
            self.image = image
            self.__init_soi()
            self.extend(shapes)
        
    def __init_soi(self):
        imgshape = self.image.shape
        self.__soi = {
            'keypoints': KeypointsOnImage([], imgshape),
            'bounding_boxes': BoundingBoxesOnImage([], imgshape),
            'polygons': PolygonsOnImage([], imgshape),
        }
    
    def __repr__(self):
        rep = '<ShapesOnImage'
        if self.image is not None:
            rep += f' {self.image.shape[2:]}'
        for k,soi in self.__soi.items():
            if soi.items:
                rep += f' {k}*{len(soi.items)}'
        rep += '>'
        return rep
    
    @property
    def dict(self):
        return self.__soi
    
    @property
    def imgshape(self):
        return self.image.shape
    
    @property
    def shape_dict(self):
        _sh_dict = {
            'point':[Point(from_iaa=kp) for kp in self.__soi['keypoints'].items],
            'rectangle':[Rectangle(from_iaa=rect) for rect in self.__soi['bounding_boxes'].items],
            'polygon':[Polygon(from_iaa=ply) for ply in self.__soi['polygons'].items]
        }
        return _sh_dict
    
    @property
    def shapes(self):
        return [sh for shapes in self.shape_dict.values() for sh in shapes]
    
    def append(self, shape):
        if isinstance(shape, (Point,Rectangle,Polygon)):
            shape = shape.iaa
        
        if isinstance(shape, Keypoint):
            self.__soi['keypoints'].items.append(shape)
        elif isinstance(shape, BoundingBox):
            self.__soi['bounding_boxes'].items.append(shape)
        elif isinstance(shape, iaaPolygon):
            self.__soi['polygons'].items.append(shape)
        else:
            raise TypeError
    
    def extend(self, shapes):
        for sh in shapes:
            self.append(sh)
    
    def draw_on_image(self, image=None, inplace=False):
        if image is None: image = self.image
        
        if inplace:
            dst = image
        else:
            dst = image.copy()
        
        for k,soi in self.dict.items():
            dst = soi.draw_on_image(dst)
        
        if not inplace: return dst
    
    def iaa(self, func, *args, **kwargs):
        for k,soi in self.dict.items():
            if func in soi.__dir__() and soi.items:
                soi = soi.__getattribute__(func)(*args, **kwargs)