from abc import ABCMeta, abstractmethod
import xml.etree.cElementTree as ET

from imgaug.augmentables.kps import Keypoint
from imgaug.augmentables.bbs import BoundingBox
from imgaug.augmentables.polys import Polygon as iaaPolygon

class Shape(metaclass=ABCMeta):
    def __init__(self, *pts, label=None):
        self.label = label
        self.x1 = float(pts[0])
        self.y1 = float(pts[1])
    
    @property
    def pt1(self):
        return (int(self.x1), int(self.y1))
    
    @abstractmethod
    def labelme(self, group_id=None, flags={}):
        raise NotImplementedError
    
    def labelme_common(self, group_id=None, flags={}):
        return {'group_id': group_id,
                'flags': flags,
                'label': self.label
                }

class Rectangle(Shape):
    def __init__(self, *pts, format='xywh', label=None):
        super().__init__(*pts, label=label)
        
        if len(pts) in [4,5]:
            if format=='xywh':
                self.w = float(pts[2])
                self.h = float(pts[3])
                self.x2 = self.x1 + self.w
                self.y2 = self.y1 + self.h
            elif format=='xyxy':
                self.x2 = float(pts[2])
                self.y2 = float(pts[3])
                self.w = abs(self.x2 - self.x1)
                self.h = abs(self.y2 - self.y1)
            else:
                raise NotImplementedError

            if len(pts)==5:
                self.blur = pts[4]
            else:
                self.blur = None
        else:
            raise ValueError
    
    def __repr__(self):
        clsname = self.__class__.__name__
        _repr = f'<{clsname}'
        
        if self.label:
            _repr += f' [{self.label}]'
        else:
            _repr += f' {self.pt1}-{self.pt2}'
        
        _repr += '>'
        return _repr

    @property
    def pt2(self):
        return (int(self.x2), int(self.y2))
    
    @property
    def area(self):
        return self.w * self.h
    
    @property
    def area_grid(self):
        return (self.w +1) * (self.h +1)
    
    @property
    def central(self):
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        return (cx, cy)
    
    @property
    def central_int(self):
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        return (int(cx), int(cy))
    
    def labelme(self, **kwargs):
        json = super().labelme_common(**kwargs)
        json['shape_type'] = 'rectangle'
        json['points'] = [[self.x1,self.y1], [self.x2,self.y2]]
        return json
    
    def labelimg(self, **kwargs):
        obj = ET.Element('object')
        ET.SubElement(obj, 'name').text = str(self.label)
        for k,v in kwargs.items():
            ET.SubElement(obj, str(k)).text = str(v)
        
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(self.x1)
        ET.SubElement(bndbox, 'ymin').text = str(self.y1)
        ET.SubElement(bndbox, 'xmax').text = str(self.x2)
        ET.SubElement(bndbox, 'ymax').text = str(self.y2)
        return obj
    
    @property
    def as_polygon(self):
        """
            (x1,y1)     (x2,y1)
            (x1,y2)     (x2,y2)
        """
        pts = [self.x1, self.y1]
        pts.extend([self.x1, self.y2])
        pts.extend([self.x2, self.y2])
        pts.extend([self.x2, self.y1])
        return Polygon(*pts, label=self.label)
    
    @property
    def iaa(self):
        return BoundingBox(self.x1, self.y1, self.x2, self.y2, self.label)
    
    def intersect(self, rectangle):
        assert isinstance(rectangle, Rectangle)
        # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        xA = max(self.x1, rectangle.x1)
        yA = max(self.y1, rectangle.y1)
        xB = min(self.x2, rectangle.x2)
        yB = min(self.y2, rectangle.y2)
        
        return Rectangle('intersect', xA,yA, xB,yB, format='xyxy')
        
    def iou(self, rectangle):
        interArea = self.intersect(rectangle).area_grid
        iou = interArea / float(self.area_grid + rectangle.area_grid - interArea)

        return iou

class Point(Shape):
    def __init__(self, *pts, label=None):
        super().__init__(*pts, label=label)
        if len(pts) in [2,3]:
            if len(pts)==3:
                self.visible = pts[2]
            else:
                self.visible = None
        else:
            raise ValueError
    
    def __repr__(self):
        clsname = self.__class__.__name__
        _repr = f'<{clsname}'
        
        if self.label:
            _repr += f' [{self.label}]'
        else:
            _repr += f' {self.pt1}'
        
        _repr += '>'
        return _repr
    
    def labelme(self, **kwargs):
        json = super().labelme_common(**kwargs)
        json['shape_type'] = 'point'
        json['points'] = [[self.x1,self.y1]]
        return json
    
    @property
    def iaa(self):
        return Keypoint(self.x1, self.y1)


class Polygon(Shape):
    def __init__(self, *pts, label=None):
        assert (len(pts) +1) %2, 'Polygon must receive even number (2x) of points'
        super().__init__(*pts, label=label)
        
        self.points = []
        for i in range(len(pts) //2):
            if label:
                ptlabel = f'{label}_{i+1}'
            else:
                ptlabel = None
            self.points.append(Point(pts[2*i], pts[2*i +1], label=ptlabel))
    
    def __repr__(self):
        _rep =  f'<Polygon'
        if self.label:
            _rep += f' [{self.label}]'
        _rep += f'*{len(self.points)}pts>'
        return _rep
    
    @property
    def as_rectangle(self):
        xs = []
        ys = []
        for pt in self.points:
            xs.append(pt.x1)
            ys.append(pt.y1)
        rect_pts = min(xs),min(ys), max(xs),max(ys)
        return Rectangle(*rect_pts, format='xyxy', label=self.label)
    
    def labelme(self, group_id=None, flags={}):
        raise NotImplementedError
        return super().labelme(group_id=group_id, flags=flags)
    
    @property
    def iaa(self):
        pts = [(pt.x1,pt.y1) for pt in self.points]
        return iaaPolygon(pts, self.label)