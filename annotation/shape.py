from abc import ABCMeta, abstractmethod
import xml.etree.cElementTree as ET

class Shape(metaclass=ABCMeta):
    def __init__(self, label, *pts):
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
    def __init__(self, label, *pts, format='xywh'):
        super().__init__(label, *pts)
        
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
        return f'<shape.Rectangle pt1={self.pt1}, pt2={self.pt2}>'

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
    
    def iou(self, rectangle):
        assert isinstance(rectangle, Rectangle)
        
        # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        xA = max(self.x1, rectangle.x1)
        yA = max(self.y1, rectangle.y1)
        xB = min(self.x2, rectangle.x2)
        yB = min(self.y2, rectangle.y2)

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        iou = interArea / float(self.area_grid + rectangle.area_grid - interArea)

        return iou

class Point(Shape):
    def __init__(self, label, *pts):
        super().__init__(label, *pts)
        if len(pts) in [2,3]:
            if len(pts)==3:
                self.visible = pts[2]
            else:
                self.visible = None
        else:
            raise ValueError
    
    def __repr__(self):
        return f'<shpae.Point {self.pt1}>'
    
    def labelme(self, **kwargs):
        json = super().labelme_common(**kwargs)
        json['shape_type'] = 'point'
        json['points'] = [[self.x1,self.y1]]
        return json
