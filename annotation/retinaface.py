import os
from tqdm import tqdm

from .base import TXTFile
from .shape import Rectangle, Point
from .image import ImageFile

class RetinaFaceTXT(TXTFile):
    def __init__(self, filepath, check_exist=True):
        super().__init__(filepath, check_exist=check_exist)
        
    def parse(self):
        self.img_filenames = []
        self.shape_dict = {}
        # shape_dict = {'filename': ['list','of','shapes']}
        
        with tqdm(self.lines) as pbar:
            pbar.set_description_str('Parsing...')
            for line in pbar:
                if line[0]=='#':
                    filename = line[2:-1]
                    self.img_filenames.append(filename)
                    self.shape_dict[filename] = []
                else:
                    filename = self.img_filenames[-1]
                    self.shape_dict[filename].extend(RetinaFaceLine(line).shapes)
                    
                    # line = x1,y1  w,h l0(x,y,visible) l1(x,y,v) l2(x,y,v) l3(x,y,v),..., blur \n
                    # annots = line.rstrip().split(' ')
                    # if len(annots)==4:
                    #     bbox = Rectangle('f', *annots[:4])
                    #     shapes.append(bbox)
                    # else:
                    #     bbox = Rectangle('f', *annots[:4])
                    #     bbox.blur = annots[-1]
                    #     shapes.append(bbox)
                        
                    #     n_point = int((len(annots) -4 -1) / 3)
                    #     for n in range(n_point):
                    #         idx = 4 + 3*n
                    #         pts = annots[idx:idx+3]
                    #         label = f'l{n}'
                    #         shapes.append(Point(label, *pts))
                
                pbar.set_postfix({'image_count': len(self.img_filenames)})
    
    def to_labelme(self, img_root=None):
        if img_root is None:
            img_root = os.path.join(self.dirname, 'images')
        
        if 'shape_dict' not in self.__dict__.keys():
            self.parse()
        
        with tqdm(self.shape_dict.items()) as pbar:
            pbar.set_description_str('Save as labelme format (.json)')
            for imgname,shapes in pbar:
                imgpath = os.path.join(img_root, imgname)
                imgFile = ImageFile(imgpath)
                imgFile.save_labelme_json(shapes)

class RetinaFaceLine():
    def __init__(self, line):
        self.line = line.strip()
        self.annots = self.line.split(' ')
        
        self.valid = True
        for ann in self.annots:
            try:
                float(ann)
            except ValueError:
                self.valid = False
                error_cannot_cvt_float()

    @property
    def shapes(self):
        if self.valid:
            shapes = []
            
            # line = x1,y1  w,h l0(x,y,visible) l1(x,y,v) l2(x,y,v) l3(x,y,v),..., blur \n
            annots = self.annots
            if len(annots)==4:
                bbox = Rectangle('f', *annots[:4])
                shapes.append(bbox)
            else:
                bbox = Rectangle('f', *annots[:4])
                bbox.blur = annots[-1]
                shapes.append(bbox)
                
                n_point = int((len(annots) -4 -1) / 3)
                for n in range(n_point):
                    idx = 4 + 3*n
                    pts = annots[idx:idx+3]
                    label = f'l{n}'
                    shapes.append(Point(label, *pts))
            
            return shapes
        
        else:
            error_cannot_cvt_float()

def error_cannot_cvt_float():
    print('!!ERROR!! cannot convert to float')
    raise ValueError