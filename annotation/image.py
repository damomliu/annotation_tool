import base64
import io
import json
import os

import numpy as np
import PIL.Image
import cv2

from .base import FilePath
from .array import Array

class ImageFile(FilePath):
    def __init__(self, filepath, check_exist=True):
        FilePath.__init__(self, filepath, check_exist)
        try:
            self.im = PIL.Image.open(filepath)
        except IOError:
            print(f'!!Failed!! reading [{filepath}]')
    
    def __str__(self):
        return self.filepath
    
    @property
    def imageData(self):
        with open(self.filepath, 'rb') as f:
            imgData = f.read()
            imgData = base64.b64encode(imgData).decode('utf-8')
        return imgData
    
    @property
    def w(self):
        return self.im.width
    
    @property
    def h(self):
        return self.im.height
    
    @property
    def c(self):
        shape = np.array(self.im).shape
        if len(shape) >= 3:
            return shape[2]
        else:
            return None
    
    @property
    def rgb_array(self):
        return np.array(self.im)
    
    @property
    def bgr_array(self):
        array = np.array(self.im)
        return array[:,:,::-1]
    
    def save_labelme_json(self, shapes=[], dst=None, flags=None):
        __version__ = '4.2.7'
        shapes = [sh.labelme() for sh in shapes]
        data = dict(
            version=__version__,
            flags=flags if flags else {},
            shapes=shapes,
            imagePath=self.filename,
            imageData=self.imageData,
            imageHeight=self.h,
            imageWidth=self.w,
        )
        
        if dst is None:
            fname = os.path.splitext(self.filepath)[0]
            dst = fname + '.json'
        with open(dst, 'w') as f:
            json.dump(data, f)

class Image(Array):
    """Image object, an iheritance of Array object
    Image(self, array=[], format=None, color=None, resize=1, from_file=None, preprocessed=False)
    
    Property:
        Image().color
        Image().format
        Image().resize
        Image().preprocessed : bool
        Image().c / Image().w / Image().h / Image().n
    
    Method:
        - Image().expand_dim(axis=0): exploit np.expand_dim() on Image().numpy and change corresponding Image().format
        - Image().squeeze(): exploit np.squeeze() on Image().numpy and change corresponding Image().format
        - Image().as_format(new_format): tranpose Image().numpy as new_format and change corresponding Image().format
        
        - Image().crop(rectangle): return copied Image() with cropped array, given by *reactangle*
    """
    def __init__(self, array=[], format=None, color=None, from_file=None,
                 resize=1, preprocessed=False,
                 copy=False,
                 ):
        assert len(array) or from_file, 'must be initiated from either "array" or "from_file"'
        
        if from_file:
            assert os.path.exists(from_file), 'file not exist'
            array = cv2.imread(from_file)
            
            assert len(array), 'cv2.imread() returns an empty matrix'
            format = 'hwc'
            color = 'bgr'
        else:
            assert format is not None and color is not None, 'must specify format/color if initiated with an array'
        
        super().__init__(array, copy)
        self.resize = resize
        self.format = format
        self.color = color
        self.preprocessed = preprocessed
    
    def __repr__(self):
        return f'<obj.Image {self.w}x{self.h}, {self.format}, {self.color}>'
    
    @property
    def c(self):
        return self.shape[self.format.index('c')]
    
    @property
    def w(self):
        return self.shape[self.format.index('w')]
    
    @property
    def h(self):
        return self.shape[self.format.index('h')]
    
    @property
    def n(self):
        if 'n' in self.format:
            return self.shape[self.format.index('n')]
        else:
            return None
    
    @property
    def PIL_im(self):
        import PIL
        rgb = self.cvt_color('rgb', inplace=False)
        rgb.as_format('hwc')
        return PIL.Image.fromarray(rgb.numpy)
    
    def expand_dim(self, axis=0):
        assert self.n is None, f'self.n = {self.n}'
        
        self.numpy = np.expand_dims(self.numpy, axis)
        
        _f = list(self.format)
        _f.insert(axis, 'n')
        self.format = ''.join(_f)
    
    def copy(self):
        return Image(array=self.numpy,
                        resize=self.resize,
                        format=self.format,
                        color=self.color,
                        preprocessed=self.preprocessed,
                        copy=True,
                        )
    
    def crop(self, rectangle):
        slices = []
        for f in self.format:
            _slc = slice(None, None)
            if f=='w':
                _slc = [int(rectangle.x1), int(rectangle.x2)]
                _slc = _correct_crop_pts(_slc, upper=self.w)
            elif f=='h':
                _slc = [int(rectangle.y1), int(rectangle.y2)]
                _slc = _correct_crop_pts(_slc, upper=self.h)
            slices.append(_slc)
        
        croparray = self.copy()
        croparray.numpy = croparray.numpy[slices]
        
        return croparray
    
    def squeeze(self):
        if 'n' in self.format:
            self.numpy = np.squeeze(self.numpy, axis=self.format.index('n'))
            self.format = self.format.replace('n', '')
        
        return self
    
    def as_format(self, new_format, inplace=False):
        assert set(self.format)==set(new_format), f'cannot convert from "{self.format}" to "{new_format}"'
        
        if inplace:
            dst = self
        else:
            dst = self.copy()
        
        if new_format==self.format:
            pass
        else:
            new_f_idx = [dst.format.index(f) for f in new_format]
            dst.numpy = dst.numpy.transpose(new_f_idx)
            dst.format = new_format
            
        if not inplace:
            return dst
        
    def cvt_color(self, new_color, inplace=False):
        old_color = self.color
        if inplace:
            dst = self
        else:
            dst = self.copy()
        
        if new_color==old_color:
            cvt = None
        elif old_color=='rgb':
            if new_color=='bgr':
                cvt = cv2.COLOR_RGB2BGR
            else:
                raise NotImplementedError
        elif old_color=='bgr':
            if new_color=='rgb':
                cvt = cv2.COLOR_BGR2RGB
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        if cvt is not None:
            cv2.cvtColor(self.numpy, cvt, dst.numpy)
            dst.color = new_color
        
        if not inplace:
            return dst


def _correct_crop_pts(slc, upper, lower=0):
    for i,p in enumerate(slc):
        if p < lower:
            slc[i] = lower
        elif p > upper:
            slc[i] = upper
    return slice(*slc)