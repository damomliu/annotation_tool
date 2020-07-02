import base64
import io
import json
import os

import numpy as np
# import PIL.Image
import cv2

from .base import FilePath
from .array import Array
from .shape import Rectangle

class ImageFile(FilePath):
    def __init__(self, filepath, check_exist=True):
        super().__init__(filepath, check_exist=check_exist)
    
    def read(self):
        self._img = Image(from_file=self.filepath)
    
    def __read(self):
        if '_img' not in self.__dict__: self.read()
    
    @property
    def img(self):
        self.__read()
        return self._img
    
    @property
    def im(self):
        """
            return a PIL.Image.Image object
        """
        self.__read()
        return self.img.PIL_im
    
    def __read_imageData(self):
        print('read imageData')
        with open(self.filepath, 'rb') as f:
            imgData = f.read()
            imgData = base64.b64encode(imgData).decode('utf-8')
        self._imageData = imgData
    
    @property
    def imageData(self):
        if '_imageData' not in self.__dict__:
            self.__read_imageData()
            return 'read', self._imageData
        else:
            return self._imageData
    
    @property
    def w(self):
        self.__read()
        return self.img.w
    
    @property
    def h(self):
        self.__read()
        return self.img.h
    
    @property
    def c(self):
        self.__read()
        return self.img.c
        # shape = np.array(self.im).shape
        # if len(shape) >= 3:
        #     return shape[2]
        # else:
        #     return None
    
    @property
    def shape(self):
        self.__read()
        return self.img.shape
    
    @property
    def rgb(self):
        self.__read()
        return self.img.cvt_color('rgb').numpy
        # return np.array(self.im)
    
    @property
    def bgr(self):
        self.__read()
        return self.img.cvt_color('bgr').numpy
        # array = np.array(self.im)
        # return array[:,:,::-1]

    @property
    def iaa(self):
        return {'image': self.rgb}
    
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
        Image().c / Image().w / Image().h / Image().n
    
    Method:
        - Image().expand_dim(axis=0): exploit np.expand_dim() on Image().numpy and change corresponding Image().format
        - Image().squeeze(): exploit np.squeeze() on Image().numpy and change corresponding Image().format
        - Image().as_format(new_format): tranpose Image().numpy as new_format and change corresponding Image().format
        
        - Image().crop(rectangle): return copied Image() with cropped array, given by *reactangle*
    """
    def __init__(self, array=[], format=None, color=None, from_file=None,
                 resized=1,
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
        self.resized = resized
        self.format = format
        self.color = color
        # self.preprocessed = preprocessed
    
    def __repr__(self):
        return f'<obj.{self.__class__.__name__} {self.w}x{self.h}, {self.format}/{self.color}>'
    
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
    
    def __read_im(self):
        print('read im')
        import PIL.Image
        rgb = self.cvt_color('rgb', inplace=False)
        rgb.as_format('hwc')
        self._im = PIL.Image.fromarray(rgb.numpy)
    
    @property
    def PIL_im(self):
        if '_im' not in self.__dict__:
            self.__read_im()
            return 'read', self._im
        else:
            return self._im
    
    def __create_normalized(self):
        if 0<=self.numpy.min() and self.numpy.max()<=1:
            array = self.numpy
        else:
            array = self.numpy / 255.
        
        self._normalized = Image(array, format=self.format, color=self.color, resized=self.resized)
    
    @property
    def normalized(self):
        if not hasattr(self, '_normalized'): self.__create_normalized()
        return self._normalized
    
    def copy(self):
        return Image(array=self.numpy,
                        resized=self.resized,
                        format=self.format,
                        color=self.color,
                        copy=True,
                        )
    
    def resize(self, newsize, inplace=False):
        """
            newsize = 'int' or '(w,h)'
        """
        if isinstance(newsize, int):
            newsize = (newsize,) *2
        else:
            newsize = tuple(newsize)
        if newsize != (self.w, self.h):
            dst = self if inplace else self.copy()
            oriColor = dst.color
            oriFormat = dst.format
            dst.as_format('hwc', inplace=True)
            
            r = (newsize[1]/self.w, newsize[0]/self.h)
            resized_array = cv2.resize(dst.numpy, newsize)
            # dst = Image(resized_array, format='hwc', color=oriColor, resized=r)
            dst.numpy = resized_array
            dst.resized = r
            dst.as_format(oriFormat, inplace=True)
            
            if not inplace: return dst
        
        else:
            if not inplace: return self.copy()

    def pad2asp_ratio(self, asp_ratio, inplace=False):
        dst = self if inplace else self.copy()
        dst.squeeze(1)
        dst.as_format('hwc', 1)
        
        wh = self.w / self.h
        if wh > asp_ratio:
            pad_w = self.w
            pad_h = int(self.w / asp_ratio)
        else:
            pad_h = self.h
            pad_w = int(self.h * asp_ratio)
        
        bg = np.zeros((pad_h,pad_w,self.c), dtype=self.numpy.dtype)
        bg[:self.h, :self.w, :] = self.numpy
        dst.numpy = bg
        
        if not inplace: return dst        
    
    def crop(self, rectangle, inplace=False):
        assert isinstance(rectangle, Rectangle)
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
        
        dst = self if inplace else self.copy()
        dst.numpy = dst.numpy[slices]
        
        if not inplace: return dst
    
    def expand_dim(self, axis=0, inplace=False):
        dst = self if inplace else self.copy()
        if dst.n is None:
            dst.numpy = np.expand_dims(dst.numpy, axis)
            
            _f = list(dst.format)
            _f.insert(axis, 'n')
            dst.format = ''.join(_f)
        
        if not inplace: return dst
    
    def squeeze(self, inplace=False):
        dst = self if inplace else self.copy()
        if 'n' in dst.format:
            dst.numpy = np.squeeze(dst.numpy, axis=dst.format.index('n'))
            dst.format = dst.format.replace('n', '')
        
        if not inplace: return dst
    
    def as_format(self, new_format, inplace=False):
        assert set(self.format)==set(new_format), f'cannot convert from "{self.format}" to "{new_format}"'
        
        dst = self if inplace else self.copy()
        
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
        dst = self if inplace else self.copy()
        
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