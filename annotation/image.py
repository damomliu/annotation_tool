import base64
import io
import json
import os

import numpy as np
import PIL.Image

from .base import FilePath

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
    def width(self):
        return self.im.width
    
    @property
    def height(self):
        return self.im.height
    
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
            imageHeight=self.height,
            imageWidth=self.width,
        )
        
        if dst is None:
            fname = os.path.splitext(self.filepath)[0]
            dst = fname + '.json'
        with open(dst, 'w') as f:
            json.dump(data, f)
