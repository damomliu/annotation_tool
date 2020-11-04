from abc import ABCMeta, abstractmethod, abstractproperty
from .base import FilePath
from .image import ImageFile
from .shapes import ShapesOnImage

class AppBase(FilePath, metaclass=ABCMeta):
    @abstractmethod
    def parse(self):
        raise NotImplementedError
    
    @abstractmethod
    def from_(self, img_path, shapes):
        raise NotImplementedError
    
    @abstractmethod
    def save(self, dst=None):
        raise NotImplementedError
    
    @property
    def shape_dict(self):
        # shape_dict = {'shape_type': ['list','of','shapes]}
        if '_sh_dict' not in self.__dict__:
            self.parse()
        return self._sh_dict

    def __get_shapes(self):
        self._shapes = [shape for sh_list in self.shape_dict.values() for shape in sh_list]
    
    @property
    def shapes(self):
        if '_shapes' not in self.__dict__: self.__get_shapes()
        return self._shapes
    
    @property
    def labels(self):
        labels = [sh.label for sh in self.shapes]
        return sorted(list(set(labels)))
    
    def __get_imgfile(self):
        self._imgfile = ImageFile(self.imgpath)
    
    @property
    def imgfile(self):
        if f'_imgfile' not in self.__dict__: self.__get_imgfile()
        return self._imgfile
    
    @property
    def imgw(self):
        return self.imgfile.w

    @property
    def imgh(self):
        return self.imgfile.h
    
    @property
    def soi(self):
        return ShapesOnImage(from_iaa=self.iaa)