import os
from pathlib import Path
from abc import ABCMeta, abstractmethod, abstractproperty

__all__ = ['FilePath', 'TXTFile']

class FilePath():
    def __init__(self, filepath, check_exist=True):
        self.__filepath = filepath
        self.dirname, self.filename = os.path.split(filepath)
        self.fname, self.ext = os.path.splitext(self.filename)
        self.foldername = os.path.basename(self.dirname)
        if check_exist and not os.path.exists(filepath):
            print(f'!!Warning!! filepath not exists [{filepath}]')
            if os.path.isdir(filepath):
                print(f'!!Warning!! is a directory [{filepath}]')
    
    def __repr__(self):
        return f'<{self.__class__.__name__} @{os.path.basename(self.__filepath)}>'
    
    @property
    def filepath(self):
        return self.__filepath
    
    @property
    def path(self):
        return Path(self.__filepath)

class TXTFile(FilePath):
    def __init__(self, filepath, check_exist=True):
        super().__init__(filepath, check_exist)
        try:
            with open(self.filepath, 'r') as f:
                self.lines = f.readlines()
        except IOError:
            print(f'!!Failed!! reading [{self.filepath}]')

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