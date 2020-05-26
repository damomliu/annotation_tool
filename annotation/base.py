import os
from pathlib import Path

__all__ = ['FilePath', 'TXTFile']

class FilePath():
    def __init__(self, filepath, check_exist):
        self.__filepath = filepath
        self.dirname, self.filename = os.path.split(filepath)
        if check_exist and not os.path.exists(filepath):
            print(f'!!Warning!! filepath not exists [{filepath}]')
    
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