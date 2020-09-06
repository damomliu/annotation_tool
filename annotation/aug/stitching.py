import os
from random import shuffle

import cv2
import numpy as np
from tqdm import tqdm

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug.augmentables.polys import PolygonsOnImage

from ..augmenter import FolderAugmenter, ANNEXT


class StitchingMaker(FolderAugmenter):
    def __init__(self,
                 src_root, src_type,
                 size, cell_n,
                 walk=True,):
        
        assert len(size)==2
        assert len(cell_n)==2
        
        super().__init__(src_root, src_type, walk=walk)
        self.cell_nx, self.cell_ny = cell_n
        self.cw = int(size[0] / self.cell_nx)
        self.ch = int(size[1] / self.cell_ny)

        self.final_size = (self.cw * cell_n[0], self.ch * cell_n[1])
        self.final_shape = (self.ch * cell_n[1], self.cw * cell_n[0], 3)
    
        self.cell_count = len(self.relpaths) // (self.cell_nx * self.cell_ny)
        self.cell_shape = (self.cell_count, self.cell_ny, self.cell_nx)
    
    @property
    def cell_relpaths(self):
        if not hasattr(self,'_cell_relpaths'): self.shuffle()
        return self._cell_relpaths
    def shuffle(self):
        relpaths = self.relpaths.copy()
        shuffle(relpaths)
        relpaths = relpaths[:self.cell_count *self.cell_nx *self.cell_ny]
        
        self._cell_relpaths = np.reshape(relpaths, self.cell_shape)
        self._cell_idx_ravel = [self.relpaths.index(rp) for rp in relpaths]
        self._cell_idx = np.reshape(self._cell_idx_ravel, self.cell_shape)
    
    @property
    def cell_annpaths(self):
        if not hasattr(self,'_cell_annpaths'): self.__get_cell_annpaths()
        return self._cell_annpaths
    def __get_cell_annpaths(self):
        if self.src_type=='image':
            self._cell_annpaths = None
        else:
            paths = [os.path.join(self.src_root, self.relpaths[i]) for i in self._cell_idx_ravel]
            self._cell_annpaths = np.reshape(paths, self.cell_shape)
        
    @property
    def cell_imgpaths(self):
        if not hasattr(self,'_cell_imgpaths'): self.__get_cell_imgpaths()
        return self._cell_imgpaths
    def __get_cell_imgpaths(self):
        if self.src_type=='image':
            paths = [os.path.join(self.src_root, self.relpaths[i]) for i in self._cell_idx_ravel]
            self._cell_imgpaths = np.reshape(paths, self.cell_shape)
        else:
            self._cell_imgpaths = []
            paths = [self[i].imgpath for i in self._cell_idx_ravel]
            self._cell_imgpaths = np.reshape(paths, self.cell_shape)
    
    def make(self, dst_imgroot, dst_annroot=None, dst_anntype=None, cell_aug=None, prefix=None, overwrite=False):
        if dst_annroot is None: dst_annroot = dst_imgroot
        if dst_anntype is None: dst_anntype = self.src_type
        for folder in [dst_imgroot, dst_annroot]:
            if not os.path.isdir(folder): os.makedirs(folder)
        
        seq = iaa.Sequential([
            iaa.PadToAspectRatio(self.cw/self.ch, pad_mode=['mean', 'median', "edge", "linear_ramp"]),
            iaa.Resize({'height':self.ch, 'width':self.cw})
        ])
        if cell_aug is not None:
            seq.append(cell_aug)
        
        for i,cell in enumerate(tqdm(self._cell_idx, desc=f'Stitching {len(self.relpaths)} images into {self.cell_count}*({self.cell_nx}x{self.cell_ny})')):
            pane = np.zeros(self.final_shape, 'uint8')
            pane_bbs = BoundingBoxesOnImage([], self.final_shape)
            pane_polys = PolygonsOnImage([], self.final_shape)
            for (y,x), idx in np.ndenumerate(cell):
                img,_,bbs,polys = seq.augment(**self[idx].iaa)
                pane[y*self.ch:(y+1)*self.ch, x*self.cw:(x+1)*self.cw,...] = img
                pane_bbs.bounding_boxes.extend(bbs.shift(x*self.cw, y*self.ch).items)
                pane_polys.polygons.extend(polys.shift(x*self.cw, y*self.ch).items)
                
            # save i'th image & annotation
            fname = str(i)
            if prefix: fname = f'{prefix}_{fname}'
            imgdst = os.path.join(dst_imgroot, fname+'.jpg')
            anndst = os.path.join(dst_annroot, fname+ANNEXT[dst_anntype])
            annfname = os.path.join(dst_annroot, fname)

            if not overwrite:
                for fp in [imgdst, anndst]:
                    if os.path.exists(fp): raise FileExistsError
            
            cv2.imwrite(imgdst, pane[:,:,::-1])
            pane_annot = self._FolderAugmenter__make_annot((pane,pane_bbs,pane_polys), annfname, dst_anntype, imgdst)
            pane_annot.save()

