import os
import imageio
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug.augmentables.polys import PolygonsOnImage

from .app import LabelmeJSON, LabelImgXML
from .shape import Point, Rectangle, Polygon

class FolderAugmenter():
    def __init__(self, src_root, src_type, walk=True):
        assert src_type in ['labelme', 'labelImg']
        self.src_root = src_root
        self.src_type = src_type
        self.walk = walk
    
    def __get_relpaths(self):
        if self.src_type=='labelme':
            ext = '.json'
        elif self.src_type=='labelImg':
            ext = '.xml'
        else:
            raise NotImplementedError
        
        filelist = []
        if self.walk:
            for root,_,files in os.walk(self.src_root):
                for f in files:
                    if os.path.splitext(f)[-1].lower() in ext:
                        relpath = os.path.relpath(os.path.join(root, f), self.src_root)
                        filelist.append(relpath)
        else:
            for f in os.listdir(self.src_root):
                if os.path.splitext(f)[-1].lower()in ext:
                    filelist.append(f)
        
        self._relpaths = filelist
    
    @property
    def relpaths(self):
        if not hasattr(self, '_relpaths'): self.__get_relpaths()
        return self._relpaths
    
    def __getitem__(self, i):
        if i >= len(self.relpaths): raise IndexError
        
        fp = os.path.join(self.src_root, self.relpaths[i])
        if self.src_type=='labelme':
            return LabelmeJSON(fp)
        elif self.src_type=='labelImg':
            return LabelImgXML(fp)
        else:
            raise NotImplementedError
    
    def augment(self, seq, dst_root=None, dst_type=None, prefix=None, postfix=None, overwrite=False):
        if not dst_root: dst_root = self.src_root
        for i,f in enumerate(iter(self)):
            results = seq.augment(image=f.imgfile.rgb, **f.iaa)
            
            dst_folder,filename = os.path.split(os.path.join(dst_root, self.relpaths[i]))
            fname = os.path.splitext(filename)[0]
            imgext = os.path.splitext(f.imgfile.filepath)[-1]
            if prefix: fname = f'{prefix}_{fname}'
            if postfix: fname = f'{fname}_{postfix}'
            
            rgb = results[0]
            if not os.path.isdir(dst_folder): os.makedirs(dst_folder)
            imgdst = os.path.join(dst_folder, fname+imgext)
            if os.path.exists(imgdst) and not overwrite: raise FileExistsError
            # cv2.imwrite(imgdst, bgr)
            imageio.imwrite(imgdst, rgb)
            
            if dst_type is not None:
                shapes = []
                for annot in results[1:]:
                    annot.clip_out_of_image()
                    if isinstance(annot, KeypointsOnImage):
                        for kp in annot.keypoints:
                            shapes.append(Point(from_iaa=kp))
                    
                    elif isinstance(annot, BoundingBoxesOnImage):
                        for box in annot.bounding_boxes:
                            shapes.append(Rectangle(from_iaa=box))
                    
                    elif isinstance(annot, PolygonsOnImage):
                        for poly in annot.polygons:
                            shapes.append(Polygon(from_iaa=poly))
                
                if dst_type=='labelme':
                    annotdst = os.path.join(dst_folder, fname+'.json')
                    annotfile = LabelmeJSON(annotdst, check_exist=False)
                elif dst_type=='labelImg':
                    annotdst = os.path.join(dst_folder, fname+'.xml')
                    annotfile = LabelImgXML(annotdst, check_exist=False)
                
                if os.path.exists(annotdst) and not overwrite: raise FileExistsError
                annotfile.from_array(rgb, imgdst, shapes)
                annotfile.save(annotdst)