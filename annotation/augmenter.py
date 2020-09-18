import os
from tqdm import tqdm
import imageio
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug.augmentables.polys import PolygonsOnImage
from imgaug.augmentables.batches import UnnormalizedBatch

from .base import FilePath
from .image import ImageFile
from .app import LabelmeJSON, LabelImgXML, YoloTXT
from .shape import Point, Rectangle, Polygon

ANNEXT = dict(
    labelme = '.json',
    labelimg = '.xml',
    yolo = '.txt',
)
IMGEXT = ['.jpg','.jpeg','.png']   


class FolderAugmenter(FilePath):
    def __init__(self, src_root, src_type, walk=True):
        super().__init__(src_root, check_exist=True)
        src_type = src_type.lower()
        assert src_type in ['labelme', 'labelimg', 'yolo', 'image']
        self.src_root = src_root
        self.src_type = src_type
        self.walk = walk
    
    def __repr__(self):
        return f'<{self.__class__.__name__} @{os.path.basename(self.src_root)}({self.src_type}*{len(self.relpaths)})>'
    
    def _get_relpaths(self):
        if self.src_type == 'image':
            ext = IMGEXT
        else:
            ext = [ANNEXT[self.src_type]]
        
        filelist = []
        if self.walk:
            for root,_,files in os.walk(self.src_root):
                for f in files:
                    if os.path.splitext(f)[-1].lower() in ext:
                        relpath = os.path.relpath(os.path.join(root, f), self.src_root)
                        filelist.append(relpath)
        else:
            for f in os.listdir(self.src_root):
                if os.path.splitext(f)[-1].lower() in ext:
                    filelist.append(f)
        
        self._relpaths = filelist
    
    @property
    def relpaths(self):
        if not hasattr(self, '_relpaths'): self._get_relpaths()
        return self._relpaths
    
    @property
    def annpaths(self):
        if self.src_type=='image':
            return None
        else:
            return [os.path.join(self.src_root, rpath) for rpath in self.relpaths]
    
    def __get_imgpaths(self):
        if self.src_type=='image':
            self._imgpaths = [os.path.join(self.src_root, rpath) for rpath in self.relpaths]
        else:
            self._imgpaths = []
            for f in iter(self):
                self._imgpaths.append(f.imgpath)
    
    @property
    def imgpaths(self):
        if not hasattr(self, '_imgpaths'): self.__get_imgpaths()
        return self._imgpaths
    
    def __getitem__(self, i):
        if i >= len(self.relpaths): raise IndexError
        
        fp = os.path.join(self.src_root, self.relpaths[i])
        if self.src_type=='labelme':
            return LabelmeJSON(fp)
        elif self.src_type=='labelimg':
            return LabelImgXML(fp)
        elif self.src_type=='yolo':
            return YoloTXT(fp)
        elif self.src_type=='image':
            return ImageFile(fp)
        else:
            raise NotImplementedError
    
    def augment(self, seq,
                dst_imgroot=None,
                dst_annroot=None, dst_anntype=None,
                prefix=None, postfix=None, overwrite=False,
                desc=None,
                yolo_labels=None,
                ):
        
        if not dst_imgroot: dst_imgroot = self.src_root
        if not dst_annroot: dst_annroot = dst_imgroot
        
        kwargs = dict(
            dst_imgroot=dst_imgroot,
            dst_annroot=dst_annroot, dst_anntype=dst_anntype,
            prefix=prefix, postfix=postfix, overwrite=overwrite,
            yolo_labels=yolo_labels,
        )
        
        pbar = tqdm(iter(self), total=len(self.relpaths))
        if desc:
            pbar.set_description(desc)
        elif prefix or postfix:
            desc = prefix if prefix else ''
            desc += '_'
            desc += postfix if postfix else ''
            pbar.set_description(desc)
            
        for i,f in enumerate(pbar):
            results = seq.augment(**f.iaa)
            self.__save_ith_result(results, i, **kwargs)
            
        pbar.close()
    
    def augment_batches(self, seq,
                        batch_size, multicore=True,
                        dst_imgroot=None,
                        dst_annroot=None, dst_anntype=None,
                        prefix=None, postfix=None, overwrite=False,
                        desc=None,
                        yolo_labels=None,
                        ):
        if not dst_imgroot: dst_imgroot = self.src_root
        if not dst_annroot: dst_annroot = dst_imgroot
        
        kwargs = dict(
            dst_imgroot=dst_imgroot,
            dst_annroot=dst_annroot, dst_anntype=dst_anntype,
            prefix=prefix, postfix=postfix, overwrite=overwrite,
            yolo_labels=yolo_labels,
        )
        
        pbar = tqdm(total=len(self.relpaths))
        if desc:
            pbar.set_description(desc)
        elif prefix or postfix:
            desc = prefix if prefix else ''
            desc += '_'
            desc += postfix if postfix else ''
            pbar.set_description(desc + ' - generating batches...')
        
        bg = BatchGenerator(self, batch_size)
        batches_aug = list(seq.augment_batches(iter(bg), background=multicore))
        
        pbar.set_description(desc)
        for bi,batch in enumerate(batches_aug):
            for j,img in enumerate(batch.images_aug):
                i = batch_size * bi + j
                i_results = [img]
                
                augkey = ['keypoints_aug', 'bounding_boxes_aug', 'polygons_aug']
                for k in augkey:
                    val = batch.__dict__[k]
                    if val is not None:
                        i_results.append(val[j])
                
                self.__save_ith_result(i_results, i, **kwargs)
            
            pbar.update(len(batch.images_aug))
        pbar.close()
    
    def __save_ith_result(self, results, i,
                          dst_imgroot=None,
                          dst_annroot=None, dst_anntype=None,
                          prefix=None, postfix=None, overwrite=False,
                          desc=None,
                          yolo_labels=None,
                          ):
        
        dst_imgfolder,filename = os.path.split(os.path.join(dst_imgroot, self.relpaths[i]))
        fname = os.path.splitext(filename)[0]
        imgpath = self[i].imgfile.filepath if self.src_type!='image' else self[i].filepath
        imgext = os.path.splitext(imgpath)[-1]
        if prefix: fname = f'{prefix}_{fname}'
        if postfix: fname = f'{fname}_{postfix}'
        
        rgb = results[0] if self.src_type!='image' else results
        imgdst = os.path.join(dst_imgfolder, fname+imgext)
        if os.path.exists(imgdst) and not overwrite: raise FileExistsError
        self.__save_img(rgb, imgdst)
        
        if dst_anntype is not None:
            dst_annfolder = os.path.split(os.path.join(dst_annroot, self.relpaths[i]))[0]
            annfname = os.path.join(dst_annfolder, fname)
            
            if isinstance(dst_anntype, str): dst_anntype = [dst_anntype]
            for anntype in dst_anntype:
                annotfile = self.__make_annot(results, annfname, anntype, imgdst, yolo_labels)
                if os.path.exists(annotfile.filepath) and not overwrite: raise FileExistsError
                annotfile.save()
    
    
    @staticmethod
    def __save_img(rgb, dst):
        dstfolder = os.path.dirname(dst)
        if not os.path.isdir(dstfolder): os.makedirs(dstfolder)
        # cv2.imwrite(dst, bgr)
        imageio.imwrite(dst, rgb)
    
    @staticmethod
    def __make_annot(results, annfname, anntype, imgdst=None, yolo_labels=None):
        rgb = results[0]
        annresults = results[1:]
        shapes = []
        for annot in annresults:
            annot = annot.clip_out_of_image()
            if isinstance(annot, KeypointsOnImage):
                for kp in annot.keypoints:
                    shapes.append(Point(from_iaa=kp))
            
            elif isinstance(annot, BoundingBoxesOnImage):
                for box in annot.bounding_boxes:
                    shapes.append(Rectangle(from_iaa=box))
            
            elif isinstance(annot, PolygonsOnImage):
                for poly in annot.polygons:
                    shapes.append(Polygon(from_iaa=poly))
        
        annfolder,fname = os.path.split(annfname)
        # fname = os.path.splitext(fname)[0]
        
        annotdst = os.path.join(annfolder, fname+ANNEXT[anntype])
        if anntype=='labelme':
            annotfile = LabelmeJSON(annotdst, check_exist=False)
        elif anntype=='labelimg':
            annotfile = LabelImgXML(annotdst, check_exist=False)
        elif anntype=='yolo':
            annotfile = YoloTXT(annotdst, check_exist=False, labels=yolo_labels)
        else:
            NotImplemented
        
        annotfile.from_array(rgb=rgb, imgpath=imgdst, shapes=shapes)
        
        return annotfile


class BatchGenerator():
    def __init__(self, faug, batch_size):
        self.faug = faug
        self.batch_size = batch_size
        
        self.nb_total = (len(self.faug.relpaths) // self.batch_size) + 1
        self.keys = self.faug[0].iaa.keys()
    
    def __len__(self):
        return self.nb_total
    
    def __getitem__(self, nb):
        if nb >= self.nb_total: raise StopIteration
        
        batchitem = {k:[] for k in self.keys}
        iterend = min(self.batch_size * (nb+1), len(self.faug.relpaths))
        for i in range(self.batch_size * nb, iterend):
            if i >= len(self.faug.relpaths): raise StopIteration
            for k,val in self.faug[i].iaa.items():
                batchitem[k].append(val)
        
        batchitem['images'] = batchitem['image']
        del batchitem['image']
        
        return UnnormalizedBatch(**batchitem)
