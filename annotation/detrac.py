import os
import json
import xml.etree.cElementTree as ET
from glob import glob

import cv2

from .appbase import AppBase
from .image import ImageFile
from .shape import Rectangle

class DetracXML(AppBase):
    def __init__(self, filepath, imgfolder=None, check_exist=True, verbose=False):
        super().__init__(filepath, check_exist)
        self.imgfolder = imgfolder
        self.verbose = verbose
    
    def parse(self):
        tree = ET.parse(self.filepath)
        root = tree.getroot()
        self.data = root
        self.seq = dict(**{'name': root.attrib['name']},
                        **root.find('sequence_attribute').attrib)
        if not self.imgfolder:
            self.imgfolder = os.path.join(self.dirname, self.seq['name'])
        
        self.ignored_shapes = []
        for box in root.find('ignored_region').findall('box'):
            box = box.attrib
            rect = Rectangle(box['left'], box['top'], box['width'], box['height'],
                             format='xywh', label='ignore')
            self.ignored_shapes.append(rect)
        
        self.frame_dict = {}
        for fr in root.findall('frame'):
            fr_d = fr.attrib['density']
            fr_n = int(fr.attrib['num'])
            assert fr_n not in self.frame_dict
            
            fr_shapes = []
            occlusions = []
            for tgt in fr.find('target_list').findall('target'):
                box = tgt.find('box').attrib
                tgt_id = tgt.attrib['id']
                box_attr = tgt.find('attribute').attrib
                rect = Rectangle(box['left'], box['top'], box['width'], box['height'],
                                 format='xywh', label=box_attr['vehicle_type'])
                rect.id = tgt_id
                rect.attrib = box_attr
                fr_shapes.append(rect)

                for occ in tgt.findall('occlusion'):
                    for region in occ.findall('region_overlap'):
                        reg = region.attrib
                        rect_occ = Rectangle(reg['left'], reg['top'], reg['width'], reg['height'],
                                             format='xywh', label='occlusion')
                        if reg['occlusion_status'] == '0':
                            rect_occ.occ_id = tgt_id
                            rect_occ.appear_id = reg['occlusion_id']
                        elif reg['occlusion_status'] == '1':
                            rect_occ.appear_id = tgt_id
                            rect_occ.occ_id = reg['occlusion_id']
                        elif reg['occlusion_status'] == '-1':
                            rect_occ.appear_id = None
                            rect_occ.occ_id = tgt_id
                        else:
                            raise ValueError
                        occlusions.append(rect_occ)
            
            for rect_occ in occlusions:
                rect = [sh for sh in fr_shapes if sh.id==rect_occ.occ_id][0]
                occ_ratio = rect.intersect_area(rect_occ) / rect.area_grid
                rect.attrib['occlusion_ratio'] = occ_ratio
            
            self.frame_dict[fr_n] = {'shapes': fr_shapes, 'density': fr_d}
        
        self._parse_imgfile()
        
    def _parse_imgfile(self):
        for f in glob(os.path.join(self.imgfolder, '*.jpg')):
            fname = os.path.splitext(os.path.basename(f))[0]
            fidx = int(fname.replace('img',''))
            if fidx in self.frame_dict:
                if not self.frame_dict[fidx].get('imgfp'):
                    self.frame_dict[fidx]['imgfp'] = f
                elif self.verbose:
                    print(f'duplicated file/frame_index: [{fidx}] {f}')
            elif self.verbose:
                print(f'missing annotation at frame_index [{fidx}]')
        
        if self.verbose:
            for fidx,val in self.frame_dict.items():
                if not val.get('imgfp'):
                    print(f'missing file at frame_index [{fidx}]')
    
    def from_(self, img_path, shapes):
        raise NotImplementedError

    def save(self, dst=None):
        raise NotImplementedError
    