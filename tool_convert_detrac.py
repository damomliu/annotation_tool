import os
import shutil
from glob import glob
from tqdm import tqdm
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from annotation import DetracXML, LabelImgXML, ImageFile, MotGT, MotDET

def cvt_detrac(folder, copy_dst=None):
    nullimg = np.zeros([540,960,3], dtype='uint8')
    with tqdm(glob(os.path.join(folder, '*.xml'))) as pbar:
        for xmlpath in pbar:
            pbar.set_description(f'{os.path.basename(folder)} [{os.path.basename(xmlpath)}]')
            
            annots = DetracXML(xmlpath)
            annots.parse()
            if copy_dst:
                seq = annots.seq
                dst = os.path.join(copy_dst, f"{seq['name']}_{seq['sence_weather']}_{seq['camera_state']}")
                if not os.path.isdir(dst): os.makedirs(dst)
                
            for fidx,val in tqdm(annots.frame_dict.items(), leave=False):
                imgpath = val['imgfp']
                if copy_dst:
                    shutil.copy2(imgpath, dst)
                    imgpath = os.path.join(dst, os.path.basename(imgpath))
                
                occ_thres = 0.5    
                shapes = [sh for sh in val['shapes'] if sh.attrib.get('occlusion_ratio', 0) < occ_thres]
                shapes_occ = [sh for sh in val['shapes'] if sh.attrib.get('occlusion_ratio', 0) >= occ_thres]
                for i,sh in enumerate(shapes):
                    attr = sh.attrib
                    # shapes[i].label = attr.get('color', 'x') + '_' + attr.get('vehicle_type','x') + f'_speed{float(attr.get("speed",0)):.1f}'
                    # shapes[i].label = sh.id
                    shapes[i].label = 'c'
                for i,sh in enumerate(shapes_occ):
                    shapes_occ[i].label = 'occlusion'
                
                shapes.extend(shapes_occ)
                shapes.extend(annots.ignored_shapes)
                
                xmlframe = os.path.splitext(imgpath)[0] + '.xml'
                xf = LabelImgXML(xmlframe, check_exist=False)
                xf.from_array(nullimg, imgpath, shapes)
                xf.save()

mot_labels = {1: 'Pedestrian',  2: 'Person_on_vehicle', 3: 'Car',
              4: 'Bicycle',     5: 'Motorbike',         6: 'Non_Motorized',
              7: 'StaticPerson',8: 'Distractor',        9: 'Occluder',
              10:'Occluder_ground', 11: 'Occluder_full', 12:'Reflection',
              13:'Crowd'}
mot_labels = {1:'r', 2:'r', 3:'c', 4:'m1', 5:'m1', 6:'others',
              7:'r', 8:'x', 9:'x', 10:'x', 11:'x', 12:'x', 13:'ignore'}
def cvt_mot(folder, mode='gt'):
    if mode == 'gt':
        app = MotGT
        filename = 'gt.txt'
    elif mode == 'gt15':
        app = MotDET
        filename = 'gt.txt'
    elif mode == 'det':
        app = MotDET
        filename = 'det.txt'
    else:
        raise ValueError
    
    with tqdm(glob(os.path.join(folder, '**', filename), recursive=True)) as pbar:
        for txtpath in pbar:
            pbar.set_description(f'{os.path.basename(folder)} [{txtpath.split(os.sep)[-3]}]')
            annots = app(txtpath,
                         imgfolder=os.path.join(os.path.dirname(txtpath),'../img1'),
                         all_labels=mot_labels,
                         one_label='r',
                         verbose=True)
            annots.parse()
            for fidx,val in tqdm(annots.frame_dict.items(), leave=False):
                imgpath = val['imgfp']
                shapes = [sh for sh in val['shapes'] if sh.label!='x']
                xmlframe = os.path.splitext(imgpath)[0] + '.xml'
                xf = LabelImgXML(xmlframe, check_exist=False)
                xf.from_(imgpath, shapes)
                xf.save()

def df_detrac(folder):
    df = pd.DataFrame(columns=['data','vidx','camera','weather','ignore'])
    df_dict = {'summary': df}
    with tqdm(glob(os.path.join(folder, '*.xml'))) as pbar:
        for xmlpath in pbar:
            df_ = pd.DataFrame(columns=['vidx','fidx','density','id',])
            pbar.set_description(f'{os.path.basename(folder)} [{os.path.basename(xmlpath)}]')
            annots = DetracXML(xmlpath)
            annots.parse()
            for fidx,val in tqdm(annots.frame_dict.items(), leave=False):
                imgpath = val['imgfp']
                shapes = val['shapes']
                for i,sh in enumerate(shapes):
                    attr = sh.attrib
                    shapes[i].label = attr.get('color', 'x') + '_' + attr.get('vehicle_type','x') + f'_speed{float(attr.get("speed",0)):.1f}'
                shapes.extend(annots.ignored_shapes)

def crop_detrac(folder, dst, key):
    if isinstance(key, list):
        if isinstance(dst, list):
            assert len(dst) == len(key)
        elif isinstance(dst, str):
            dst = [f'{dst}_{k}' for k in key]
    
    else:
        if key not in dst: dst += f'_{key}'
        key = [key]
        dst = [dst]
    
    cnt = Counter()
    with tqdm(glob(os.path.join(folder, '*.xml'))) as pbar:
        for xmlpath in pbar:
            pbar.set_description(f'{os.path.basename(folder)} [{os.path.basename(xmlpath)}]')
            annots = DetracXML(xmlpath)
            annots.parse()
            for fidx,val in tqdm(annots.frame_dict.items(), leave=False):
                img = ImageFile(val['imgfp']).img
                for sh in val['shapes']:
                    crop = img.crop(sh).cvt_color('bgr').numpy
                    if max(crop.shape) < 100:
                        cnt.update(['skip(w/h)'])
                        continue
                    elif float(sh.attrib['truncation_ratio']) > 0.2:
                        cnt.update(['skip(truncation)'])
                        continue
                    elif sh.attrib.get('occlusion_ratio', 0) > 0.2:
                        cnt.update(['skip(occlusion)'])
                        continue
                    
                    for k,d in zip(key, dst):
                        attrib = sh.attrib.get(k)
                        if attrib:
                            # dstfolder = os.path.join(dst, annots.seq['name'], f'{sh.id}_{attrib}')
                            dstfolder = os.path.join(d, attrib, f'{annots.seq["name"]}_id{sh.id}')
                            dstfilename = f'frame{fidx}.jpg'
                            if not os.path.isdir(dstfolder): os.makedirs(dstfolder)
                            cv2.imwrite(os.path.join(dstfolder, dstfilename), crop)
                            cnt.update([f'crop_{k}'])
                        else:
                            cnt.update([f'skip_{k}(no attrib)'])
                pbar.set_postfix(cnt)

def imgblur(crop, resize=None):
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if resize:
        if isinstance(resize, int): resize = (resize, resize)
        crop = cv2.resize(crop, resize)
    blur = cv2.Laplacian(crop, cv2.CV_64F).var()
    return blur

def pick_one_image(src):
    pbar = tqdm(total=sum([len(files) for _,_,files in os.walk(src)]))
    cnt = Counter()
    for root,dirs,files in os.walk(src):
        if not dirs and files:
            blur_list = [imgblur(cv2.imread(os.path.join(root, f))) for f in files]
            best_idx = np.argmax(blur_list)
            best_file = os.path.join(root, files[best_idx])
            newfp = os.path.join(os.path.dirname(root), os.path.basename(root)+'_'+os.path.basename(best_file))
            shutil.move(best_file, newfp)
            shutil.rmtree(root)
            
            cnt.update(['pick'])
            pbar.set_postfix(cnt)
        pbar.update(len(files))

if __name__ == "__main__":
    # DETRAC
    cvt_detrac('/media/alpha/liebe/vreuflow_data/DETRAC-test',
               copy_dst='/media/alpha/liebe/vreuflow_data/DETRAC')
    # crop_detrac('/media/alpha/liebe/vreuflow_data/DETRAC-test',
    #             dst='/media/alpha/liebe/vreuflow_data/DETRAC-test',
    #             key=['color','vehicle_type'])
    # pick_one_image('/media/alpha/liebe/vreuflow_data/DETRAC-test_vehicle_type')
    
    # MOT
    # cvt_mot('/media/alpha/liebe/pr_data/MOT16/images/train/MOT16-13', mode='gt')
    
    pass