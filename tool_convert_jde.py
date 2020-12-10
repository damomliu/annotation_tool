import os
from glob import glob
from tqdm import tqdm
from collections import Counter
from shutil import copy2

import numpy as np
import pandas as pd

from annotation import JdeTXT, VisDET, VisMOT, LabelImgXML

__img__ = ['.png', '.jpg', '.jpeg']

def _init_parm(img_name, label_name, label_src, img_src=None, dst=None, img_ext=None):
    if img_src is None:
        img_src = os.path.join(label_src, img_name)
        label_src = os.path.join(label_src, label_name)
    if img_ext is None:
        for _,_,files in os.walk(img_src):
            for f in files:
                ext = os.path.splitext(f)[-1]
                if ext in __img__:
                    img_ext = ext
                    break
    if dst is None:
        dst = img_src
    
    return label_src, img_src, dst, img_ext

def _cvt_base(app, img_name, label_name, label_src, img_src=None, dst=None, img_ext=None, **kwargs):
    label_src, img_src,dst,img_ext = _init_parm(img_name, label_name, label_src, img_src, dst, img_ext)
    
    cnt = Counter({'OK': 0, 'Fail': 0})
    with tqdm(sorted(glob(os.path.join(label_src, '**/*.txt'), recursive=True))) as pbar:
        for f in pbar:
            try:
                relpath = os.path.relpath(f, label_src)
                fname_rel = os.path.splitext(relpath)[0]
                imgfile_src = os.path.join(img_src, fname_rel+img_ext)
                xml_dst = os.path.join(dst, fname_rel+'.xml')
                
                annot = app(f, imgfile_src, check_exist=False, **kwargs)
                annot.parse()
                # if set(['7','8']).issubset(set(annot.labels)):
                #     copy2(imgfile_src, '/media/alpha/liebe/vreuflow_data/VisDrone2019-DET-train/tri')
                #     xml_dst = os.path.join('/media/alpha/liebe/vreuflow_data/VisDrone2019-DET-train/tri', fname_rel+'.xml')
                xf = annot.to_labelImg()
                xf.save(xml_dst)
                # cnt.update(jt.labels)
                cnt.update(['OK'])
            except Exception as e:
                print(str(e))
                print(f)
                cnt.update(['Fail'])
            
            pbar.set_postfix(cnt)

pr_labels = ['r']
def cvt_jde(label_src, img_src=None):
    _cvt_base(JdeTXT, 'images', 'labels_with_ids', label_src, img_src, all_labels=pr_labels)

vis_labels = ['ignore', 'pedestrian(1)', 'people(2)', 'bicycle(3)', 'car(4)',
              'van(5)', 'truck(6)', 'tricycle(7)', 'awning-tricycle(8)', 'bus(9)',
              'motor(10)', 'others(11)']
vis_labels = ['ignore', 'r', 'r', 'm1', 'c',
              'c', 'c', 'm1', 'm1', 'c', 'm1', 'other']
def cvt_vis_det(label_src, img_src=None):
    _cvt_base(VisDET, 'images', 'annotations', label_src, img_src, all_labels=vis_labels)

def cvt_vis_mot(folder):
    pairs = _get_vis_mot_pair(folder)
    with tqdm(pairs) as pbar:
        for txtpath,imgfolder in pbar:
            vm = VisMOT(txtpath, imgfolder, all_labels=vis_labels)
            vm.parse()
            with tqdm(vm.shape_dict.items(), leave=False, desc=os.path.basename(txtpath)) as pbar2:
                for fidx,shapes in pbar2:
                    imgpath = vm.imgfile_dict[fidx]
                    xmlpath = os.path.splitext(imgpath)[0] + '.xml'
                    xf = LabelImgXML(xmlpath, check_exist=False)
                    xf.from_(imgpath, shapes)
                    xf.save()

def _get_vis_mot_pair(folder):
    pairs = []
    annfolder = os.path.join(folder, 'annotations')
    for txt in os.listdir(annfolder):
        fname = os.path.splitext(txt)[0]
        imgfolder = os.path.join(folder, 'sequences', fname)
        if os.path.isdir(imgfolder):
            pairs.append([os.path.join(annfolder, txt), imgfolder])
    return pairs

def merge_visdrone(src, dst, postfix=None, mode='auto'):
    if postfix is None:
        postfix = os.path.basename(src).replace('VisDrone2019-', '')
    
    if mode == 'auto':
        mode = 'DET' if 'DET' in os.path.basename(src) else 'MOT'
    
    df = pd.DataFrame(columns=['vidx','sec','cat','fidx', 'data', 'labels'])
    for xmlpath in tqdm(glob(os.path.join(src, '**/*.xml'), recursive=True), desc=postfix):
        if mode == 'DET':
            fname = os.path.splitext(os.path.basename(xmlpath))[0]
            vidx,sec,cat,fidx = fname.split('_')
            vidx = 'uav' + vidx
        
        elif mode == 'MOT':
            dname,fname = os.path.split(xmlpath)
            dname = os.path.basename(dname)
            vidx,sec,cat = dname.split('_')
            
            fidx = os.path.splitext(fname)[0]
            
        else:
            raise ValueError
        
        # folder1 = os.path.join(dst, f'{vidx}')
        # if not os.path.isdir(folder1): os.makedirs(folder1)
        # fn1 = os.path.join(folder1, f'{sec}_{fidx}_{cat}_{postfix}')
        
        xf = LabelImgXML(xmlpath)
        row = dict(vidx=vidx, sec=sec, cat=cat, fidx=fidx,
                   data=postfix, labels=[sh.label for sh in xf.shapes])
        df = df.append(row, ignore_index=True)
        # xf1 = LabelImgXML(fn1+'.xml', check_exist=False)
        # imgpath1 = os.path.join(fn1 + os.path.splitext(xf.imgpath)[-1])
        
        # xf1.from_array(np.zeros([xf.imgh,xf.imgw,xf.imgc], dtype='uint8'),
        #                imgpath1, xf.shapes)
        # xf1.save()
        # copy2(xf.imgpath, imgpath1)
    return df

def df_xml(src, mode=None):
    df = pd.DataFrame(columns=['category','data','vidx','filename','labels'])
    cnt = Counter()
    with tqdm(sorted(glob(os.path.join(src, '**/*.xml'), recursive=True))) as pbar:
        for xmlpath in pbar:
            annot = LabelImgXML(xmlpath)
            labels = [sh.label for sh in annot.shapes]
            _paths = xmlpath.split(os.sep)[::-1]
            if mode == 'MOT':
                filename,chk0,vidx,data,chk1,dirname = _paths[:6]
                assert chk0=='img1' and chk1=='images'
                row = dict(category=dirname, data=data, vidx=vidx, filename=filename, labels=labels)
                pbar.set_description(f'[{dirname}/{vidx}]')
            
            elif mode == 'ETHZ':
                filename,chk0,vidx,chk1 = _paths[:4]
                assert chk0=='images' and chk1=='ETHZ'
                row = dict(category=chk1,vidx=vidx,filename=filename, labels=labels)
            
            elif mode == 'common':
                filename,vidx,chk0,cat = _paths[:4]
                assert chk0=='images'
                row = dict(category=cat,vidx=vidx, filename=filename,labels=labels)
            
            elif mode is None:
                cat = os.path.basename(src)
                row = dict(category=cat, filename=os.path.basename(xmlpath), labels=labels)
            else:
                raise ValueError
            df = df.append(row, ignore_index=True)

            cnt.update(labels)
            pbar.set_postfix(cnt)
    
    for i,row in df.iterrows():
        file_labels = Counter(row.labels)
        for l in cnt.keys():
            l_count = file_labels.get(l, 0)
            df.loc[i,l] = l_count
    
    del df['labels']
    for l in cnt.keys():
        df[l] = df[l].astype(int)
    
    return df

if __name__ == "__main__":
    # cvt_jde('/media/alpha/liebe/vreuflow_data/Citypersons', img_ext='.png')
    
    # PRW
    # cvt_jde('/media/alpha/liebe/pr_data/PRW')
    
    # CUHK
    # cvt_jde('/media/alpha/liebe/pr_data/CUHK-SYSU')

    # ETHZ
    # for folder in glob('/media/alpha/liebe/pr_data/ETHZ/*'):
    #     cvt_jde(folder)

    MOT
    cvt_jde('/media/alpha/liebe/pr_data/MOT16')

    # Visdrone
    # cvt_vis_det('/media/alpha/liebe/vreuflow_data/VisDrone2019-DET-val')
    # cvt_vis_mot('/media/alpha/liebe/vreuflow_data/VisDrone2019-MOT-test-dev')
    # cvt_vis_mot('/media/alpha/liebe/vreuflow_data/VisDrone2020-CC')
    
    # vis_dst = '/media/alpha/liebe/vreuflow_data/VisDrone2019'
    # merge_visdrone('/media/alpha/liebe/vreuflow_data/VisDrone2019-VID-train', vis_dst)

    pass