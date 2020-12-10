import os
from glob import glob
from tqdm import tqdm

from annotation.shape import Rectangle
from annotation import KittiTXT

annroot = '/media/alpha/liebe/KITTI/object/training/label_2'
imgroot = '/media/alpha/liebe/KITTI/object/training/image_2_jpg'
dstroot = imgroot

if __name__ == "__main__":
    pbar = tqdm(sorted(glob(os.path.join(annroot,'*.txt'))))
    for f in pbar:
        fname = os.path.splitext(os.path.basename(f))[0]
        # with open(annfp,'r') as f:
        #     lines = f.readlines()
        # for line in lines:
        #     linesplit = line.split(' ')
        #     label = linesplit[0]
        kf = KittiTXT(f)
        kf.parse()
        for sh in kf.shapes:
            if sh.label == 'Car': sh.label = 'c'
        
        imgpath = glob(os.path.join(imgroot, fname+'.*'))[0]
        xf = kf.to_labelImg(imgpath)
        xf.save()
        pass
