import numpy as np
import imgaug.augmenters as iaa
from annotation.aug.stitching import StitchingMaker

srcroot = '/home/alpha/vrs/vreu_data/images/tmp_for_stitching'

sm = StitchingMaker(srcroot, 'labelme', (1920,1080), (4,3))
sm.shuffle()

sm.make(srcroot+'_stitching', overwrite=True, cell_aug=iaa.Rotate((-30,30)))
# i = 0
# self = sm
# cell = self._cell_idx[i]
# pane = np.zeros((self.final_size[1],self.final_size[0],3), 'uint8')
# for (y,x), idx in np.ndenumerate(cell):
#     img,_,bbs,polys = seq.augment(**self[idx].iaa)