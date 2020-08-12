import math
from imgaug.augmenters.meta import Lambda
import imgaug.augmenters as iaa

class SquashRotation:
    def __init__(self,
                 squash_ratio=None,
                 squash_labels=['c'],
                 rotate=30,
                 order=1, cval=0, mode="constant", **kwargs):
        
        self.squash_ratio = squash_ratio
        self.squash_labels = squash_labels
        self.rotate = rotate
        self.rotate_kwargs = dict(rotate=rotate, order=order, cval=cval, mode=mode, **kwargs)

    def func_images(self, images, random_state, parents, hooks):
        return iaa.Rotate(**self.rotate_kwargs).augment_images(images, parents, hooks)
    
    def func_polygons(self, polygons_on_images, random_state, parents, hooks):
        return iaa.Rotate(**self.rotate_kwargs).augment_polygons(polygons_on_images, parents, hooks)
    
    def func_bounding_boxes(self, bounding_boxes_on_images, random_state, parents, hooks):
        bbox_aug = iaa.Rotate(**self.rotate_kwargs).augment_bounding_boxes(bounding_boxes_on_images, parents, hooks)
        for i,boi in enumerate(bbox_aug):
            for j,bbox in enumerate(boi.items):
                if bbox.label in self.squash_labels:
                    bbox_ori = bounding_boxes_on_images[i].items[j]
                    dw = (bbox.width - bbox_ori.width) / 2
                    dh = (bbox.height - bbox_ori.height) / 2
                    
                    if self.squash_ratio:
                        dw *= (1 - self.squash_ratio)
                        dh *= (1 - self.squash_ratio)
                    else:
                        dw *= abs(math.sin(math.radians(self.rotate)))
                        dh *= abs(math.sin(math.radians(self.rotate)))
                    
                    bbox.extend_(top=-dh, right=-dw, bottom=-dh, left=-dw)
        return bbox_aug
    
    def __call__(self):
        return self.iaa
        
    @property
    def iaa(self):
        return Lambda(func_images=self.func_images,
                      func_polygons=self.func_polygons,
                      func_bounding_boxes=self.func_bounding_boxes)