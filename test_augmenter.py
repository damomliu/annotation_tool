import imgaug.augmenters as iaa
from annotation.augmenter import FolderAugmenter, BatchGenerator
from annotation.aug.special import SquashRotation

srcfolder = '/home/alpha/vrs/vreu_data/try_inline_aug/images/debug_squashrotation'
faug = FolderAugmenter(srcfolder, 'labelme')
# print(faug.relpaths)
# print(faug.relpaths)

faug.augment(iaa.Rotate(30),
             postfix='iaap30', dst_anntype='labelme', dst_imgroot=srcfolder+'_debug')

faug.augment(SquashRotation(rotate=30, squash_labels=['c','m0','m1'], squash_ratio=0.3).iaa,
             postfix='p30', dst_anntype='labelme', dst_imgroot=srcfolder+'_debug')

# bg = BatchGenerator(faug, batch_size=20)
# for bi,batch in enumerate(bg):
#     print(bi, batch, len(batch.images_unaug))


# faug.augment_batches(iaa.Rotate(30), batch_size=8, multicore=False,
#                      prefix='rot', postfix='p30', dst_anntype='labelimg', dst_imgroot=srcfolder+'rot1')
