import os, glob
from annotation import YoloTXT, LabelImgXML, LabelmeJSON

xml = LabelImgXML('/media/alpha/liebe/vreu_data/newdata_0615/images/val/170902_L/P9170008.xml')
yolo = YoloTXT(xml.filepath.replace('.xml','.txt'), labels=['c','p','m1'], check_exist=False)
yolo.from_array(xml.imgfile.rgb, xml.shapes)
yolo.save()

# yolo = YoloTXT('/media/alpha/liebe/vreu_data/newdata_0615/labels/val/170902_L/P9170008.txt',
#                '/media/alpha/liebe/vreu_data/newdata_0615/images/val/170902_L/P9170008.jpg')