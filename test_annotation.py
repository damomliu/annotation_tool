import os
from glob import glob

from annotation import ImageFile, LabelImgXML, LabelmeJSON

jf = LabelmeJSON('/media/alpha/liebe/vreu_data/newdata_0615/images/from114/20200616_EU_france_03_VR/11-BLN-77__VR_EU_france_03.json')
print(jf.imgfile)
print(jf.imgfile)

xml = LabelImgXML('img320191126_TaipeiRoadNH_night1.xml')
print(xml.shapes)

xml = LabelImgXML('/media/alpha/liebe/vr_data/201804_20200318/20191203_20191205/20191203_20191205_KYEC/252/K11_192.168.0.95_20200305153320_BAK0372.xml')
folder = '/media/alpha/liebe/vr_data/VRType/VRS_VR_original_image_Type/Negative_Sample/20191117'
dst_folder = '/home/alpha/Desktop/tmp20191117'
for f in glob(os.path.join(folder, '*.jpg')):
    imgpath = os.path.join(folder, f)
    img = ImageFile(imgpath)
    
    xmlpath = os.path.join(folder, img.fname+'.xml')
    xml = LabelImgXML(xmlpath)
    
    xml.from_(imgpath, xml.shapes)
    xml.save(os.path.join(dst_folder, xml.filename))
    

# # LabelMe .json 轉 LabelImg .xml
# from annotation.labelme import LabelmeJSON
# lm = LabelmeJSON('/media/alpha/liebe/face_dataset/WIDERFACE/val/images/0--Parade/0_Parade_Parade_0_960.json')
# lmpair = LabelmePair(
#     '/media/alpha/liebe/face_dataset/WIDERFACE/val/images/0--Parade/0_Parade_Parade_0_960.jpg',
#     '/media/alpha/liebe/face_dataset/WIDERFACE/val/images/0--Parade/0_Parade_Parade_0_960.json',
# )
# lmpair.to_labelimg()

# # RetinaFace 轉 LabelMe 的 .json
# from annotation.retinaface import RetinaFaceTXT

# RF = RetinaFaceTXT('/media/alpha/liebe/face_dataset/WIDERFACE/val/label.txt')
# RF.parse()
# RF.to_labelme()