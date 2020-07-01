# annotation
## 安裝
- pip
    ```shell
    pip install -U git+http://192.168.0.116:10080/damon/annotation_tool.git
    ```

- clone & copy to your project
    ```shell
    git clone http://192.168.0.116:10080/damon/annotation_tool.git
    cd annotation_tool
    cp -R annotation /path/to/your/project/
    ```

## 使用

### 讀取 LabelmeJSON 內的資訊 / 重寫
```python
from annotation import LabelmeJSON
jf = LabelmeJSON('path/to/json')
jf.parse()
jf.data['imagePath'] = 'new_image_name'
jf.save()
```

### FolderAugmenter : iaa.Augmenter 再封裝 (folder --> folder)
```python
import imgaug.augmenters as iaa
from annotation import FolderAugmenter

faug = FolderAugmenter('srcfolder', src_type="labelme", walk=True)

print(faug.annpaths)  # ['1.json', '2.json', ...]
print(faug.imgpaths)  # ['1.jpg', '2.jpg', ...]
```
- example: 旋轉30度、並將圖片命名為 'rot30_%' (不儲存標記檔)
  ```python
  faug.augment(iaa.Rotate(30), prefix='rot30', overwrite=True)
  ```
- example: 旋轉-30度、重命名為 'rot_%_n30'、並儲存 labelImg .xml 標記檔
  ```python
  faug.augment(iaa.Rotate(-30), prefix='rot', postfix='n30',
               dst_type='labelImg', overwrite=True))
  ```
- example: 旋轉30度、重命名為 '%_rot30'、同時儲存 labelme(.json) & yolo(.txt)；旋轉完的圖片儲存到 'img_aug' 、旋轉完的標記檔儲存到 'ann_aug'
  ```python
  faug.augment(iaa.Rotate(-30), postfix='rot30',
               dst_imgroot='img_aug', dst_annroot='ann_aug',
               dst_anntype=['labelme','yolo'])
  ```

### Labelme .json --> LabelImg .xml
```python
from annotation import LabelmeJSON
json_path = 'example/img320191126_TaipeiRoadNH_night1.json'
json_ = LabelmeJSON(json_path)
```
![image](example/sample_labelme%20.png)
- 將 Labelme 的 .json 直接轉換成 LabelImg 可以讀的 .xml
    1. 用 LabelmeJSON.to_labelImg(), 忽略 polygon
        ```python
        xml = json_.to_labelImg()
        xml.save()
        ```
        ![image](example/sample_xml1.png)

    2. 用 LabelmeJSON.to_labelImg(), 將所有 polygon 轉換成  rectangle
        ```python
        xml = json_.to_labelImg(poly2rect=True)
        xml.save()
        ```
        ![image](example/sample_xml2.png)
    
    3. 用 LabelmeJSON.to_labelImg(), 僅將 label 為"p"的 polygon 轉換成  rectangle
        ```python
        xml = json_.to_labelImg(poly2rect=True, poly2rect_labels=['p'])
        xml.save()
        ```
        ![image](example/sample_xml3.png)


# API
## image
- image.ImageFile()
- image.Image()

## shape
分別對應 imgaug.augmentables 的 Keypoint / BoundingBox / Polygon
- shape.Point()
- shape.Rectangle()
- shape.Polygon()

## ShapesOnImage
集合 imgaug.augmentables 裡面的 KeypointsOnImage / BoundingBoxesOnImage / PolygonsOnImage
- ShapesOnImage(_self, from_iaa=None, shapes=None, image=None_)
- property
  - .dict
  - .shapes / .shape_dict
- method
  - .append() / .extend()
  - .draw_on_image(_image=None, inplace=False_)
  - .iaa(_func, *args, **kwargs_)

## AppBase
各種 annotation tool 產生的 annotation 格式
### LabelImgXML()
- property
  - .imgpath / .imgw / .imgh / .imgfile
  - .shapes / .shape_dict
  - .labels
  - .iaa / .soi
- method
  - .parse()
  - .save(_dst=None_)
  - .from_(_self, img_path, shapes=None, database='unknown', **kwargs_)
  - .from_array(_self, rgb, imgpath, shapes, database='unknown', **kwargs_)

### LabelmeJSON()
- property
  - .imgpath / .imgw / .imgh / .imgfile
  - .shapes / .shape_dict
  - .labels
  - .iaa / .soi
- method
  - .parse()
  - .save(_dst=None_)
  - .**to_labelImg**(_self, poly2rect=False, poly2rect_labels=None, xml_path=None_)
  - .from_(*self, img_path, shapes=None, flags=None, save_imageData=False*)
  - .from_array(*self, rgb, imgpath, shapes, flags=None*)

### YoloTXT()
- property
  - .imgpath / .imgw / .imgh / .imgfile
  - .shapes / .shape_dict
  - .labels
  - .iaa / .soi
- method
  - .parse()
  - .save(_dst=None_)
  - from_array(_self, rgb, shapes, **kwargs_)

### RetinaFaceLine()
### RetinaFaceTXT()

## Augmenter
- FolderAugmenter(*self, src_root, src_type, walk=True*)
  - src_root : folder path
  - src_type : one of 'labelme' / 'labelImg' / 'yolo'
  - walk: bool
- property:
  - .relpaths
  - .annpaths
  - .imgpaths
- methods:
  - augment(self, seq, dst_root=None, dst_type=None, prefix=None, postfix=None, overwrite=False):
    - seq: iaa.augmenter / iaa.seq
    - dst_root: aug後的圖檔/標記檔路徑，若為 None 則會存在 src_root
    - prefix/postfix: aug後圖檔/標記檔檔名
    - dst_type: None / 'labelme' / 'labelImg' / 'yolo'
    - overwrite: bool, 若為 False 則在目標路徑存在時 raise FileExistsError
  - augment_batches(_self, seq,
                    batch_size, multicore=True,
                    dst_imgroot=None,
                    dst_annroot=None, dst_anntype=None,
                    prefix=None, postfix=None, overwrite=False,
                    desc=None,
                    yolo_labels=None_) / .imgw / .imgh / .imgfile
  - .shapes / .shape_dict
  - .labels
  - .iaa / .soi
- method
  - .parse()
  - .save(_dst=None_)
  - .from_(_self, img_path, shapes=None, database='unknown', **kwargs_)
  - .from_array(_self, rgb, imgpath, shapes, database='unknown', **kwargs_)
