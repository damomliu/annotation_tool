from annotation import ImageFile, LabelImgXML, LabelmeJSON
from annotation.shape import Point, Rectangle, Polygon

def json2xml_plate_poly2rect():
    json_path = 'example/img320191126_TaipeiRoadNH_night1.json'
    xml_path = json_path.replace('.json', '.xml')
    
    json_ = LabelmeJSON(json_path)
    xml_ = LabelImgXML(xml_path, check_exist=False)
    
    rects = [sh for sh in json_.shape_dict['rectangle']]
    for poly in json_.shape_dict['polygon']:
        if poly.label in ['p']:
            rects.append(poly.as_rectangle)
            
    xml_.from_(img_path=json_.imgpath, shapes=rects)

def json2xml():
    json_path = 'example/img320191126_TaipeiRoadNH_night1.json'
    json_ = LabelmeJSON(json_path)
    xml = json_.to_labelImg(poly2rect=True, poly2rect_labels=['c'])
    
def rotate_json():
    json_path = 'example/img320191126_TaipeiRoadNH_night1.json'
    json_ = LabelmeJSON(json_path)
    
    colors = {'keypoints': (255,255,0), 'bounding_boxes': (0,255,0), 'polygons': (0,255,255)}
    drawimg = ImageFile(json_.imgpath).rgb
    for k,oi in json_.iaa.items():
        drawimg = oi.draw_on_image(drawimg, colors[k])
    
    import imgaug as ia
    import imgaug.augmenters as iaa
    ia.imshow(drawimg)
    
    rgb = ImageFile(json_.imgpath).rgb
    rot = iaa.Rotate(30)
    rotrgb,kpoi,boxoi,polyoi = rot.augment(image=rgb, **json_.iaa)
    
    draw_rotrgb = rotrgb.copy()
    for oi,color in zip([kpoi,boxoi,polyoi], [(255,255,0),(0,255,0),(0,255,255)]):
        draw_rotrgb = oi.draw_on_image(draw_rotrgb, color)
    ia.imshow(draw_rotrgb)
    
    