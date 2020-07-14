import os
import numpy as np
import cv2
import base64
import shutil
import random
from xml.dom import minidom  
import xml.etree.ElementTree as ET
import xml
import codecs
import math
import itertools
import json


class LP_aug():
    def __init__(self,image_path,json_path,xml_path):
        self.image_path = image_path
        self.json_path = json_path
        self.xml_path = xml_path
    
    def read_json(self):
        with open(self.json_path,'r') as f:
            json_file = json.load(f)

        p1 = []
        p2 = []
        p3 = []
        p4 = []
        info = json_file['shapes']    
        for i in range(len(info)):
            
            if info[i]['label']=='p1':
                p1.append(info[i]['points'][0])

            if info[i]['label']=='p2':
                p2.append(info[i]['points'][0])

            if info[i]['label']=='p3':
                p3.append(info[i]['points'][0])
            
            if info[i]['label']=='p4':
                p4.append(info[i]['points'][0])

        label_list = ['p1','p2','p3','p4']
        # points = [['p1', p1] , ['p2',p2], ['p3',p3] ,['p4',p4] ]
        points = p1 + p2 + p3 + p4 

        return points
    
    #self,image_path,points
    def write_json(self,image_path, points, img_shape):
        repeat_time = int(len(points)/4)
    
        labels = sorted(['p1','p2','p3','p4']*repeat_time)
        f = open(image_path[:-3] + "json", "w")    
        with open(image_path, mode='rb') as d_file:
            img_str = base64.b64encode(d_file.read()).decode('utf-8')
                    
        shape = []
        for i in range(len(points)):
            shape.append({
                "shape_type": "point",
                "label": labels[i],
                "line_color": None,
                "fill_color": None,
                "points": [[
                            points[i][0],
                            points[i][1]
                        ]],
                "flags": {}
            })
        data = {
            "version":     "3.16.2",
            "flags":       {},
            "shapes":      shape,
            "lineColor":   [0, 255, 0, 128],
            "fillColor":   [255, 0, 0, 128],
            "imagePath":   os.path.abspath(image_path),
            "imageData":   img_str,
            "imageHeight": img_shape[0],
            "imageWidth":  img_shape[1]
        }

        json_str = json.dumps(data, sort_keys=False, indent=4, separators=(',', ': '))
        f.write(json_str)
        f.close()
        
        return json_str

    def readXml(self):
            
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        label = [i.text for i in root.iter('name')]

        if len(label)>0:
            file_path = [i.text for i in root.iter('filename')][0]

            shape = [0,0]
            shape[0] = [float(i.text) for i in root.iter('width')][0]
            shape[1] = [float(i.text) for i in root.iter('height')][0]

            xmin = [float(i.text) for i in root.iter('xmin')]
            ymin = [float(i.text) for i in root.iter('ymin')]
            xmax = [float(i.text) for i in root.iter('xmax')]
            ymax = [float(i.text) for i in root.iter('ymax')]

            point_list = []
            p_list = []

            for i in range(len(label)):
                image_path = self.xml_path[:-4] + '.jpg'
                class_type = label[i]
                points = [label[i], [float(xmin[i]),float(ymin[i])] , [float(xmax[i]),float(ymin[i])] ,
                        [float(xmin[i]),float(ymax[i])] , [float(xmax[i]),float(ymax[i])]]
                p = [[float(xmin[i]),float(ymin[i])] ,[float(xmax[i]),float(ymax[i])]]
                point_list.append(points)
                p_list.append(p)

        return point_list



    def write_xml(self,image_path,points):

        if image_path[-3:] in ['jpg','png']:

            img = cv2.imread(image_path)
            height, width, depth = img.shape[0], img.shape[1], img.shape[2]
            
            image_name = image_path.split('/')[-1]

            #start to create xml
            annotation = ET.Element('annotation')
            ET.SubElement(annotation, 'filename').text = image_name
            ET.SubElement(annotation, 'path').text = image_path
            
            size = ET.SubElement(annotation, "size")
            ET.SubElement(size, "width").text = str(width)
            ET.SubElement(size, "height").text = str(height)
            ET.SubElement(size, "depth").text = str(depth)
            
            for p in points:
                obj_box = ET.SubElement(annotation, "object")
                class_type = p[0]
                ET.SubElement(obj_box, "name").text = class_type
                bndbox = ET.SubElement(obj_box, 'bndbox')
                ET.SubElement(bndbox, "xmin").text = str(int(p[1][0]))
                ET.SubElement(bndbox, "ymin").text = str(int(p[1][1]))
                ET.SubElement(bndbox, "xmax").text = str(int(p[4][0]))
                ET.SubElement(bndbox, "ymax").text = str(int(p[4][1]))
            
            tree = ET.ElementTree(annotation)
            tree.write(image_path[:-4] + ".xml")
            
            # format xml file
            dom = xml.dom.minidom.parse(image_path[:-4] + ".xml")
            f = codecs.open(image_path[:-4] + ".xml", 'w', 'utf-8') 
            dom.writexml(f, addindent='  ', newl='\n', encoding = 'utf-8')  
            f.close()
            cv2.imwrite(image_path[:-4] + ".jpg",img)
            print('img_path:',image_path)


    def get_rotation(self,image,angle):
        (h, w) = image.shape[:2]
        center = (w/2,h/2)
        Rotation_Matrix = cv2.getRotationMatrix2D(center, angle, scale=1)
        image_rotation = cv2.warpAffine(image, Rotation_Matrix, (w, h))

        return image_rotation,Rotation_Matrix


    def get_projection(self,image,Direction,theta):
        h,w,_ = image.shape
        add_x = w/312*100
        add_y = h/312*100

        if Direction == "right":
            pst1 = np.float32([ [0,0],[w + add_x,-add_y],[0,h],[w + add_x,h + add_y] ])
            pst2 = np.float32([ [theta/1.5,theta/1.5],[w - theta/2, 0 + theta],[theta/1.5,h-theta/1.5],[w - theta/2,h - theta] ]) # right project

        elif Direction == "left":
            pst1 = np.float32([ [-add_x,-add_y],[w,0],[-add_x,h + add_y],[w,h]])
            pst2 = np.float32([ [0 + theta ,0 + theta],[w-theta/1.5,theta/1.5],[0 + theta,h - theta],[w-theta/1.5,h-theta/1.5]]) # left project
        
        elif Direction == "upper":
            pst1 = np.float32([ [-add_x,-add_y],[w + add_x,-add_y],[0,h],[w,h]])
            pst2 = np.float32([ [0 ,0 + theta],[w,0 + theta],[0,h - theta/3],[w,h - theta/3]]) # upper project
        
        elif Direction == "lower":
            pst1 = np.float32([ [0,0],[w,0],[-add_x,h + add_y],[w + add_x,h + add_y]])
            pst2 = np.float32([ [0,0 + theta/3],[w,0 + theta/3],[0 ,h - theta],[w,h- theta]]) # bottom project

        Perspective_Matrix = cv2.getPerspectiveTransform(pst1,pst2)
        image_projection = cv2.warpPerspective(image,Perspective_Matrix,(w,h))

        return image_projection,Perspective_Matrix

    def points_transform(self,points,matrix,Direction,rotation_angle):
        matrix_t = np.array(matrix).transpose()
        k1 = points
        for p in range(len(k1)):
            
            k1_trans = np.matmul(k1[p] + [1],matrix_t)

            if Direction is not None:
                k1[p][0] = k1_trans[0]/k1_trans[2]
                k1[p][1] = k1_trans[1]/k1_trans[2]

            if rotation_angle != 0:
                k1[p][0] = k1_trans[0]
                k1[p][1] = k1_trans[1]
        return k1
    
    def box_points_transform(self,points,matrix,Direction,rotation_angle):
        matrix_t = np.array(matrix).transpose()
        k1 = points
        for p in range(len(k1)):
            for i in range(1,5):
                k1_trans = np.matmul(k1[p][i] + [1],matrix_t)

                if Direction is not None:
                    k1[p][i][0] = k1_trans[0]/k1_trans[2]
                    k1[p][i][1] = k1_trans[1]/k1_trans[2]

                if rotation_angle != 0:
                    k1[p][i][0] = k1_trans[0]
                    k1[p][i][1] = k1_trans[1]
        return k1

   
    def points_bbox(self,points): # correct the bbox location
        k1 = points
                
        for p in range(len(k1)):
            x_list = []
            y_list = []

            for i in range(1,5):
                x_list.append(k1[p][i][0])
                y_list.append(k1[p][i][1])
     
            k1[p][1][0] = int(min(x_list))
            k1[p][1][1] = int(min(y_list))
            k1[p][2][0] = int(max(x_list))
            k1[p][2][1] = int(min(y_list))
            k1[p][3][0] = int(min(x_list))
            k1[p][3][1] = int(max(y_list))
            k1[p][4][0] = int(max(x_list))
            k1[p][4][1] = int(max(y_list))

        return k1
        

    def image_tranform_xml(self,Direction1,theta1,Direction2,theta2,rotation_angle):
        image = cv2.imread(self.image_path)

        points_box = self.readXml()
        
        if Direction1 in ["left","right"]:
            image,Perspective_Matrix1 = self.get_projection(image,Direction1,theta1)
            points_box = self.box_points_transform(points_box,Perspective_Matrix1,Direction1,0)


        if Direction2 in ["upper","lower"]: 
            image,Perspective_Matrix2 = self.get_projection(image,Direction2,theta2)
            points_box = self.box_points_transform(points_box,Perspective_Matrix2,Direction2,0)
           
        # rotation
        image_rotation,Rotation_Matrix = self.get_rotation(image,rotation_angle)

        points_box = self.box_points_transform(points_box,Rotation_Matrix,None,rotation_angle)

        points_box = self.points_bbox(points_box)
            
        return image_rotation,points_box


    def image_tranform_json(self,Direction1,theta1,Direction2,theta2,rotation_angle):
        image = cv2.imread(self.image_path)

        points = self.read_json()
        
        if Direction1 in ["left","right"]:
            image,Perspective_Matrix1 = self.get_projection(image,Direction1,theta1)
            points = self.points_transform(points,Perspective_Matrix1,Direction1,0)


        if Direction2 in ["upper","lower"]: 
            image,Perspective_Matrix2 = self.get_projection(image,Direction2,theta2)
            points = self.points_transform(points,Perspective_Matrix2,Direction2,0)
           
        # rotation
        image_rotation,Rotation_Matrix = self.get_rotation(image,rotation_angle)
        points = self.points_transform(points,Rotation_Matrix,None,rotation_angle)
        return image_rotation,points

def Create_AugProjectSet(input_path,read_type,Direction1,theta1_range,Direction2,theta2_range,rotation_angle_range,random_s,sample):
    output_path = input_path + '_aug'

    try:
        os.mkdir(output_path)
    except:
        pass
    
    img_list = [i for i in os.listdir(input_path) if i[-3:] == 'jpg']
    for img in img_list:
        image_path = input_path + '/' + img

        if 'xml' in read_type:
            xml_path = input_path + '/' + img[:-3] + 'xml'
            json_path = ''
        elif 'json' in read_type:
            json_path = input_path + '/' + img[:-3] + 'json'
            xml_path = ''
        
        img_shape = cv2.imread(image_path).shape

        # shutil.copy(image_path,output_path + "/" + img) # copy origin jpg
        # shutil.copy(xml_path,output_path + "/" + img[:-3] + 'xml') # copy origin xml

        aug_class = LP_aug(image_path,json_path,xml_path)
    

        if random_s == True:

            for num in range(sample):
                i = random.randint(0,len(Direction1)-1)
                j = random.randint(0,len(Direction2)-1)

                if Direction1[i] == None and Direction2[j] == None:
                    dir1 = [x for x in Direction1 if x != None]
                    dir2 = [x for x in Direction2 if x != None]
                    i = random.randint(0,len(Direction1)-1)
                    j = random.randint(0,len(Direction2)-1)


                theta1 = random.randint(theta1_range[0],theta1_range[1])
                theta2 = random.randint(theta2_range[0],theta2_range[1])
                
                rotation_angle = random.randint(rotation_angle_range[0],rotation_angle_range[1])

               
                write_img = output_path + '/' + img[:-4] + "_p" + str(num) + '.jpg'
                
                
                if 'xml' in read_type:
                    image_rotation,points_box = aug_class.image_tranform_xml(Direction1[i],theta1,Direction2[j],theta2,rotation_angle)
                    cv2.imwrite(write_img,image_rotation)
                    aug_class.write_xml(write_img, points_box)
                    
                elif 'json' in read_type:
                    image_rotation,points = aug_class.image_tranform_json(Direction1[i],theta1,Direction2[j],theta2,rotation_angle)
                    cv2.imwrite(write_img,image_rotation)
                    aug_class.write_json(write_img, points, img_shape)
                
                print(write_img)
        
        else:
    
            for i in range(len(Direction1)):
                for j in range(len(Direction2)):

                    if Direction1[i] == None and Direction2[j] == None:
                        pass

                    theta1 = random.randint(theta1_range[0],theta1_range[1])
                    theta2 = random.randint(theta2_range[0],theta2_range[1])
                    
                    rotation_angle = random.randint(rotation_angle_range[0],rotation_angle_range[1])

                    write_img = output_path + '/' + img[:-4] + "_p" + str(i) + "_" + str(j) + '.jpg' # img[:-4]
                    
                    if 'xml' in read_type:
                        image_rotation,points_box = aug_class.image_tranform_xml(Direction1[i],theta1,Direction2[j],theta2,rotation_angle)
                        cv2.imwrite(write_img,image_rotation)
                        aug_class.write_xml(write_img, points_box)
                    elif 'json' in read_type:
                        image_rotation,points = aug_class.image_tranform_json(Direction1[i],theta1,Direction2[j],theta2,rotation_angle)
                        cv2.imwrite(write_img,image_rotation)
                        aug_class.write_json(write_img, points, img_shape)

                    
                    print(write_img)
        


Direction1 = ["left","right"]
theta1_range = [5,10]
Direction2 = ["upper","lower"]
theta2_range = [5,10]
rotation_angle_range = [-5,5]
sample = 5
random_s = False
read_type = ['xml']

input_path = "/home/eddylin/Desktop/plate_generator_special/Fake_special"
Create_AugProjectSet(input_path,read_type,Direction1,theta1_range,Direction2,theta2_range,rotation_angle_range,random_s,sample)

