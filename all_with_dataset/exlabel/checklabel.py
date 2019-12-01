import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(box[0].text)
        y_min = int(box[1].text)
        x_max = int(box[2].text)
        y_max = int(box[3].text)
        coords.append([x_min, y_min, x_max, y_max, name])
    return coords

def show_pic(img, bboxes=None):
    
    # cv2.imwrite('./1.jpg', img)
    # img = cv2.imread('./1.jpg')
    # os.remove('./1.jpg')

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),1)
    cv2.imshow('pic', img)
    pass

if __name__ == "__main__":
    # pics_path = 'data/data'
    # xmls_path = 'data/data'
    pics_path = "../chicken_detect/dataset/farm_test/"
    xmls_path = "../chicken_detect/dataset/farm_test/"

    cv2.namedWindow('pic', 0)  # 1表示原图
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200,800)  # 可视化的图片大小

    for parent, _, files in os.walk(pics_path):
        files = [i for i in files if i.endswith('.jpg') or i.endswith('.png')]
        files = sorted(files)
        for file in files:
            pic_path = os.path.join(parent, file)
            xml_path = os.path.join(xmls_path, file[:-4]+'.xml')
            
            # 解析得到box信息，格式为[[x_min,y_min,x_max,y_max,name]]
            coords = parse_xml(xml_path)        
            coords = [coord[:4] for coord in coords]

            img = cv2.imread(pic_path)
            show_pic(img, coords)    # 原图

            if cv2.waitKey(-1) == 27:
                print("press [ ESC ]")
                break
    cv2.destroyAllWindows()
    pass