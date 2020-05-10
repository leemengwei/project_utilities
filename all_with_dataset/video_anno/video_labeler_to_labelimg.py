import os,sys
import numpy as np
import pandas as pd
import time
from IPython import embed
import converter
import glob
import matplotlib.pyplot as plt

import base64
import cv2
import json
import numpy as np
import os
from shapely.geometry import Polygon
import tripy
import xml.etree.ElementTree as ET

XML_FILTER = ".xml"
PNG_FILTER = ".png"

def get_path_to_img(path_to_xml):
    return os.path.abspath(path_to_xml)[:-len(XML_FILTER)] + PNG_FILTER

def add_base_information(root, path_to_xml):
    folder_element = ET.Element('folder')
    folder_element.text = os.path.basename(os.path.split(path_to_xml)[0])
    filename_element = ET.Element('filename')
    filename_element.text = os.path.basename(path_to_xml)[:-len(XML_FILTER)] + PNG_FILTER
    path_element = ET.Element('path')
    path_element.text = get_path_to_img(path_to_xml)
    source_element = ET.Element('source')
    ET.SubElement(source_element, 'database').text = "Unknown"
    size_element = ET.Element('size')
    tmp = plt.imread(path_to_xml.replace(".xml",".jpg"))
    ET.SubElement(size_element, 'width').text = str(tmp.shape[0])
    ET.SubElement(size_element, 'height').text = str(tmp.shape[1])
    ET.SubElement(size_element, 'depth').text = "1"
    segmented_element = ET.Element('segmented')
    segmented_element.text = "0"
    root.append(folder_element)
    root.append(filename_element)
    root.append(path_element)
    root.append(source_element)
    root.append(size_element)
    root.append(segmented_element)

def add_object(root, name, bbox):
    object_element = ET.Element('object')
    ET.SubElement(object_element, 'name').text = name
    ET.SubElement(object_element, 'pose').text = "Unspecified"
    ET.SubElement(object_element, 'truncated').text = "0"
    ET.SubElement(object_element, 'difficult').text = "0"
    bbox_element = ET.SubElement(object_element, 'bndbox')
    ET.SubElement(bbox_element, 'xmin').text = str(bbox[0])
    ET.SubElement(bbox_element, 'ymin').text = str(bbox[1])
    ET.SubElement(bbox_element, 'xmax').text = str(bbox[2])
    ET.SubElement(bbox_element, 'ymax').text = str(bbox[3])
    root.append(object_element)


if __name__ == "__main__":
    print("Start...")
    which_one = sys.argv[1]
    label_txt_file = "../output/labels.txt"
    data_path = "../data/%s/"%which_one
    pics = glob.glob(data_path+"/*.jpg")
    if not os.path.exists("%s"%label_txt_file):
        print("Can't find %s"%label_txt_file)
        sys.exit()
    all_labels = pd.read_csv(label_txt_file, names=['name', 'class', 'x0', 'y0', 'x1', 'y1'], delimiter=" ")
    for idx,i in enumerate(pics):
        pic_name = i.split("/")[-1].split('.')[0]

        path_to_xml = data_path + str(pic_name)+".xml"
        which_lines = all_labels.name == int(pic_name)
        lines = all_labels[which_lines]
        labels = lines['class']
        if labels.shape[0]>0:
            tree = ET.ElementTree(element=ET.Element('annotation'))
            root = tree.getroot()
            add_base_information(root, path_to_xml)
            for j_idx,j in enumerate(labels.values):
                label = j
                print("%s with label %s"%(i, label))
                #parse xml:
                add_object(root, label, [all_labels[which_lines].x0.values[j_idx], all_labels[which_lines].x1.values[j_idx], all_labels[which_lines].y0.values[j_idx], all_labels[which_lines].y1.values[j_idx]])
            tree.write(path_to_xml)
        else:
            print("No label for %s"%i)
            continue   #this image may not be labeled yet.






