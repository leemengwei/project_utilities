#!/usr/bin/python

# pip install lxml
import numpy as np
import sys
import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
from IPython import embed
import glob
START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {}
# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
                         #  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
                         #  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                         #  "motorbike": 14, "person": 15, "pottedplant": 16,
                         #  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))

def label_reassign(label, reassign_list):
    if reassign_list is not None:
        #print("Reassigning %s to %s"%(label, reassign_list[label]))
        label = reassign_list[label] 
    else:
        pass
    return label

def convert(xml_dir, json_file, reassign_list, label_to_id):
    #list_fp = open(xml_list, 'r')
    #num = int(os.popen("cat %s|wc -l"%xml_list).read())
    list_fp = glob.glob("./%s/*.xml"%xml_dir)
    json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for idx,line in tqdm(enumerate(list_fp), total=len(list_fp)):
        line = line.strip()
        #print("Processing %s"%(line))
        #xml_f = os.path.join(xml_dir, line)
        xml_f = line
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError('%d paths found in %s'%(len(path), line))
        ## The filename must be a number
        #image_id = get_filename_as_int(filename)
        image_id = idx
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):
            raw_category = get_and_check(obj, 'name', 1).text
            category = label_reassign(raw_category, reassign_list)   #Change from head12345 to head.
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
                categories[category] = label_to_id[category]
                #print("Giving %s %s"%(category, label_to_id[category]))
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    #cat = list(np.array(self.categories)[np.argsort(ids)])
    ids = np.array([i['id'] for i in json_dict['categories']])     #WTF:  I've check all the transfer which is correct. Yet model is still bug with class order. I can only assume that when reading the coco.json in mmdetection, THIS order matters. So I re-sort them here.
    json_dict['categories'] = list(np.array(json_dict['categories'])[np.argsort(ids)])
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict,indent=4, separators=(',', ': '))
    json_fp.write(json_str)
    json_fp.close()
    #list_fp.close()


if __name__ == '__main__':
    reassign_list = None
    reassign_list = {"head":"head", "head1":"head", "head2":"head", "head3":"head", "head4":"head", "head5":"head", "head6":"head", "angle_r":"angle", "angle":"angle", "top_r":"top", "top":"top"}
    label_to_id = {"angle":1, "top":2, "head":3}
    
    #xml_dir = "pool_train"
    assert sys.argv[1], "what is your labelImg generated png&xml path?"
    xml_dir = sys.argv[1]
    convert(xml_dir, '%s.json'%xml_dir.strip('.').strip('/'), reassign_list, label_to_id)
