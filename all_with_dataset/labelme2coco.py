# -*- coding:utf-8 -*-
# !/usr/bin/env python
import os 
import shutil
import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from labelme import utils
import numpy as np
import glob
import PIL.Image
import sys 
from tqdm import tqdm
from IPython import embed
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
 
class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path='./train.json', reassign_list = None, something_dont_want = None, label_to_id = {}):
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.reassign_list = reassign_list
        self.something_dont_want = something_dont_want
        self.label_to_id = label_to_id

    def label_reassign(self, label):
        if self.reassign_list is not None:
            label = self.reassign_list[label] 
        else:
            pass
        return label
 
    def data_transfer(self):
 
        for num, json_file in tqdm(enumerate(self.labelme_json), total=len(self.labelme_json)):
            with open(json_file, 'r') as fp:
                data = json.load(fp)
                try:  
                    self.images.append(self.image(data, num))
                except Exception as e:  
                    print(num, json_file)
                    print("Problem with image json, skipping ...", e)
                    #print("mv %s %s"%(json_file, json_file+"_corrupt"))
                    #shutil.move(json_file, json_file+"_corrupt")
                    continue
                for shapes in data['shapes']:
                    #print(".", end='')
                    sys.stdout.flush()
                    raw_label_name = shapes['label']
                    label = self.label_reassign(raw_label_name)     #Change from head12345 to head and somethings like that.
                    if label not in self.label:
                        self.categories.append(self.get_categorie(label))
                        self.label.append(label)
                    points = shapes['points']#这里的point是用rectangle标注得到的，只有两个点，需要转成四个点
                    points.append([points[0][0],points[1][1]])
                    points.append([points[1][0],points[0][1]])
                    self.annotations.append(self.get_annotation(points, label, num))
                    #print(self.get_annotation(points, label, num), "at %s"%label)
                    self.annID += 1
 
    def image(self, data, num):
        #embed()
        image = {}
        img = utils.img_b64_to_arr(data['imageData']) 
        height, width = img.shape[:2]
        img = None
        image['height'] = height
        image['width'] = width
        image['id'] = num + 1
        image['file_name'] = data['imagePath'].split('/')[-1]
        assert all(i not in image['file_name'] for i in self.something_dont_want)
 
        self.height = height
        self.width = width
 
        return image
 
    def get_categorie(self, label):
        categorie = {}
        categorie['supercategory'] = 'component'
        #categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['id'] = self.label_to_id[label]  #Make id consistent over runs...
        categorie['name'] = label
        print(categorie)
        return categorie
 
    def get_annotation(self, points, label, num):
        annotation = {}
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        annotation['iscrowd'] = 0
        annotation['image_id'] = num + 1
        # annotation['bbox'] = str(self.getbbox(points)) # 使用list保存json文件时报错（不知道为什么）
        # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 使用该方式转成list
        annotation['bbox'] = list(map(float, self.getbbox(points)))
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        annotation['category_id'] = self.getcatid(label)
        annotation['id'] = self.annID
        return annotation
 
    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                #print("returning %s for %s"%(categorie['id'], label))
                return categorie['id']
        print("warning, %s not temporary in %s"%(label, self.categories))
        return 1
 
    def getbbox(self, points):
        polygons = points
 
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)
 
    def mask2box(self, mask):
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x
 
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)
 
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  
 
    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask
 
    def data2coco(self):
        data_coco = {}
        ids = np.array([i['id'] for i in self.categories])     #WTF:  I've check all the transfer which is correct. Yet model is still bug with class order. I can only assume that when reading the coco.json in mmdetection, THIS order matters. So I re-sort them here.
        self.categories = list(np.array(self.categories)[np.argsort(ids)])
        data_coco['categories'] = self.categories
        data_coco['images'] = self.images
        data_coco['annotations'] = self.annotations
        print("Using:%s"%self.categories)
        print("Check THESE mannually if you're not assured:----------------------")
        try:
            print(self.images[0]['file_name'], 'should have at least some of these followed:', self.annotations[0:4])
        except Exception as e:
            print(e)
        return data_coco
 
    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4, cls=MyEncoder)  # indent=4 更加美观显示
 
if __name__ == "__main__": 

    reassign_list = None
    reassign_list = {"head":"head", "head1":"head", "head2":"head", "head3":"head", "head4":"head", "head5":"head", "head6":"head", "angle_r":"angle", "angle":"angle", "top_r":"top", "top":"top"}
    label_to_id = {"angle":1, "top":2, "head":3}
    something_dont_want = []
    something_dont_want = ['..', 'bmp']
    #labelme_json = glob.glob('../data/train_det/*.json')
    #labelme_json = glob.glob('../data/train_seg/*.json')
    #labelme2coco(labelme_json, './train_seg.json')
    #labelme_json = glob.glob('../data/val_seg/*.json')
    #labelme2coco(labelme_json, './val_seg.json')
    
    #labelme_json = glob.glob('../data/train_det/*.json')
    #labelme2coco(labelme_json, './train_det.json')
    #mode = "mannual_select_%s"%sys.argv[1]
    mode = "xml2json"
    labelme_json = glob.glob('./%s/*/*.json'%mode)
    print("jsons:%s"%labelme_json)
    runner = labelme2coco(labelme_json, './%s.json'%mode, reassign_list, something_dont_want, label_to_id)
    runner.save_json()
