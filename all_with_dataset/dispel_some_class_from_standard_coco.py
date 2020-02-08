import json
from IPython import embed
import os
import cv2
import tqdm
from libs.labelFile import LabelFile
labelFile = LabelFile()

image_path = '/mfs/home/data/coco/2017/MScocoDATA/val2017/'
data = json.load(open('instances_val2017.json','r'))
alter_image_format = ['jpg', 'png']   #jpg to png
to_gray = True
assert os.path.exists('./tmp_taken/')

#category dont want:
imageid_dont_want=[]
id_not_want=[1,3,6,8]  #person car bus truck 
for parts in data['annotations']:
    if parts['category_id'] in id_not_want:
        imageid_dont_want.append(parts['image_id'])
imageid_dont_want = set(imageid_dont_want)

#images left:
image_left = []
for img_blob in data['images']:
    if img_blob['id'] not in imageid_dont_want:
        image_left.append(img_blob)

#take out these:
for left in tqdm.tqdm(image_left):
    this = image_path+left["file_name"]
    if to_gray:
        data = cv2.imread(this, cv2.IMREAD_GRAYSCALE)
    else:
        data = cv2.imread(this)
    image_save_as = "./tmp_taken/%s"%(left['file_name'].replace(alter_image_format[0], alter_image_format[1]))
    cv2.imwrite(image_save_as, data)
    labelFile.savePascalVocFormat(image_save_as.replace(alter_image_format[1],"xml"), [], image_save_as, data)   #Use one channel gray image here.


