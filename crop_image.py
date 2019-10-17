import cv2
import os,sys,time
from IPython import embed
import json
import glob
import labelme2coco
import numpy as np
import matplotlib.pyplot as plt
#img_path = "/home/user/storage/kangni/cv_detection/data/camera/image-top/positive/"
#img_path = "/home/user/storage/kangni/cv_detection/data/mate20pro/images/"
img_path = "."
files = os.popen("ls %s/*.jpg"%img_path).read().split('\n')
files.pop()

for file_index,file in enumerate(files[:]):
    img = cv2.imread("%s"%file)
    with open(file.replace('.jpg','.json'),'r') as f:
        shapes = json.load(f)['shapes']
    for shape_index, shape in enumerate(shapes):
#        embed()
        x1 = shape['points'][0][0]
        x2 = shape['points'][1][0]
        y1 = shape['points'][1][1]
        y2 = shape['points'][2][1]
        xmin = min(x1,x2)
        xmax= max(x1,x2)
        ymin = min(y1,y2)
        ymax = max(y1,y2)
        crop_img = img[ymin:ymax, xmin:xmax]
        print("On file:(%s of %s, shape %s)"%(file_index,len(files),shape_index), file)
        cv2.imwrite('%s_%s.jpg'%(file.split('/')[-1].replace('.jpg',''),shape_index), crop_img)
#        cv2.imshow("cropped", crop_img)
        cv2.waitKey(0)
        
    
known_json_files = '*/*.json'
images_path = './images/'

reassign_list = None
reassign_list = {"head":"head", "head1":"head", "head2":"head", "head3":"head", "head4":"head", "head5":"head", "head6":"head", "angle_r":"angle", "angle":"angle", "top_r":"top", "top":"top"}
something_dont_want = None
something_dont_want = ['..', 'bmp']

jsons = glob.glob(known_json_files)
runner = labelme2coco.labelme2coco(jsons, None, reassign_list, something_dont_want)
runner.data_transfer()

#Crop out not labeled data.
print("re-aranging data...")
ids = []
cats = []
bboxs = np.empty(shape=(0,4))
paths = {}
_lefts_ = {}
_rights_ = {}
_ups_ = {}
_downs_ = {}
for i in range(len(runner.images)):
    paths[str(runner.images[i]['id'])] = runner.images[i]['file_name']
    _lefts_[runner.images[i]['file_name']] = []
    _rights_[runner.images[i]['file_name']] = []
    _ups_[runner.images[i]['file_name']] = []
    _downs_[runner.images[i]['file_name']] = []
for i in range(len(runner.annotations)): 
    runner.annotations[i]['segmentation'] = np.array(runner.annotations[i]['segmentation'], dtype=np.int).tolist()
    ids.append(_paths_[str(runner.annotations[i]['image_id'])])
    cats.append(runner.annotations[i]['category_id'])
    bboxs = np.vstack((bboxs, runner.annotations[i]['bbox']))
for idx, name in enumerate(ids):
    #data = plt.imread(images_path+name)
    _lefts_[name].append(bboxs[idx][0])
    _rights_[name].append(bboxs[idx][0]+bboxs[idx][2])
    _ups_[name].append(bboxs[idx][1])
    _downs_[name].append(bboxs[idx][1]+bboxs[idx][3])
crop_fields = {}
for i in ids:
    plt.clf()
    _lefts_[i] = np.array(_lefts_[i]).min()
    _rights_[i] = np.array(_rights_[i]).max()
    _ups_[i] = np.array(_ups_[i]).min()
    _downs_[i] = np.array(_downs_[i]).max()
    crop_fields[i] = [int(np.array(_ups_[i]).min()), int(np.array(_downs_[i]).max()), int(np.array(_lefts_[i]).min()), int(np.array(_rights_[i]).max())]
    data = plt.imread(images_path+i)
    base = np.random.random(data.shape)
    base[crop_fields[i][0]:crop_fields[i][1],crop_fields[i][2]:crop_fields[i][3]] = data[crop_fields[i][0]:crop_fields[i][1],crop_fields[i][2]:crop_fields[i][3]]
    imshow(base)
    plt.pause(0.001)

# abort, this is useless


#embed()
