import cv2
import os,sys,time
from IPython import embed
import json
import glob
import labelme2coco
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import resize, rescale
from tqdm import tqdm
#img_path = "/home/user/storage/kangni/cv_detection/data/camera/image-top/positive/"
#img_path = "/home/user/storage/kangni/cv_detection/data/mate20pro/images/"
#img_path = "."
#files = os.popen("ls %s/*.jpg"%img_path).read().split('\n')
#files.pop()
#
#for file_index,file in enumerate(files[:]):
#    img = cv2.imread("%s"%file)
#    with open(file.replace('.jpg','.json'),'r') as f:
#        shapes = json.load(f)['shapes']
#    for shape_index, shape in enumerate(shapes):
##        embed()
#        x1 = shape['points'][0][0]
#        x2 = shape['points'][1][0]
#        y1 = shape['points'][1][1]
#        y2 = shape['points'][2][1]
#        xmin = min(x1,x2)
#        xmax= max(x1,x2)
#        ymin = min(y1,y2)
#        ymax = max(y1,y2)
#        crop_img = img[ymin:ymax, xmin:xmax]
#        print("On file:(%s of %s, shape %s)"%(file_index,len(files),shape_index), file)
#        cv2.imwrite('%s_%s.jpg'%(file.split('/')[-1].replace('.jpg',''),shape_index), crop_img)
##        cv2.imshow("cropped", crop_img)
#        cv2.waitKey(0)
        
def on_pick(event):
    artist = event.artist
    xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
    x, y = artist.get_xdata(), artist.get_ydata()
    ind = event.ind
    print( 'Artist picked:', event.artist)
    print( 'Data point:', x[ind[0]], y[ind[0]], name[i])
   
known_json_files = 'train/*/*.json'
images_path = './train_images/'

reassign_list = None
reassign_list = {"head":"head", "head1":"head", "head2":"head", "head3":"head", "head4":"head", "head5":"head", "head6":"head", "angle_r":"angle", "angle":"angle", "top_r":"top", "top":"top"}
something_dont_want = None
something_dont_want = ['..', 'bmp']
label_to_id = {"angle":1, "top":2, "head":3}

jsons = glob.glob(known_json_files)
print("Getting data...")
runner = labelme2coco.labelme2coco(jsons, None, reassign_list, something_dont_want, label_to_id)
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
    ids.append(paths[str(runner.annotations[i]['image_id'])])
    cats.append(runner.annotations[i]['category_id'])
    bboxs = np.vstack((bboxs, runner.annotations[i]['bbox']))
for idx, name in enumerate(ids):
    #data = plt.imread(images_path+name)
    _lefts_[name].append(bboxs[idx][0])
    _rights_[name].append(bboxs[idx][0]+bboxs[idx][2])
    _ups_[name].append(bboxs[idx][1])
    _downs_[name].append(bboxs[idx][1]+bboxs[idx][3])
all_stuff = {}   #all_stuff is : {"1":[], "2":[], "3":[]}
for i in set(cats):
    all_stuff[str(i)] = []
for index,i in enumerate(cats):
    all_stuff[str(i)].append([ids[index]]+list(np.array(runner.annotations[index]['bbox'], dtype=int)))

objects = np.empty(shape=(50, 0))

#------------Display of Collection of labeled data:-------------
num = int(np.ceil(np.sqrt(len(cats))))
base = np.zeros((num*50, num*50))
i = 0
fig, ax = plt.subplots()
for key in all_stuff.keys():
    for name, a,b,c,d in tqdm(all_stuff[key]):
        data = plt.imread(images_path+name)
        tmp_data = rescale(data[b:b+d, a:a+c], (50/d,50/c))
        base[i//num*50:(i//num+1)*50, i%num*50:(i%num+1)*50] = tmp_data
#        plt.text(i%num*50+25, i//num*50+25, name, size=5, color = 'red')
        i += 1
#----------Display of name:------------
plt.imshow(base)
embed()
fig.canvas.callbacks.connect('pick_event', on_pick)
plt.show()

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
    plt.imshow(base)
    plt.pause(0.001)

# abort, this is useless



