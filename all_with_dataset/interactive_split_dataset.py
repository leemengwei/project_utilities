import numpy as np
import glob
from IPython import embed
import matplotlib.pyplot as plt
import os,sys
import tqdm
import shutil

file_path = "./mannual_select_train/*"
filelist = glob.glob(file_path)
filelist.sort()
for idx,this_file in tqdm.tqdm(enumerate(filelist[:]), total=len(filelist)):
    images = glob.glob(this_file+"/*.png")
    jsons = glob.glob(this_file+"/*.json")
    if len(jsons)==0:
        shutil.move(this_file, "./not_labeled")
        filelist.remove(this_file)
print("%s file to go"%len(filelist))
plt.ion()
fig = plt.figure(figsize=(18,6))
for idx,this_file in tqdm.tqdm(enumerate(filelist[:]), total=len(filelist)):
    plt.clf()
    images = glob.glob(this_file+"/*.png")
    jsons = glob.glob(this_file+"/*.json")
    if len(jsons)==0:
        continue
    print("num of images:", len(images))
    this_file = this_file.replace(" ", "\ ")
    cmd = 'mv %s %s'%(this_file, './mannual_select_del/')
    n = int(np.sqrt(len(images)))
    axes = fig.subplots(1,2).reshape(-1)
    for idx, this_image in enumerate(images[:6][::3]):
        try:
            data = plt.imread(this_image)
            axes.flatten()[idx].imshow(data, 'gray')
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
            axes.flatten()[idx]
        except Exception as e:
            print(e)
            pass
    #plt.draw()
    key = input()
    #sys.exit()
    if key=='d':
        cmd = 'mv %s %s'%(this_file, "./mannual_select_del/")
        print(cmd)
    else:
        cmd = ''
        print('pass %s pass'%(this_file))
    os.system(cmd)
    
    
    
