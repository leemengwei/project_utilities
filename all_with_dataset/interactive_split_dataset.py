import numpy as np
import glob
from IPython import embed
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

file_path = "./mannual_select_train/*"
filelist = glob.glob(file_path)

plt.ion()
plt.figure(figsize=(10,8))
for idx,this_file in enumerate(tqdm(filelist[:])):
    print(idx, this_file)
    images = glob.glob(this_file+"/*.png")
    jsons = glob.glob(this_file+"/*.json")
    if len(jsons)==0:
        continue
    this_file = this_file.replace(" ", "\ ")
    plt.clf()
    cmd = 'mv %s %s'%(this_file, './mannual_select_del/')
    for this_image in images:
        data = plt.imread(this_image)
        plt.imshow(data, 'gray')
        plt.draw()
        key = input()
        if key == 'g':
            cmd = 'mv %s %s'%(this_file, "./mannual_select_val/")
            print(cmd)
            break
        elif key=='f':
            cmd = ''
            print("check further", this_file)
            continue
        elif key=='d':
            cmd = 'mv %s %s'%(this_file, "./mannual_select_del/")
            print(cmd)
            break
        else:
            cmd = ''
            print('pass %s pass'%(this_file))
            break

    os.system(cmd)
    
    
    
