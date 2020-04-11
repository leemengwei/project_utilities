import pandas as pd
import numpy as np
import os,time,sys
from IPython import embed
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
import copy
import glob
import tqdm
#import textract
import xml.etree.ElementTree as ET

def show_me_input_data(pd_data):
    del(pd_data['Temp'])
    assert isinstance(pd_data, pd.core.frame.DataFrame), "I would like pandas with column names"
    pd_target_data = pd.read_csv("/mfs/home/limengwei/friction_compensation/data/planning_simulator.csv",sep=' ',index_col=None)[:-1][pd_data.columns].astype(float)
    data = pd_data.values
    target_data = pd_target_data.values
    #embed()

    f1, ax1 = plt.subplots()
    f2, ax2 = plt.subplots()
    #Distribution:
    ax1.plot(data[::max(data.shape)//1000].T, c='r', linestyle='--', alpha=0.2)
    ax2.plot(target_data[::max(target_data.shape)//1000].T, c='b', linestyle='--', alpha=0.2)
    #Channel:
    ax1.plot(data.min(axis=0), 'k', label="train_min_max")
    ax1.plot(data.max(axis=0), 'k')
    ax2.plot(target_data.min(axis=0), c='k', label="simulator_min_max", alpha=0.8)
    ax2.plot(target_data.max(axis=0), c='k', alpha=0.8)
    #Misc:
    ax1.set_xticks(list(range(25-1)))   #Temp is del here.
    ax1.set_xticklabels(list(pd_data.columns), rotation=90, fontsize=5)
    ax2.set_xticks(list(range(25-1)))   #Temp is del here.
    ax2.set_xticklabels(list(pd_data.columns), rotation=90, fontsize=5)
    f1.legend()
    f2.legend()

    for idx,line in enumerate(ax1.lines): 
        line.set_color(plt.cm.Spectral_r(np.linspace(0,1,100))[idx])
    lineobj = plt.plot(data.T[:,:]) 
    plt.legend(iter(lineobj), list(range(25)))
    return

#project friction compensation
def performance_shape(raw_plan, inputs, model_part1):
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(212, projection='3d')
    #General:
    new = np.tile(inputs.detach().numpy().mean(axis=0), (1000,1))
    v_idx = 12
    p_idx = 13
    mannual_v = np.linspace(-inputs.detach().numpy().min(axis=0)[v_idx], inputs.detach().numpy().min(axis=0)[v_idx], 1000)
    mannual_p = np.linspace(-inputs.detach().numpy().min(axis=0)[p_idx], inputs.detach().numpy().min(axis=0)[p_idx], 1000)
    #F-v curve:
    new_v = copy.deepcopy(new)
    new_v[:,v_idx] = mannual_v
    output_v =  model_part1(torch.FloatTensor(new_v)).detach().numpy()
    ax1.plot(mannual_v, output_v)
    ax1.set_xlabel("Normalized speed")
    ax1.set_ylabel("Normalized compensation")
    ax1.set_title("v-F curve")
    #F-pos curve:
    new_p = copy.deepcopy(new)
    new_p[:,p_idx] = mannual_p
    output_p = model_part1(torch.FloatTensor(new_p)).detach().numpy()
    ax2.plot(mannual_p, output_p)
    ax2.set_xlabel("Normalized position")
    ax2.set_ylabel("Normalized compensation")
    ax2.set_title("p-F curve")
    #F-v-pos surface:
    new_vp = np.tile(new, (1000,1))
    new_vp[:,p_idx] = np.random.uniform(-inputs.detach().numpy().min(axis=0)[v_idx], inputs.detach().numpy().min(axis=0)[v_idx], 1000*1000)
    new_vp[:,v_idx] = np.random.uniform(-inputs.detach().numpy().min(axis=0)[p_idx], inputs.detach().numpy().min(axis=0)[p_idx], 1000*1000)
    output_vp = model_part1(torch.FloatTensor(new_vp)).detach().numpy()
    ax3.scatter3D(new_vp[:,v_idx][::100], new_vp[:,p_idx][::100], output_vp.flatten()[::100], s=0.5, alpha=0.5, c=output_vp.flatten()[::100], cmap='Spectral_r')
    ax3.set_xlabel("Normalized speed")
    ax3.set_ylabel("Normalized position")
    ax3.set_zlabel("Normalized compensation")
    ax3.set_title("v-p-F surface")
    plt.show()

    sys.exit()
    return

#Project car_face
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
def show_bbox_sizes():
    BACK_xmls = glob.glob('./*/*/BACK*.xml')
    FRONT_xmls = list(set(glob.glob('./*/*/*.xml'))-set(BACK_xmls))
    #def convert(xml_dir, json_file, reassign_list, label_to_id):
    list_fp = FRONT_xmls
    list_fp = BACK_xmls
    json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": []}
    heights = []
    widths = []
    categories = {}
    label_to_id = {"angle":1, "top":2, "head":3}
    bnd_id = 0
    for idx,line in tqdm.tqdm(enumerate(list_fp), total=len(list_fp)):
        line = line.strip()
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
        if len(filename)>60:
            print("%s is too long, Might from windows? using %s instead"%(filename, filename.split('\\')[-1]), "you might want to check output.xml")
        filename = filename.split('\\')[-1]
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
        #  aissert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
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
            widths.append(o_width)
            heights.append(o_height)
            #embed()
            ann = {'area': o_width*o_height, 'o_width':o_width, 'o_height':o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
    #For my statistics:
    areas = []
    o_widths = []
    o_heights = []
    for i in json_dict['annotations']: 
        if i['category_id']==3:
            areas.append(i['area']) 
            o_widths.append(i['o_width']) 
            o_heights.append(i['o_height'])


    embed()


    return



if __name__ == "__main__":
    print("NN input Analyzer")
    print("Call me in your scripts...")
    show_bbox_sizes()
