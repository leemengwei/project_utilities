import numpy as np
import json
from IPython import embed
import os,sys
import cv2
import tqdm
from libs.labelFile import LabelFile
import glob
labelFile = LabelFile()

def generate_xml(images):
    #take out these:
    for image_left in tqdm.tqdm(images):
        image_left = image_left.split('/')[-1]
        this = image_path+image_left
        if to_gray:
            data = cv2.imread(this, cv2.IMREAD_GRAYSCALE)
        else:
            data = cv2.imread(this)
        h_scaler = np.ceil(540/data.shape[0])
        w_scaler = np.ceil(960/data.shape[1])
        blurer = int(max(3, max(data.shape)/65))
        data = np.tile(data, (int(h_scaler), int(w_scaler)))
        data = data[:540,:960]
        data = cv2.blur(data,(blurer,blurer))
        image_save_as = "./tmp/%s"%(image_left.replace(alter_image_format[0], alter_image_format[1]))
        cv2.imwrite(image_save_as, data)
        labelFile.savePascalVocFormat(image_save_as.replace(alter_image_format[1],".xml"), [], image_save_as, data)   #Use one channel gray image here.
    
if __name__ == "__main__":
    image_path = sys.argv[1]
    to_gray = True
    alter_image_format = ['.jpg', '.png']
    assert os.path.exists('./tmp/')
    os.system("rm ./tmp/*")

    images = glob.glob(image_path+"*.png")
    xmls = glob.glob(image_path+"*.xml")
    images_with_xml = '|'.join(xmls).replace('.xml',alter_image_format[1]).split('|')
    images_without_xml = list(set(images) - set(images_with_xml))
    #print(images)
    generate_xml(images_without_xml)


