from IPython import embed
import numpy as np
import os,sys,time
import cv2
from labelimg2coco import *
from libs.labelFile import LabelFile
import shutil
labelFile = LabelFile()

if __name__ == '__main__':
    assert sys.argv[1], "what is your labelImg generated png path?"
    os.system("rm ./tmp_COVERED/*")
    png_dir = sys.argv[1]
    pngs = glob.glob(png_dir+"/*.png")
    xmls = glob.glob(png_dir+"/*.xml")
    pngs_with_xml = '|'.join(xmls).replace('.xml','.png').split('|')
    pngs_without_xml = list(set(pngs) - set(pngs_with_xml))
    if len(pngs_without_xml)>0:
        print("There %s pngs without xml! Your sure want to move them to ./not_labeled/ ??"%len(pngs_without_xml))
        input()
        input()
        input()
        if not os.path.exists("./not_labeled/"):os.system('mkdir not_labeled')
        for _i in pngs_without_xml:
            shutil.move(_i, "./not_labeled/"+_i.split('/')[-1])
    to_gray = True

    #read gt box positions and image names
    reassign_list = {"head":"head", "head1":"head", "head2":"head", "head3":"head", "head4":"head", "head5":"head", "head6":"head", "angle_r":"angle", "angle":"angle", "top_r":"top", "top":"top"}
    label_to_id = {"angle":1, "top":2, "head":3}
    categories, json_dict = parse_xmls(png_dir, reassign_list, label_to_id)
    #read each image out and cover their bbox
    for this in tqdm(json_dict['images']):
        png = this['file_name']
        image_id = this['id']
        this_bboxes = []
        for i in json_dict['annotations']: 
            if i['image_id'] == image_id: 
                this_bboxes.append(i['bbox'])
        if to_gray:
            data = cv2.imread(png_dir+png, cv2.IMREAD_GRAYSCALE)
            if data is None:   #Just a work around
                print("Skipping Corrupt file? Name exists?", png_dir+png,png_dir+png.replace(".png",'.xml'))
                continue
        else:
            data = cv2.imread(png_dir+png, cv2)
        #cover it
        #if not len(this_bboxes)>0:continue
        #print(this_bboxes,data,png_dir+png)
        for one_bbox in this_bboxes:
            try:
                data[one_bbox[1]:one_bbox[1]+one_bbox[3], one_bbox[0]:one_bbox[0]+one_bbox[2]] = np.random.randint(0,255)
            except:
                print("Error covering bbox")
                sys.exit()
                #os.remove(png_dir+png)
                #os.remove(png_dir+png.replace(".png",'.xml'))
                #os.remove(png_dir+png)
                #os.remove(png_dir+png.replace(".png",'.xml'))
                #os.remove(png_dir+png)
                #os.remove(png_dir+png.replace(".png",'.xml'))
                #os.remove(png_dir+png)
                #os.remove(png_dir+png.replace(".png",'.xml'))
                #os.remove(png_dir+png)
                #os.remove(png_dir+png.replace(".png",'.xml'))
        #save its empty suppresion xml:
        image_save_as = "./tmp_COVERED/COVERED_%s"%png
        cv2.imwrite(image_save_as, data)
        labelFile.savePascalVocFormat(image_save_as.replace(".png",".xml"), [], image_save_as, data)   #Use one channel gray image here.
 


    #embed()





