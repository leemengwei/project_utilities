import json
from IPython import embed
import os,sys
import cv2
import tqdm
from libs.labelFile import LabelFile
import glob
labelFile = LabelFile()

def parse_json(json_file):
    if json_file is None:
        print("None json file given, return full image list...")
        images_left = glob.glob("*.png")
        return images_left
    json_data = json.load(open(json_file,'r'))
    #category dont want:
    imageid_dont_want = []
    id_dont_want = []   #[1,3,6,8]
    names_dont_want = ['person', 'car', 'bus', 'truck', 'boat'] 
    for i in json_data['categories']:
        if i['name'] in names_dont_want:
            print("Don't want ", i['name'])
            id_dont_want.append(i['id'])
    
    for parts in json_data['annotations']:
        if parts['category_id'] in id_dont_want:
            imageid_dont_want.append(parts['image_id'])
    print(len(imageid_dont_want), "object dont want", len(set(imageid_dont_want)), "images dropped.")
    imageid_dont_want = set(imageid_dont_want)
    
    #images left:
    images_left = []
    for img_blob in json_data['images']:
        if img_blob['id'] not in imageid_dont_want:
            images_left.append(img_blob['file_name'])
    return images_left

def generate_xml(images_left):
    #take out these:
    for image_left in tqdm.tqdm(images_left):
        this = image_path+image_left
        if to_gray:
            data = cv2.imread(this, cv2.IMREAD_GRAYSCALE)
        else:
            data = cv2.imread(this)
        image_save_as = "./tmp_taken_%s/%s"%(mode, image_left.replace(alter_image_format[0], alter_image_format[1]))
        cv2.imwrite(image_save_as, data)
        labelFile.savePascalVocFormat(image_save_as.replace(alter_image_format[1],".xml"), [], image_save_as, data)   #Use one channel gray image here.
    
if __name__ == "__main__":
    mode = sys.argv[1]
    #image_path = '/mfs/home/data/coco/2017/MScocoDATA/%s2017/'%mode
    image_path = '/mfs/home/limengwei/car_face/car_face/object_detection_data_angle_top_head/pool_exlusive/'
    #json_file = 'instances_%s2017.json'%mode
    json_file = None
    alter_image_format = ['.jpg', '.png']   #jpg to png
    to_gray = True
    assert os.path.exists('./tmp_taken_%s/'%mode)
    os.system("rm ./tmp_taken_%s/*"%mode)

    images_left = parse_json(json_file)
    generate_xml(images_left)


