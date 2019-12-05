# -*- coding=utf-8 -*- 
import os 
import cv2 


import xml.etree.ElementTree as ET
import os
import shutil
import random 

# 从xml文件中提取bounding box信息和w,h信息, 格式为[[x_min, y_min, x_max, y_max, name]]
def parse_xml(xml_path):
    tree = ET.parse(xml_path)		
    root = tree.getroot()

    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)


    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(box[0].text)
        y_min = int(box[1].text)
        x_max = int(box[2].text)
        y_max = int(box[3].text)
        coords.append([x_min, y_min, x_max, y_max, name])
    return filename, (width, height), coords


# 从xml文件中提取bounding box信息和w,h信息, 格式为[[x_min, y_min, x_max, y_max, name]]
def fix_xml(xml_path, newxml_path, x0,x1,y0,y1):
    tree = ET.parse(xml_path)		
    root = tree.getroot()
 
    size = root.find('size')
    size.find('width').text = str( x1-x0 )
    size.find('height').text = str( y1-y0 )
  
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        box[0].text = str( int(box[0].text) - x0  )
        box[1].text = str( int(box[1].text) - y0  )
        box[2].text = str( int(box[2].text) - x0  )
        box[3].text = str( int(box[3].text) - y0  ) 

    open(newxml_path, "wb").write(  ET.tostring(root) ) 


def write_single_label(xml, class2indice):
    "输入原xml的绝对路径"
    filename, (width, height), coords = parse_xml(xml)
    #"56 0.855422 0.633625 0.087594 0.208542"    
    label = xml[:-3] + "txt"
    f = open(label, 'w')
    for i,coor in enumerate(coords):
        try:
            xmin, ymin, xmax, ymax, c = coor
            x = (xmin + xmax) / 2 / width
            y = (ymin + ymax) / 2 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            ci = class2indice[c]
            line = [str(c),str(x), str(y), str(w), str(h)]
            line = ' '.join(line) + "\n"
            f.write(line)
        except:
            print( 'write_single_label(', xml , '...) met unsupported class:', c)
            pass 
    f.close()
    

def get_random_pos(w,nw,lb0,lb1 , border = 100 ):
    if lb0 < border:
        x0 = 0
        x1 = nw 
        return x0,x1 
    elif  w - lb1 < border:
        x1 = w-1 
        x0 = x1 - nw 
        return x0,x1 
    cnt = 10
    while cnt > 0:
        cnt -= 1
        x0 =   int( (lb0 - border) * random.random() )
        x1 = x0 + nw 
        if x1 > lb1 + border  :
            return x0,x1 
    return None,None 



def fit_img_lb( imgf,xmlf,imgof,xmlof , min_ratio = 0.6 , max_ratio = 0.7 ):
    print( imgf ) 
    filename, (w, h), coords = parse_xml(xmlf)
    if len(coords) <= 0 :
        return 
    xminv ,yminv,xmaxv,ymaxv ,nm= coords[0]
    for xmin,ymin,xmax,ymax,nm in coords:
        if xminv > xmin:
            xminv = xmin 
        if xmaxv < xmax :
            xmaxv = xmax 
        if yminv > ymin:
            yminv = ymin 
        if ymaxv < ymax:
            ymaxv = ymax 
    ratio = ( xmaxv - xminv ) / w
    print( w,h,xminv ,yminv,xmaxv,ymaxv , ratio ) 
    if ratio >= min_ratio  :
        # no need modify
        shutil.copy( imgf, imgof )
        shutil.copy( xmlf, xmlof )
    else:
        dest_ratio = min_ratio + random.random()*(max_ratio - min_ratio)
        print('->', dest_ratio )
        nw = (int)((xmaxv-xminv)/ dest_ratio)
        nh = (int)( h * nw / w )
        x0,x1 = get_random_pos( w,nw, xminv,xmaxv )
        y0,y1 = get_random_pos( h,nh, yminv,ymaxv )
        while x0 is  None or y0 is  None:
            dest_ratio = min_ratio + random.random()*(max_ratio - min_ratio)
            print('->', dest_ratio )
            nw = (int)((xmaxv-xminv)/ dest_ratio)
            nh = (int)( h * nw / w )
            x0,x1 = get_random_pos( w,nw, xminv,xmaxv )
            y0,y1 = get_random_pos( h,nh, yminv,ymaxv )
        print( x0,x1, y0,y1 )
        fix_xml(xmlf,xmlof,x0,x1,y0,y1)
        img = cv2.imread(imgf) 
        cv2.imwrite( imgof, img[ y0:y1 , x0:x1  ] )

 

def run( input_paths , output_path ):
    cnt = 0
    for ipath in input_paths:
        for pos, ds, fs in os.walk( ipath ):
            for f in fs:
                if f.endswith('jpg'):
                    imgf = os.path.join( pos,f )
                    xmlf = imgf[:-3] + "xml"
                    if os.path.exists(xmlf):
                        imgof = os.path.join(output_path,f)
                        xmlof = imgof[:-3] + "xml"
                        print(cnt)
                        fit_img_lb( imgf,xmlf,imgof,xmlof )
                        cnt+=1

def split_train_verify( src, dest , train_ratio = 0.8 ):
    destt = os.path.join(dest,"train")
    destv = os.path.join(dest,"verify")
    if not os.path.exists( destt ):
        os.mkdir(destt)
    if not os.path.exists( destv ):
        os.mkdir(destv)

    imgs = []
    for pos,ds,fs in os.walk(src):
        for f in fs:
            if f.endswith("jpg"):
                imgs.append(os.path.join( pos,f ))
    random.shuffle(imgs)
    l = len(imgs)
    tn = int(l *train_ratio )
    t = imgs[:tn]
    v = imgs[tn:]
    for f in t:
        shutil.move( f, destt )
        shutil.move( f[:-3]+"xml", destt )
    for f in v:
        shutil.move( f, destv )
        shutil.move(f[:-3]+"xml", destv )
    


if __name__ == "__main__":
    run( [  "../iron_mark_images/labeled/normal" ,  
            "../iron_mark_images/labeled/normal2" ,  
            "../iron_mark_images/labeled/screen" ,  
            "../iron_mark_images/labeled/verify" 
            ] , "../iron_mark_images/labeled/test" )

    split_train_verify( "../iron_mark_images/labeled/test" , "../iron_mark_images/labeled/fordemo" )
