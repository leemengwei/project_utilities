from PyQt5.QtGui import QImage, qRgb
import numpy as np
import cv2 

class NotImplementedException:
    pass

gray_color_table = [qRgb(i, i, i) for i in range(256)]

def NumpyToQImage(im, copy=False):
    if im is None:
        return QImage()

        
    im = np.require(im, np.uint8, 'C') 

    if im.dtype == np.uint8:
        if len(im.shape) == 2:
            qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
            qim.setColorTable(gray_color_table)
            return qim.copy() if copy else qim

        elif len(im.shape) == 3:
            if im.shape[2] == 3:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888);
                return qim.copy() if copy else qim
            elif im.shape[2] == 4:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32);
                return qim.copy() if copy else qim

    raise NotImplementedException

def ensure_value_in_range( v, minv,maxv ):
    if v < minv :
        return minv 
    if v > maxv:
        return maxv 
    return v 

def get_diff_for_1chn(npimg,meanv, enhance_ratio ): 
    rt = (npimg.astype( np.float ) - meanv) * enhance_ratio + 128
    rt[rt>255] = 255
    rt[rt<0] = 0     
    return  rt.astype(np.uint8)  
 
    

def bkg_diff_enhance(fn , npimg, enhance_ratio = 2.0  ):
    h,w = npimg.shape[:2]
    chn = 3
    if len(npimg.shape ) == 2:
        chn = 1
    
    # decide left or right or center 
    id = None 
    pos =  fn.find( 'GC' )
    if pos >= 0:
        id = fn[ pos+2 ] 
    else:
        pos = fn.find('GM')
        if pos >= 0:
            id = fn[ pos+2 ]  
        else:
            id = '2'
    rg=None # ( xmin xmax ymin ymax )
    
    step = int(h * 0.05)
    if id == '1': #left 
        rg = [ w * 0.4, w * 0.9  , h * 0.1 , h* 0.5 ] 
    elif id == '2': # center
        rg = [ w * 0.1, w * 0.9  , h * 0.1 , h* 0.5 ] 
    else: # right
        rg = [ w * 0.1, w * 0.4  , h * 0.1 , h* 0.5 ]
    rg = [int(x) for x in rg]

    sampleimg = npimg[  rg[ 2 ] : rg[3]   ,   rg[0] : rg[1]   ]
    meanimg = cv2.resize( sampleimg , (1,1) ).reshape(chn)
 
    if chn == 1:
        return get_diff_for_1chn( npimg, meanimg[0] , enhance_ratio ) 
    else:
        imgs  = cv2.split(npimg) 
        for i,img in enumerate(imgs):
            imgs[i] =  get_diff_for_1chn( img, meanimg[i] , enhance_ratio ) 
        return cv2.merge(imgs)
          

if __name__ == "__main__":
    import cv2  
    img = cv2.imread( "../data/GC2_20191012_104008_069.png" )
    qimg = NumpyToQImage(img)
    print('ok')

