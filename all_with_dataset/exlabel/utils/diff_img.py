#coding: utf-8
### 程序功能：求图片RGB均值，做差分

import os
import numpy as np
import cv2


'''
计算单个 图片的 RGB 均值
img: 一个张 numpy.ndarray 图片

# 返回: R、G、B 均值
'''
def img_rgb_mean(img):
    B_mean = np.mean(img[:,:,0])
    G_mean = np.mean(img[:,:,1])
    R_mean = np.mean(img[:,:,2])
    return R_mean, G_mean, B_mean

'''
计算 一个目录下所有 图片的 RGB 均值
image_path：图片目录路径
'''
def imgs_rgb_mean(image_path):
    file_names = os.listdir(image_path)
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    for file_name in file_names:
        img = cv2.imread(os.path.join(image_path, file_name), 1)
        per_image_Bmean.append(np.mean(img[:,:,0]))
        per_image_Gmean.append(np.mean(img[:,:,1]))
        per_image_Rmean.append(np.mean(img[:,:,2]))
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)
    return R_mean, G_mean, B_mean

'''
对图片在列方向上，每 @step 个像素采一次样
img: 一个张 numpy.ndarray 图片
img_file_name: 图片的名称
step： 采样步长

返回：采样的 numpy.ndarray 图片
'''
def simple_pixel(img, img_file_name = None, step = 16):
    h = img.shape[0]
    w = img.shape[1]

    y = int(h/4)
    h = h - int(h/4)

    if img_file_name is None:
        img_file_name = ""

    if "GC1" in img_file_name and "_0000." in img_file_name:          ## 最靠左的部分
        x = int(w/2)
        w = w - x
    elif "GC4" in img_file_name and "_3072." in img_file_name:       ## 最靠右的部分
        x = 0
        w = int(w/2)
    else:
        x = int(w/4)
        w = w - x

    ## 在列方向上每 step 个像素采一次样
    simple_img = img[y:h, [x + i for i in range(w) if i % step == 0 ]] 
    return simple_img

'''
图片差分放大
img: 一个张 numpy.ndarray 图片
img_file_name: 图片的名称
diff_multiplier: 差分倍数
step： 做差时的采样步长

返回： 差分放大的 numpy.ndarray 图片
'''
def img_scale_diff(img, img_file_name = None, diff_multiplier = 5 , step = 16):
    ## 在列方向上每16个像素采一次样
    simple_img = simple_pixel(img, img_file_name, step = step)

    R_mean, G_mean, B_mean = img_rgb_mean(simple_img)
    mean = np.array([B_mean, G_mean, R_mean]).reshape(1,1,3) 
    diff_img = (img - mean) * diff_multiplier + 128
    diff_img[diff_img > 255] = 255
    diff_img[diff_img < 0] = 0
    return diff_img.astype(np.uint8)


## ------------------------ demo ------------------------

def diff_demo():
    a = np.random.randn(3,9) * 255
    b = a.reshape(3,3,3)
    img = b.astype(np.uint8)
    print("Image: ", img)

    print("\nR、G、B mean: ", img_rgb_mean(img))

    print("\nDiff Image: ", img_scale_diff(img, diff_multiplier=5, step=1))
    pass

if __name__ == "__main__":
    diff_demo()
    pass





