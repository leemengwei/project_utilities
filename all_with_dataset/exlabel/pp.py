'''
post-process 

'''
from copy import copy 

data = '''7 649.0 697.0 330.0 412.0 0.9999609 0.999956
0 503.0 551.0 327.0 407.0 0.9999876 0.9999782
0 367.0 417.0 323.0 401.0 0.9962029 0.99987936
0 187.0 235.0 319.0 393.0 0.9626819 0.9998109
0 305.0 355.0 323.0 397.0 0.59276015 0.9998148
0 736.0 786.0 333.0 421.0 0.9999244 0.5168277
1 578.0 624.0 329.0 411.0 0.9999566 0.9995963
1 436.0 484.0 324.0 404.0 0.99928194 0.99977535
1 246.0 294.0 323.0 399.0 0.86521775 0.99989164
1 730.0 780.0 332.0 428.0 0.9998499 0.6661703
'''

def read_data(txt):
    l = txt.split('\n')
    l = [ x.split(' ') for x in l if len(x) > 10 ]
    rt = []
    for x in l:
        rt.append( [eval(xx) for xx in x])
    return rt 

def calc_center(d):
    for l in d:
        l.append( (l[1] + l[2])/2 )
        l.append( (l[3] + l[4])/2 ) 
    return d 

def fit_line(d):
    '''
    get y = a+bx 
    return a,b
    '''
    d= calc_center(d)
    N = len(d)
    sxy = 0
    sx = 0
    sy = 0
    sx2 = 0
    for l in d:
        sxy += l[-2]*l[-1]
        sx += l[-2]
        sy += l[-1]
        sx2 += l[-2] * l[-2]
    b =  (N * sxy - sx*sy )/( N*sx2 - sx*sx )
    a = (sx2*sy - sx*sxy)/(N*sx2-sx*sx)
    return a,b 
    
def get_project_point(x0,y0,a,b):
    '''
    line : y = a+bx
    return x1,y1  on the line 
    '''
    x1=  (b*(y0-a) + x0)/(b*b + 1)
    y1 = b* x1 + a 
    return x1,y1

def sort_range(d):
    #print(d)
    a,b = fit_line(d)
    #print('sort range:',a,b)
    for l in d:
        l.append( get_project_point(l[-2],l[-1],a,b )) 
    if abs(b) < 1.0: # 横着的字符串
        d = sorted( d ,key = lambda l: l[-1][0] ) # 以 x 排序
        return d 
    return None 

def detect_overlap( rg1, rg2 , tolerance = 3 ):
    '''
    rg1: x1,x2 , y1,y2
    '''
    rt = False 
    if rg1[0]+rg1[1] < rg2[0] + rg2[1]:
        minv = min(rg2[0] , rg2[1] ) + tolerance        
        rt =  rg1[0] > minv or rg1[1] > minv 
    else :
        minv = min(rg1[0] , rg1[1] ) + tolerance
        rt = rg2[0] > minv or rg2[1] > minv 
    return rt 

def confidence( d ):
    return d[5] * d[6]


'''
simple of 'd':
[
['0', array(471, dtype=float32), array(561, dtype=float32), array( 38, dtype=float32), array(190, dtype=float32), array(0.98503, dtype=float32), array(1, dtype=float32)], 
['0', array(108, dtype=float32), array(144, dtype=float32), array(260, dtype=float32), array(302, dtype=float32), array(0.89357, dtype=float32), array(1, dtype=float32)], 
['0', array(835, dtype=float32), array(873, dtype=float32), array(511, dtype=float32), array(559, dtype=float32), array(0.85518, dtype=float32), array(1, dtype=float32)], 
['0', array(284, dtype=float32), array(354, dtype=float32), array(319, dtype=float32), array(419, dtype=float32), array(0.83443, dtype=float32), array(1, dtype=float32)], 
['0', array(870, dtype=float32), array(896, dtype=float32), array( 63, dtype=float32), array(117, dtype=float32), array( 0.8263, dtype=float32), array(1, dtype=float32)]
]
'''
def filter_string_of_text_remove_overlap(d):
    '''
    另一种思路解决误识别，检测所有重叠的区域， 比较其中的 conf，保留conf最大的那个
    '''    
    l = len(d)
    if l <= 1:
        return d ,d 
    # d = sort_range(d) 

    return d ,d 

    d2 = copy(d)

    l = len(d)
    # print(__file__,"filter_string_of_text_remove_overlap", type(d), d)

    mark = [1 for i in range(l)] 

    for i in range(l-1):
        if mark[i]==0:
            continue

        for j in range(i+1, l):
            if mark[j]==0:
                continue

            # print("d[" , i, j, "] =", d[i], d[j])
            if detect_overlap( d[i][1:5] , d[j][1:5] ):
                if confidence(d[i]) > confidence(d[j]):
                    mark[j] = 0
                else:
                    mark[i] = 0

    rt = []
    for i in range(l-1):
        if mark[i]==1:
            rt.append(d[i])
    return d2,rt

if __name__ == '__main__':
    d = read_data(data)
    d= filter_string_of_text_remove_overlap(d)
    for l in d:
        print(l)
     