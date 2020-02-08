import cv2
import numpy as np,sys
import matplotlib.pyplot as plt
from IPython import embed
import sys,os

raw_A = cv2.imread('car.png')
B = cv2.imread('head1.png')
border=int(B.shape[0]/6)
upper_left_y = np.random.randint(border, int(raw_A.shape[0]-border-B.shape[0]))
upper_left_x = np.random.randint(border, int(raw_A.shape[1]-border-B.shape[1]))
A = raw_A[int(upper_left_y-border):int(B.shape[0]+upper_left_y+border), int(upper_left_x-border):int(B.shape[1]+upper_left_x+border)]

layers = 2#int(sys.argv[1])
#normer = 2**layers
#A_fixer = (np.array(A.shape)%normer)[:2]
#B_fixer = (np.array(B.shape)%normer)[:2]
#try:
#    A = cv2.resize(A, tuple(A.shape[:2][::-1]-A_fixer[::-1]))
#    B = cv2.resize(B, tuple(B.shape[:2][::-1]-B_fixer[::-1]))
#except:
#    print("Layers to much or fig too small to make prymaid!")

#generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(layers):
    G = cv2.pyrDown(G)
    gpA.append(G)
    #print("A", G.shape)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(layers):
    G = cv2.pyrDown(G)
    gpB.append(G)
    #print("B", G.shape)

# generate Laplacian Pyramid for A
lpA = [gpA[layers-1]]
for i in range(layers-1,0,-1):
    GE = cv2.pyrUp(gpA[i])
    #print(GE.shape)
    try:
        L = cv2.subtract(gpA[i-1],GE[:gpA[i-1].shape[0],:gpA[i-1].shape[1]])
    except:
        L = cv2.subtract(gpA[i-1][:GE.shape[0],:GE.shape[1]],GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[layers-1]]
for i in range(layers-1,0,-1):
    GE = cv2.pyrUp(gpB[i])
    try:
        L = cv2.subtract(gpB[i-1],GE[:gpB[i-1].shape[0],:gpB[i-1].shape[1]])
    except:
        L = cv2.subtract(gpB[i-1][:GE.shape[0],:GE.shape[1]],GE)
    lpB.append(L)


#####################intergration############################

# Now insert B into A in each level
LS = []
for la,lb in zip(lpA,lpB):
    #embed()
    #la[int(0.5*lb.shape[0]*ratio_to_B):int(1.5*lb.shape[0]*ratio_to_B), int(0.5*lb.shape[1]*ratio_to_B):int(1.5*lb.shape[1]*ratio_to_B)] = lb
    starter_x, starter_y = np.round((np.array(la.shape)-np.array(lb.shape))[:2]/2).astype(int)
    la[starter_x:starter_x+lb.shape[0], starter_y:starter_y+lb.shape[1]] = lb
    #ls = np.hstack((la[:,0:int(cols/2)], lb[:,int(cols/2):]))
    LS.append(la)

# now reconstruct
ls_ = LS[0]
ls_ = (LS[0]/(1+(layers*0.025))).astype(np.uint8)
for i in range(1,layers):
    ls_ = cv2.pyrUp(ls_)
    print(ls_.shape, LS[i].shape)
    try:
        ls_ = cv2.add(ls_, LS[i][:ls_.shape[0],:ls_.shape[1]])
    except:
        ls_ = cv2.add(ls_[:LS[i].shape[0],:LS[i].shape[1]], LS[i])


# image with direct connecting each half
#embed()
raw_A[int(upper_left_y-border):int(B.shape[0]+upper_left_y+border), int(upper_left_x-border):int(B.shape[1]+upper_left_x+border)] = ls_


plt.imshow(raw_A[:,:,[2,1,0]])
plt.show()

