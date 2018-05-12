import os
import skimage.io

import cv2
path="./lfw"
c=[]
i=0
file=os.listdir(path)
file.sort()
print(file)
for f in file:
    for j in os.listdir(path+"/"+f):
        # print(j)
        a=path+"/"+str(f)+"/"+str(j)+" "+str(i)
        c.append(a)
    i = i + 1

f=open("./image",'a')
for i in range(len(c)):
    f.write("\n"+c[i])