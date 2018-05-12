
import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import shutil
from PIL import Image


def load_img(path):
    img = skimage.io.imread(path)
    img = img / 255.0
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (112, 112))[:, :, :]   # shape [1, 224, 224, 3]
    return resized_img
def one_hot(labels):
    '''one-hot 编码'''
    n_sample = len(labels)
    n_class = max(labels) + 1
    #print(np.arange(n_sample))
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    #print(onehot_labels.shape,onehot_labels[2090:2100,:])
    return onehot_labels
# def move():
#     dir='./casia-face'
#     filedir=os.listdir(dir)
#     filedir.sort()
#     for x in filedir:
#         for face in os.listdir(dir+'/'+x):
#             shutil.move(os.path.join(dir,x,face),'./test')
#             break

def load_data():

    imgs=[]
    dir = './train'
    filedir=os.listdir(dir)
    filedir.sort()
    for file in filedir:

        #print(file)
        for face in os.listdir(dir+"/"+file):
            resized_img = load_img(os.path.join(dir, file,face))
            #print(resized_img,resized_img.shape)

            imgs.append(resized_img)
    imgs=np.array(imgs)
    #print(imgs.shape)
    # print(imgs[0],imgs[0].shape,type(imgs[0]))
    lable=[]
    for i in range(480):
        if i<=19:
            for j in range(4):
                lable.append(i)
        elif 19<i<=199:
            for k in range(5):
                lable.append(i)
        else:
            for l in range(4):
                lable.append(i)
    #print(lable)
    lables=np.array(lable)
   # print(lables)
    #print(imgs,imgs.shape)
    onehot_labels=one_hot(lables)
    # data = data * 255
    # new_im = Image.fromarray(data.astype(np.uint8))
    # new_im.show()
    # new_im.save('lena_1.jpg')
    return imgs,onehot_labels
def load_testdata():
     imgs = []
     dir = './test'
     filedir = os.listdir(dir)
     filedir.sort()
     for file in filedir:
        resized_img = load_img(os.path.join(dir, file))
        imgs.append(resized_img)
     testdata = np.array(imgs)
     #print(testdata,testdata.shape)
     return testdata
# load_data()
# load_testdata()
# def load_img(path):
#     img = skimage.io.imread(path)
#     img = img / 255.0
#     # print "Original Image Shape: ", img.shape
#     # we crop image from center
#     short_edge = min(img.shape[:2])
#     yy = int((img.shape[0] - short_edge) / 2)
#     xx = int((img.shape[1] - short_edge) / 2)
#     crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
#     # resize to 224, 224
#     resized_img = skimage.transform.resize(crop_img, (64, 64))[:, :, :]   # shape [1, 224, 224, 3]
#     return resized_img
# def one_hot(labels):
#     '''one-hot 编码'''
#     n_sample = len(labels)
#     n_class = max(labels) + 1
#     #print(np.arange(n_sample))
#     onehot_labels = np.zeros((n_sample, n_class))
#     onehot_labels[np.arange(n_sample), labels] = 1
#     #print(onehot_labels,onehot_labels.shape)
#     return onehot_labels
#
# def load_data():
#
#     imgs=[]
#     dir = './face'
#     filedir=os.listdir(dir)
#     filedir.sort()
#     for file in filedir:
#
#         #print(file)
#         for face in os.listdir(dir+"/"+file):
#             #print(face)
#             resized_img = load_img(os.path.join(dir, file,face))
#
#
#             imgs.append(resized_img)
#         #print(resized_img, resized_img.shape)
#     imgs=np.array(imgs)
#     #print(imgs,imgs.shape)
#     data=imgs[0]
#     #print(imgs[0].shape,type(imgs[0]))
#     lable=[]
#     for i in range(2):
#         if i<=0:
#             for j in range(170):
#                 lable.append(i)
#         else:
#             for k in range(170):
#                 lable.append(i)
#     #print(lable)
#     lables=np.array(lable)
#    # print(lables)
#     #print(imgs,imgs.shape)
#     onehot_labels=one_hot(lables)
#     # data = data * 255
#     # new_im = Image.fromarray(data.astype(np.uint8))
#     # new_im.show()
#     # new_im.save('lena_1.jpg')
#     return imgs,onehot_labels
# def load_testdata():
#     imgs = []
#     dir = './testface'
#     filedir = os.listdir(dir)
#     filedir.sort()
#     for file in filedir:
#         resized_img = load_img(os.path.join(dir, file))
#         imgs.append(resized_img)
#     testdata = np.array(imgs)
#     return testdata
#     #print(imgs,imgs.shape)
#
# #load_testdata()



