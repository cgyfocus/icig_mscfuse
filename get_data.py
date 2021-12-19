# -*- coding:utf-8-*-
import os

# os.listdir 返回指定目录下的所有文件名和目录名
# os.mkdir(path) 创建目录
from os import listdir, mkdir, sep

# os.path.joim(path, name) 连接目录
# os.split
from os.path import join, exists, splitext
import random
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from scipy.misc import imread, imsave, imresize
import matplotlib as mpl
import cv2
from torchvision import datasets, transforms


def list_images(directory):
    """
    该函数的作用是读取名为 directory 里面的图片名称，返回一个列表 [1.png, 2.png, ..., n.png]
    split() 通过指定分割符对字符串进行切分
    该函数作用是对图片进行读取
    返回每个图片的路径       ['C:\\Users\\cgy19\\Desktop\\densefuse-attention\\images\\IV_images\\IR1.png',...]
    """

    images = []
    dir = listdir(directory)
    #dir.sort()                                      # 进行排序
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
    return images



# 将数据切分到 batch
def load_dataset(image_path):
    num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    #random.shuffle(original_imgs_path)
    original_imgs_path.sort()
    return original_imgs_path                                                # 80000 % 4 = 20000


def information(image_path, BATCH_SIZE):

    num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return batches

# 输入数据
def get_train_images(paths, batchsize):

    images = []
    for i in range(batchsize):
        image = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
        image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        images.append(image)

    # np.stack 该函数主要是用来提升维度
    images = np.array(images)                                   # 将列表数据转为数组
    images = torch.from_numpy(images).float()

    return images