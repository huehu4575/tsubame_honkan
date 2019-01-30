#!python3.6.6
import os
import re
import random
import numpy as np
import chainer

def search_img_name(image_root):
    data_number = 0
    image_name = []
    files = os.listdir(image_root)
    for file in files:
        #index = re.search('.jpg', file)# 拡張子が，jpgのものを検出
        index = re.search('.png', file)# 拡張子が，pngのものを検出
        if index:
            data_number += 1
            image_name.append(file)
    return data_number, image_name

def create_data_set(dir_root, image_root_set, label_set, N):
    img_name_set = []
    label_data_set = []
    for (image_root,label) in zip(image_root_set,label_set):
        (data_number, img_name) = search_img_name(os.path.join(dir_root,image_root))
        if N < data_number:
            img_name_set.extend([os.path.join(image_root,i) for i in random.sample(img_name, k=N)])
            data_number = N
        else :
            img_name_set.extend([os.path.join(image_root,i) for i in img_name])
        label_data_set.extend([label for i in range(data_number)])
    dataset = chainer.datasets.LabeledImageDataset(list(zip(img_name_set,label_data_set)),root=dir_root)
    return dataset
