import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import cv2
from math import sqrt 
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils import data
import random as rnd
import sys

class fer_2013_dataset(data.Dataset):
    def __init__(self, path="./fer2013.csv", mode='train'):
        super(fer_2013_dataset, self).__init__()
        self.samples = pd.read_csv(path)
        self.labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neural']

        if mode == 'train':
        	split = self.samples[(self.samples.Usage == 'Training')]
        elif mode == 'val':
        	split = self.samples[(self.samples.Usage == 'PublicTest')]
        elif mode == 'train':
        	split = self.samples[(self.samples.Usage == 'PrivateTest')]
        else:
        	raise ValueError

        self.x = split.pixels
        self.y = list(split.emotion)

        # augment with random horizontal flip
        self.transform_list = transforms.Compose([
                                                transforms.ToPILImage(),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                            ]) if mode == 'train' else transforms.Compose([
                                                transforms.ToPILImage(),
                                                transforms.ToTensor(),
                                            ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        t = self.transform_list
        return t(np.float32(self.x[index].split()).reshape(1, 48, 48)), self.y[index]

    def cat_distribution(self):
        # show histogram of categories in the whole dataset
        labels_num = self.samples.emotion.value_counts()
        labels_num = [labels_num[i] for i in range(len(labels_num))]
        plt.bar(range(len(labels_num)), labels_num,color='rgbc',tick_label=self.labels)
        for i,b in enumerate(labels_num):
            plt.text(i, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=10)  
        plt.show()

    def show_samples(self, y, num_samples=28):
        # show random num_samples samples from dataset
        # random samples
        sample_idxs = rnd.sample(list(range(len(y))), num_samples)
        fig = plt.figure(figsize = (10,8))
        for i, idx in enumerate(sample_idxs):
            img = Image.fromarray(np.uint8(self.x[idx].split()).reshape(48, 48), mode='L')
            label = self.labels[y[idx]]
            f = fig.add_subplot(4,7,i+1)
            f.imshow(img,cmap='gray')
            plt.title(label)
            f.axes.get_xaxis().set_visible(False)
            f.axes.get_yaxis().set_visible(False)
        plt.show()
