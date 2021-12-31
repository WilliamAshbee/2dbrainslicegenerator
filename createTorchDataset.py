#createTorchDataset.py
#https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = torch.LongTensor(targets)
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    def __len__(self):
        return len(self.data)

path = '/home/users/washbee1/projects/2dbrainslicesgenerator/slices'

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
patients = dict()
for file in onlyfiles:
    s= file.split('.')
    if len(s)==3:
        if s[0] not in patients:
            patients[s[0]] = []
        patients[s[0]].append(file)
    #print(patients[s[0]])

shapes = []
size = 2500
#inds = [i for i in range(size)]

for i,j in enumerate(patients):
    if len(patients[j]) == 2:
        #patients[j] = patients[j].sort()
        patients[j].sort()
        print(i,patients[j])
        imgfile = patients[j][0]
        targetfile = patients[j][1]
        imgpath = join(path, imgfile)
        targetpath = join(path, targetfile)
        img = np.zeros((312,312))
        
        img[:260,:311] = np.load(imgpath)
        
        target = np.load(targetpath)
        #shapes.append(target.shape[0])
        if target.shape[0] >= size:
            target = target[random.sample(range(target.shape[0]),size),:]
        else:
            continue
        print(img.sum(),target.sum(),img.shape,target.shape)
#shapes = np.array(shapes)
#print(np.min(shapes))