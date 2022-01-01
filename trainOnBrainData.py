#createTorchDataset.py
#https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from vit_pytorch import ViT
import matplotlib.pyplot as plt
transform = transforms.Compose([
     transforms.ConvertImageDtype(torch.float),
     transforms.Resize([32,32])
 ])


class MyDataset(Dataset):
    def __init__(self, data, targets, transform = None):
        self.data = data
        self.targets = targets
    def __getitem__(self, index):
        x = self.data[index]
        #print(x.shape)
        #if x.shape[0]==1:
        x = x.unsqueeze(0).repeat(3,1,1)
        #else:
        #    x = x.unsqueeze(1).repeat(1,3,1,1)
        if transform:
            x = transform(x)
        #print(x.shape)
        y = self.targets[index]
        #print('x',x.shape)
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
imgs = []
targets = []
for i,j in enumerate(patients):
    if len(patients[j]) == 2:
        #patients[j] = patients[j].sort()
        patients[j].sort()
        print(i,patients[j])
        imgfile = patients[j][0]
        targetfile = patients[j][1]
        imgpath = join(path, imgfile)
        targetpath = join(path, targetfile)
        #img = np.zeros((312,312))
        
        #img[:260,:311] = np.load(imgpath)
        img = np.load(imgpath)
        target = np.load(targetpath)
        #shapes.append(target.shape[0])
        if target.shape[0] >= size:
            target = target[random.sample(range(target.shape[0]),size),:]
        else:
            continue
        targets.append(target)
        imgs.append(img)
        print(img.sum(),target.sum(),img.shape,target.shape)
#shapes = np.array(shapes)
#print(np.min(shapes))
targets = np.array(targets)
targets = torch.from_numpy(targets)
train_targets = targets[:880,:]
test_targets = targets[880:,:]
imgs = np.array(imgs)
imgs = torch.from_numpy(imgs)
train_imgs = imgs[:880,:,:]
test_imgs = imgs[880:,:,:]
print(imgs.shape,targets.shape)
train_data = MyDataset(train_imgs, train_targets,transform)
test_data = MyDataset(test_imgs, test_targets,transform)

batch_size =  32 if transform else 1
img_size = 32 if transform else 1

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
dataset = None

img_size = 32 if transform else 312
class ViTNet(torch.nn.Module):
    def __init__(self,img_size):
       super().__init__()
       model = ViT(
            image_size = img_size,
            patch_size = 4,
            num_classes = 5000,
            dim = 1024,
            depth = 3,
            heads = 8,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
       self.model = torch.nn.Sequential(
            model,
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)*312.0

    
v = ViTNet(img_size).cuda()



img = torch.randn(2, 3, img_size, img_size).cuda()

preds = v(img) # (1, 1000)
print(preds)
losses = []
optimizer = torch.optim.Adam(v.parameters(), lr=0.001)

for epoch in range(1000):
    for x,y in train_dataloader:
        x = x.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        #print('data loader',x.shape,y.shape)
        out = v(x)
        out = out.reshape(y.shape)
        loss = torch.mean((out-y)**2)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print('epoch',epoch,'avg loss' ,np.array(losses).mean())
    losses = []
    

for x,y in train_dataloader:
    fig = plt.figure()
    plt.imshow(x[0,0,:,:],extent=[0,311,259,0])
    plt.scatter(y[0,:,0],y[0,:,1],c="black", s= .01)
    x = x.cuda()
    y = y.cuda()
    out = v(x).reshape(y.shape).cpu().detach().numpy()
    
    plt.scatter(out[0,:,0],out[0,:,1],c="red", s= .01)
    
    #plt.xlim(0, 256)
    #plt.ylim(0, 256)

    plt.savefig('trainsethelloworld1000.png',#""".format(mmaxs,mmins,smaxs,smins).replace(' ','s')"""
                dpi=100)
    plt.clf()
    break

for x,y in test_dataloader:
    fig = plt.figure()
    plt.imshow(x[0,0,:,:],extent=[0,311,259,0])
    plt.scatter(y[0,:,0],y[0,:,1],c="black", s= .01)
    x = x.cuda()
    y = y.cuda()
    out = v(x).reshape(y.shape).cpu().detach().numpy()
    loss = np.mean((out-y.cpu().detach().numpy())**2)
    print('test avg loss' ,loss.item())
    
    
    plt.scatter(out[0,:,0],out[0,:,1],c="red", s= .01)
    
    #plt.xlim(0, 256)
    #plt.ylim(0, 256)

    plt.savefig('testsethelloworld1000.png',#""".format(mmaxs,mmins,smaxs,smins).replace(' ','s')"""
                dpi=100)
    plt.clf()
    break

optimizer = torch.optim.Adam(v.parameters(), lr=0.0001)

for epoch in range(1000):
    for x,y in train_dataloader:
        x = x.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        #print('data loader',x.shape,y.shape)
        out = v(x)
        out = out.reshape(y.shape)
        loss = torch.mean((out-y)**2)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print('epoch',epoch,'avg loss' ,np.array(losses).mean())
    losses = []
    

for x,y in train_dataloader:
    fig = plt.figure()
    plt.imshow(x[0,0,:,:],extent=[0,311,259,0])
    plt.scatter(y[0,:,0],y[0,:,1],c="black", s= .01)
    x = x.cuda()
    y = y.cuda()
    out = v(x).reshape(y.shape).cpu().detach().numpy()
    
    plt.scatter(out[0,:,0],out[0,:,1],c="red", s= .01)
    
    #plt.xlim(0, 256)
    #plt.ylim(0, 256)

    plt.savefig('trainsethelloworld2000.png',#""".format(mmaxs,mmins,smaxs,smins).replace(' ','s')"""
                dpi=100)
    plt.clf()
    break

for x,y in test_dataloader:
    fig = plt.figure()
    plt.imshow(x[0,0,:,:],extent=[0,311,259,0])
    plt.scatter(y[0,:,0],y[0,:,1],c="black", s= .01)
    x = x.cuda()
    y = y.cuda()
    out = v(x).reshape(y.shape).cpu().detach().numpy()
    loss = np.mean((out-y.cpu().detach().numpy())**2)
    print('test avg loss' ,loss.item())
    
    plt.scatter(out[0,:,0],out[0,:,1],c="red", s= .01)
    
    #plt.xlim(0, 256)
    #plt.ylim(0, 256)

    plt.savefig('testsethelloworld2000.png',#""".format(mmaxs,mmins,smaxs,smins).replace(' ','s')"""
                dpi=100)
    plt.clf()
    break