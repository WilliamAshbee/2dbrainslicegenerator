import torchvision.transforms.functional as F
import models

import random
from pytorch3d.loss import chamfer_distance
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from scipy.special import spherical_jn as besseli
import numpy as np
#from scipy import *
#from math import *
import matplotlib.pyplot as plt
import torch
import pylab as plt
#from skimage import filters
import time

from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

import pylab as plt

sf = .99999
mini_batch = 100
#xbaseline = np.random.randn(2, 1000)

theta = np.linspace(0.0, 2*3.14159, num=1000)
x = np.cos(theta)
y = np.sin(theta)
xbaseline = np.zeros((2,1000))
xbaseline[0,:] = x
xbaseline[1,:] = y

useGPU = True
train_size = mini_batch*20

class MyDataset(Dataset):
    def __init__(self, data, targets, transform = None):
        self.data = data
        self.targets = targets
    def __getitem__(self, index):
        x = self.data[index]
        #root_logger.info(x.shape)
        #if x.shape[0]==1:
        x = x.unsqueeze(0).repeat(3,1,1)
        #else:
        #    x = x.unsqueeze(1).repeat(1,3,1,1)
        if transform:
            x = transform(x)
        #root_logger.info(x.shape)
        y = self.targets[index]
        #root_logger.info('x',x.shape)
        return x, y
    def __len__(self):
        return len(self.data)

path = '/home/users/washbee1/projects/2dbrainslicesgenerator/slices'

import logging
root_logger= logging.getLogger()
root_logger.setLevel(logging.DEBUG) # or whatever
handler = logging.FileHandler('/home/users/washbee1/projects/2dbrainslicesgenerator/logging/logchamfer.log', 'w', 'utf-8') # or whatever
handler.setFormatter(logging.Formatter('%(name)s %(message)s')) # or whatever
root_logger.addHandler(handler)

global transform
transform = transforms.Compose([
     transforms.ConvertImageDtype(torch.float),
     transforms.Resize([256,256])
 ])


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

size = 2500
#inds = [i for i in range(size)]
imgs = []
targets = []
for i,j in enumerate(patients):
    if len(patients[j]) == 2:
        #patients[j] = patients[j].sort()
        patients[j].sort()
        root_logger.info((i,patients[j]))
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
        root_logger.info((img.sum(),target.sum(),img.shape,target.shape))


targets = np.array(targets)
targets = torch.from_numpy(targets)
train_targets = targets[:880,:]
test_targets = targets[880:,:]
imgs = np.array(imgs)
imgs = torch.from_numpy(imgs)
train_imgs = imgs[:880,:,:]
test_imgs = imgs[880:,:,:]
root_logger.info((imgs.shape,targets.shape))
train_data = MyDataset(train_imgs, train_targets,transform)
test_data = MyDataset(test_imgs, test_targets,transform)
batch_size = 32
loader_train = DataLoader(train_data, batch_size=batch_size)
loader_test = DataLoader(test_data, batch_size=batch_size)
dataset = None



def bbox(img):
    a = np.where(img != 0)    
    return np.min(a[0]),np.min(a[1]),np.max(a[0]),np.max(a[1])


global numpoints
numpoints = 1000
side = 32



# import torch
# from vit_pytorch import ViT


# v = ViT(
#     image_size = 32,
#     patch_size = 4,
#     num_classes = 2000,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 1024,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )

# v = torch.nn.Sequential(
#     v,
#     torch.nn.Sigmoid()
# )
#import numpy as np
#mind = np.random.randint(10)
# def chamfer_loss(input, target,model=None,ret_out = False):
#     try:
#         out = models.predict(model,input)
#     except RuntimeError as e:
#         model = None
#         torch.cuda.empty_cache()
#         print(mind,'has error')
#         return None
#     out = out.reshape(target.shape)#64, 1000, 2
#     out = out
#     if not ret_out:
#         return torch.mean((out-target)**2)
#     else:
#         return torch.mean((out-target)**2),out

def distOfPred(a):
    total = 0.0
    count = 0
    for i in range(a.shape[0]):
        for j in range(i,a.shape[0]):
            total += torch.mean(torch.abs(a[i,:,:]-a[j,:,:]))
            count+=1
    return total/float(count)

def chamfer_loss(input, target,model=None,ret_out = False):
    try:
        out = model(input)
        out = out.reshape(target.shape)
        #print('dop',distOfPred(out.detach().clone()))
        
        loss,_ = chamfer_distance(out,target.float())
    except:
        import traceback
        print(traceback.format_exc())
        exit()

    if not ret_out:
        return loss
    else:
        return loss,out

    
#for mind in range(10):
mind = 2
print('------------mind is ',mind)
v = None
try:
    v = models.getModel(mind)
except RuntimeError as e:
    model = None
    torch.cuda.empty_cache()
    print(mind,'has error')
    exit()

if useGPU:
    model = v.cuda()
else:
    model = v

optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001, betas = (.9,.999))#ideal


print('begin')
for epoch in range(10):
    for x,y in loader_train:
        optimizer.zero_grad()
        if useGPU:
            x = x.cuda()
            y = y.cuda()
        loss_train = chamfer_loss(x,y,model=model)
        if loss_train == None:
            break    
        loss_train.backward()
        optimizer.step()
    if loss_train == None:
        break
    print('epoch',epoch,'loss',loss_train)

if loss_train == None:
    exit()
print('begin')
# optimizer = torch.optim.Adam(model.parameters(),lr = 0.00001, betas = (.9,.999))#ideal

# for epoch in range(1):
#     for x,y in loader_train:
#         optimizer.zero_grad()
#         if useGPU:
#             x = x.cuda()
#             y = y.cuda()
#         loss_train = chamfer_loss(x,y,model=model)
#         loss_train.backward()
#         optimizer.step()
#     print('epoch',epoch,'loss',loss_train)

model = model.eval()

for x,y in loader_test:
    if useGPU:
        x = x.cuda()
        y = y.cuda()
    loss_test = chamfer_loss(x,y,model=model)
    print('validation loss',loss_test)
    break

torch.save(model.state_dict(), '/home/users/washbee1/projects/3d-synthd/models/model_{:10.8f}_{:10.8f}_{}.pth'.format(
    loss_train,loss_test,mind))
# DonutDataset.displayCanvas('vit-test-set-2d_{:10.8f}_{:10.8f}_{}_{}_{}_{}.png'.format(
#       loss_train,loss_test,dim,mlp_dim,heads,depth),loader_test, model = model)
# DonutDataset.displayCanvas('vit-test-set-2d_{:10.8f}_{:10.8f}_{}.png'.format(
#     loss_train,loss_test,mind),loader_test, model = model)
###

model = None
torch.cuda.empty_cache()
#time.sleep(10.0)
