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
        x = x.unsqueeze(0).repeat(3,1,1)
        #else:
        #    x = x.unsqueeze(1).repeat(1,3,1,1)
        if transform:
            x = transform(x)
        y = self.targets[index]
        min0,min1,max0,max1 = bbox(x)
        x = x[min0:(max0+1),min1:(max1+1)]
        y[:,0] = (y[:,0]-torch.min(y[:,0]))/(torch.max(y[:,0])-torch.min(y[:,0]))
        y[:,1] = (y[:,1]-torch.min(y[:,1]))/(torch.max(y[:,1])-torch.min(y[:,1]))
        
        x = F.resize(x, (side,side))
        #root_logger.info('x',x.shape)
        return x, y
    def __len__(self):
        return len(self.data)

path = '/home/users/washbee1/projects/2dbrainslicesgenerator/slices'

import logging
root_logger= logging.getLogger()
root_logger.setLevel(logging.INFO) # or whatever
handler = logging.FileHandler('/home/users/washbee1/projects/2dbrainslicesgenerator/logging/chamf2dbd.log', 'w', 'utf-8') # or whatever
handler.setFormatter(logging.Formatter('%(name)s %(message)s')) # or whatever
root_logger.addHandler(handler)

global transform
transform = transforms.Compose([
     transforms.ConvertImageDtype(torch.float)#,
     #transforms.Resize([256,256])
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
side = 512



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

def chamfer_loss(input, target,mod=None,ret_out = False):
    try:
        out = mod(input)
        out = out.reshape(target.shape)
        
        loss,_ = chamfer_distance(out,target.float())
    except:
        import traceback
        root_logger.info(traceback.format_exc())
        exit()

    if not ret_out:
        return loss
    else:
        return loss,out

    
#for mind in range(10):
mind = 2
root_logger.info(('------------mind is ',mind))
try:
    model = models.getModel(mind)
except RuntimeError as e:
    torch.cuda.empty_cache()
    root_logger.info((mind,'has error'))
    exit()

if useGPU:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001, betas = (.9,.999))#ideal


epochs = 500
for epoch in range(epochs):
    for x,y in loader_train:
        optimizer.zero_grad()
        if useGPU:
            x = x.cuda()
            y = y.cuda()
        loss_train = chamfer_loss(x,y,mod=model)
        if loss_train == None:
            break    
        loss_train.backward()
        optimizer.step()
    if loss_train == None:
        break
    root_logger.info(('epoch',epoch,'loss',loss_train))
img_size,patch_size,train_loss = side, side//16, loss_train.item()
for x,y in loader_train:
    fig = plt.figure()
    plt.imshow(x[0,0,:,:],extent=[0,1,1,0])
    plt.scatter(y[0,:,0],y[0,:,1],c="black", s= .01)
    x = x.cuda()
    y = y.cuda()
    out = model(x).reshape(y.shape).cpu().detach().numpy()
    
    plt.scatter(out[0,:,0],out[0,:,1],c="red", s= .01)
    
    #plt.xlim(0, 256)
    #plt.ylim(0, 256)

    plt.savefig('trainsetchamfer{}_{}_{}_{:.1f}.png'.format(epochs,img_size,patch_size,train_loss),#""".format(mmaxs,mmins,smaxs,smins).replace(' ','s')"""
                dpi=600)
    plt.clf()
    break

for x,y in loader_test:
    fig = plt.figure()
    plt.imshow(x[0,0,:,:],extent=[0,1,1,0])
    plt.scatter(y[0,:,0],y[0,:,1],c="black", s= .01)
    x = x.cuda()
    y = y.cuda()
    
    #out = model(x)
    #out = out.reshape(y.shape)
    #loss = np.mean((out-y.cpu().detach().numpy())**2)
    loss,out = chamfer_loss(x,y,mod=model,ret_out = True)
    
    root_logger.info(('test avg loss' ,loss.item()))
    
    plt.scatter(out[0,:,0].cpu().detach().numpy(),out[0,:,1].cpu().detach().numpy(),c="red", s= .01)
    
    #plt.xlim(0, 256)
    #plt.ylim(0, 256)

    plt.savefig('testsetchamfer{}_{}_{}_{:.1f}.png'.format(epochs,img_size,patch_size,loss.item()),#""".format(mmaxs,mmins,smaxs,smins).replace(' ','s')"""
                dpi=600)
    plt.clf()
    break


if loss_train == None:
    exit()

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

model = model.eval()

for x,y in loader_test:
    if useGPU:
        x = x.cuda()
        y = y.cuda()
    loss_test = chamfer_loss(x,y,mod=model)
    root_logger.info(('validation loss',loss_test))
    break

torch.save(model.state_dict(), '/home/users/washbee1/projects/3d-synthd/models/model_{:10.8f}_{:10.8f}_{}_{}.pth'.format(
    loss_train,loss_test,mind,epochs))
# DonutDataset.displayCanvas('vit-test-set-2d_{:10.8f}_{:10.8f}_{}_{}_{}_{}.png'.format(
#       loss_train,loss_test,dim,mlp_dim,heads,depth),loader_test, model = model)
# DonutDataset.displayCanvas('vit-test-set-2d_{:10.8f}_{:10.8f}_{}.png'.format(
#     loss_train,loss_test,mind),loader_test, model = model)
###

torch.cuda.empty_cache()
#time.sleep(10.0)

            