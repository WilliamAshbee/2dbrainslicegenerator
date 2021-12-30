
mpath = '/data/hcp-plis/drive01/377451/T1w/377451/surf/rh.white.nii.gz'
spath = '/data/hcp-plis/drive01/377451/T1w/377451/surf/rh.white'

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from freesurfer_surface import Surface
import os
import cv2
from skspatial.objects import Plane

#image = nib.load(mpath)
#image = image.get_fdata()


def calculateCentroid(tri):
    v0 = tri[0]
    v1 = tri[1]
    v2 = tri[2]
    rc = (v0.right+v1.right+v2.right)/3.0
    ac = (v0.anterior+v1.anterior+v2.anterior)/3.0
    sc = (v0.superior+v1.superior+v2.superior)/3.0
    return np.array([rc,ac,sc])

def bbox1(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1]), np.min(a[2]),np.max(a[2])
    return bbox

def bbox1_2(img):
    a = np.where(img != 0)    
    return np.array([np.min(a[0]),np.min(a[1]),np.min(a[2])]),np.array([np.max(a[0]),np.max(a[1]),np.max(a[2])])

# def bbox2(img):
#     rows = np.any(img, axis=1)
#     cols = np.any(img, axis=0)
#     depths = np.any(img, axis=2)
#     rmin, rmax = np.where(rows)[0][[0, -1]]
#     cmin, cmax = np.where(cols)[0][[0, -1]]
#     dmin, dmax = np.where(depths)[0][[0, -1]]
#     return rmin, rmax, cmin, cmax, dmin,dmax
def sliceMri(mpath,whre,mmaxs,mmins):
    image = nib.load(mpath)
    image = image.get_fdata()
    slices = None
    k = (mmaxs[2]+mmins[2])//2
    slices = image[:,:,k]
    slices = cv2.resize(slices, dsize=(256, 256))
    slices = np.flipud(slices)
    return slices

def sliceSurface(sRep,whre,smaxs,smins):
    print('slicesurface',sRep,whre,smaxs,smins)
    nptriangles = np.zeros((len(surface.triangles),3))
    ind = 0
    for triangle_index, triangle in enumerate(surface.triangles):
        nptriangles[ind,:] = calculateCentroid(surface.select_vertices(triangle.vertex_indices))
        ind+=1
    k = (smaxs[2]+smins[2])/2.0
    pl = Plane(point=[0,0,k],normal=[0,0,1])
    print (pl)
    distances = np.zeros(nptriangles.shape[0])
    for i in range(nptriangles.shape[0]):
        distances[i] = pl.distance_point(nptriangles[i,:])

    if np.sum(distances<.500) > 0:
        points = nptriangles[distances<.500,:]
        return points
        #hull = ConvexHull(points)
        #return points[hull.vertices,:]
    else:
        return np.zeros((1,3))
    

    

def slice(mRep,sRep,whre,mmaxs,mmins,smaxs,smins):
    img = sliceMri(mRep,whre,mmaxs,mmins)
    points = sliceSurface(sRep,whre,smaxs,smins)
    points = points[:,[1,0,2]]
    #points[:,1] = points[:,1]+100.0
    points[:,0] = points[:,0]+125.0
    fig = plt.figure()
    plt.imshow(img)
    plt.scatter(points[:,0],points[:,1],c="red")
    #plt.xlim(0, 256)
    #plt.ylim(0, 256)

    plt.savefig('savemri_{}_{}_{}_{}.png'.format(mmaxs,mmins,smaxs,smins).replace(' ','s'),dpi=100)
    plt.clf()

#TODO ! don't need to normalize as much as just map region onto another
#[0 2 1] np.transpose(a,[1,2,0]), b = b[:,[1,2,0]]
#normalize each slice to between 0 and 1 (coordinates and image)
#find width of 1 slice, use that as d, 
#find coordinates of 1 slice min + index*d +-d/2
def findBoundingBoxSurface(surface):
    nptriangles = np.zeros((len(surface.triangles),3))
    ind = 0
    for triangle_index, triangle in enumerate(surface.triangles):
        nptriangles[ind,:] = calculateCentroid(surface.select_vertices(triangle.vertex_indices))
        ind+=1
    smaxs = np.max(nptriangles,axis=0)
    smins = np.min(nptriangles,axis=0)
    srange = smaxs-smins
    sind = np.argsort(srange)

    print('max',np.max(nptriangles,axis = 0))
    print('min',np.min(nptriangles,axis = 0))
    print('srange',srange)

    return smaxs,smins,srange, sind

def findBoundingBoxMri(mpath,perm = 0,k = None):
    image = nib.load(mpath)
    image = image.get_fdata()    
    #rmin, rmax, cmin, cmax, dmin,dmax = bbox1(image)
    mmins,mmaxs = bbox1_2(image)
    mrange = mmaxs-mmins
    mind = np.argsort(mrange)
    print('mrange bbox1 range', mrange)
    return mmaxs,mmins,mrange,mind

count = 0
mricount = 0 
mranges = np.zeros((897,3))
sranges = np.zeros((897,3))
with open('rh.white.txt') as infile:
    id4Prev = None
    mri = None
    surface = None
    while True:
        currentPath = infile.readline().strip()
        path = os.path.normpath(currentPath)
        patharr = path.split(os.sep)

        #print(currentPath,type(currentPath))
        count+=1
        
        try:
            id4 = patharr[4]
            id6 = patharr[6]
            cfname = patharr[-1]
            assert id4 == id6
            
            if id4!= id4Prev:
                assert mri == None
                assert surface == None
                id4Prev = id4
            
            if 'touch' in currentPath:
                continue
            elif 'deformed' in currentPath:
                continue
            elif 'hires' in currentPath:
                continue
            elif 'nii.gz' in currentPath:
                print('lastpath',patharr[-1].strip())
                mri = currentPath
            elif patharr[-1].strip() == 'rh.white':
                print('lastpath',patharr[-1].strip())
                surface = Surface.read_triangular(currentPath)
            else:
                continue
            
            if mri != None and surface !=None:
                assert id4 == id4Prev #assert every file has both an mri and surface
                print('create input output pair')
                smaxs,smins,srange,sind = findBoundingBoxSurface(surface)
                mmaxs,mmins,mrange,mind = findBoundingBoxMri(mri)
                slice(mri,surface,.5,mmaxs,mmins,smaxs,smins)

                print(mrange[0]/srange[0],mrange[1]/srange[1],mrange[2]/srange[2])
                mranges[mricount,:]=mrange
                sranges[mricount,:]=srange 
                mricount+=1
                print('mmaxs', mmaxs)
                print('mmins', mmins)
                print('mranges avg', np.mean(mranges[:mricount,:], axis=0))
                print('mranges std', np.std(mranges[:mricount,:], axis=0))
                print('smaxs', smaxs)
                print('smins', smins)
                print('sranges avg', np.mean(sranges[:mricount,:], axis=0))
                print('sranges std', np.std(sranges[:mricount,:], axis=0))
                print('mind,sind', mind,sind)
                #exit()
                mri = None
                surface = None
            print(id4,id6)
            print()
        except:
            assert currentPath == ''
            break
print(count)
