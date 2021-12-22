
mpath = '/data/hcp-plis/drive01/377451/T1w/377451/surf/rh.white.nii.gz'
spath = '/data/hcp-plis/drive01/377451/T1w/377451/surf/rh.white'

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from freesurfer_surface import Surface
import os
image = nib.load(mpath)
image = image.get_fdata()


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

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    depths = np.any(img, axis=2)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    dmin, dmax = np.where(depths)[0][[0, -1]]
    return rmin, rmax, cmin, cmax, dmin,dmax
print(bbox1(image))
print(bbox2(image))

surface = Surface.read_triangular(spath)

nptriangles = np.zeros((len(surface.triangles),3))
ind = 0
for triangle_index, triangle in enumerate(surface.triangles):
    nptriangles[ind,:] = calculateCentroid(surface.select_vertices(triangle.vertex_indices))
    ind +=1
nptriangles = nptriangles+(256.0/2.0)

rmin, rmax, cmin, cmax, dmin,dmax = bbox1(image)


print('max',np.max(nptriangles,axis = 0))
print('min',np.min(nptriangles,axis = 0))
srange = np.max(nptriangles,axis=0)-np.min(nptriangles,axis = 0)
sind = np.argsort(srange)
srange = np.sort(srange)
mrange = np.array([rmax-rmin,cmax-cmin,dmax-dmin])
mind = np.argsort(mrange)
mrange = np.sort(mrange)
print('argsorts m,s',mind,sind)
print('mrange bbox1 range', mrange)
print(mrange[0]/srange[0],mrange[1]/srange[1],mrange[2]/srange[2])
print('triangles',len(surface.triangles))


def findBoundingBoxSurface(surface):
    nptriangles = np.zeros((len(surface.triangles),3))
    ind = 0
    for triangle_index, triangle in enumerate(surface.triangles):
        nptriangles[ind,:] = calculateCentroid(surface.select_vertices(triangle.vertex_indices))
        ind+=1

    srange = np.max(nptriangles,axis=0)-np.min(nptriangles,axis = 0)
    sind = np.argsort(srange)
    srange = np.sort(srange)

    
    print('max',np.max(nptriangles,axis = 0))
    print('min',np.min(nptriangles,axis = 0))
    print('srange',srange)

    return srange, sind

def findBoundingBoxMri(mpath,perm = 0,k = None):
    image = nib.load(mpath)
    image = image.get_fdata()
    
    rmin, rmax, cmin, cmax, dmin,dmax = bbox1(image)
    mrange = np.array([rmax-rmin,cmax-cmin,dmax-dmin])
    mind = np.argsort(mrange)
    mrange = np.sort(mrange)

    print('mrange bbox1 range', mrange)
        
    return mrange,mind
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
                srange, sind = findBoundingBoxSurface(surface)
                mrange, mind = findBoundingBoxMri(mri)
                print(mrange[0]/srange[0],mrange[1]/srange[1],mrange[2]/srange[2])
                mranges[mricount,:]=mrange
                sranges[mricount,:]=srange 
                mricount+=1
                print('mranges avg', np.mean(mranges[:mricount,:], axis=0))
                print('mranges std', np.std(mranges[:mricount,:], axis=0))
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