fn = 'rh.white.txt'
import os
from freesurfer_surface import Surface
from skspatial.objects import Plane
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
#from skimage import measure
import numpy as np
import os
import cv2
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import ConvexHull, convex_hull_plot_2d
    
def calculateCentroid(tri):
    v0 = tri[0]
    v1 = tri[1]
    v2 = tri[2]
    rc = (v0.right+v1.right+v2.right)/3.0
    ac = (v0.anterior+v1.anterior+v2.anterior)/3.0
    sc = (v0.superior+v1.superior+v2.superior)/3.0
    return np.array([rc,ac,sc])

#def convertKtoPlane(i):
    
    
def sliceSurface(surface,norm = [0,0,1]):
    vertex_0 = surface.vertices[0]
    print(len(surface.triangles))
    nptriangles = np.zeros((len(surface.triangles),3))
    ind = 0
    for triangle_index, triangle in enumerate(surface.triangles):
        nptriangles[ind,:] = calculateCentroid(surface.select_vertices(triangle.vertex_indices))
        ind+=1

    nptriangles = nptriangles+(256.0/2.0)
    

    #scaler = MinMaxScaler()
    #scaler.fit(nptriangles)
    #nptriangles = scaler.transform(nptriangles)    
    
    print('max',np.max(nptriangles,axis = 0))
    print('min',np.min(nptriangles,axis = 0))
    print('mean',np.mean(nptriangles,axis = 0))
    print('triangles',len(surface.triangles))
    pl = Plane(np.mean(nptriangles,axis = 0),norm)
    print (pl)
    distances = np.zeros(nptriangles.shape[0])
    for i in range(nptriangles.shape[0]):
        distances[i] = pl.distance_point(nptriangles[i,:])

    print(pl)
    print(np.sum(distances<.100))

    points = nptriangles[distances<.100,:]
    hull = ConvexHull(points)
    return points[hull.vertices,:]
    #return np.random.uniform(0,1,(9000,2))#return single numpy surface 

def sliceMri(mpath,k=80,i=None,j=None):
    image = nib.load(mpath)
    image = image.get_fdata()
    slices = None
    if i == None or j == None or k == None:
        slices = image[k, :,: ]
    else:
        assert i != None and j != None and k != None
        slices = image[:,:,k]
    slices = cv2.resize(slices, dsize=(256, 256))
    slices = np.flipud(slices)
        
    return slices
    #np.save(args.ot_path, slices)
    #return np.random.uniform(0,1,(256,256))#return single numpy mri 
    

def displaySM(s,m,i,pli,axi):
    print('display sm')
    fig = plt.figure()
    #ax=fig.add_axes([0,0,256,256])

    plt.imshow(m)
    plt.scatter(s[:,0],s[:,1])
    plt.xlim(0, 256)
    plt.ylim(0, 256)
    
    plt.savefig('savemri_{}_{}_{}.png'.format(i,pli,axi),dpi=100)
    plt.clf()
count = 0 

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
                normals = np.identity(3)
                for axi in range(3):
                    for pli in range(3):
                        for i in range(64,256,64):
                            surfacer = sliceSurface(surface,normals[pli,:])
                            mrir = sliceMri(mri,k=int(np.mean(surfacer,axis=0)[axi]))
                            displaySM(surfacer,mrir,i,pli,axi)
                mri = None
                surface = None
                exit()
            print(id4,id6)
            
            print(count)
            print()
        except:
            assert currentPath == ''
            break
        