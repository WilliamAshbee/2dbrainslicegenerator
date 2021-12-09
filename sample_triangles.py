from freesurfer_surface import Surface

surface = Surface.read_triangular('/data/hcp-plis/hdd05/101410/T1w/101410/surf/lh.pial')

for vertex in surface.vertices[:3]:
    print(vertex)

vertex_0 = surface.vertices[0]
print('coordinates of vertex #0:', (vertex_0.right, vertex_0.anterior, vertex_0.superior))

print(type(surface))
print(type(surface.triangles))
print(len(surface.triangles))
a = set()
import numpy as np

def calculateCentroid(tri):
    v0 = tri[0]
    v1 = tri[1]
    v2 = tri[2]
    rc = (v0.right+v1.right+v2.right)/3.0
    ac = (v0.anterior+v1.anterior+v2.anterior)/3.0
    sc = (v0.superior+v1.superior+v2.superior)/3.0
    return np.array([rc,ac,sc])

def distPointToPlane(points,n,d):
    dists = np.dot(n,points.T)+d
    return dists

def pointsToPlane(points,plane,d):
    norm = np.linalg.norm(plane)
    n = plane/norm
    dists = distPointToPlane(points,n,d)
    return points - dists.reshape(-1,1)*n.reshape(1,-1)

pointsTest = np.array([[10,20,-5],[10,20,-5]])
plane_test = np.array([0,1,0])
d_test = -10
ptpRes = pointsToPlane(pointsTest,plane_test,d_test)
test_result = np.array([[10., 10., -5.],
       [10., 10., -5.]])
assert np.sum(ptpRes != test_result)==0

#print('surface.triangles',type(surface.triangles))
nptriangles = np.zeros((len(surface.triangles),3))
ind = 0
for triangle_index, triangle in enumerate(surface.triangles):
    nptriangles[ind,:] = calculateCentroid(surface.select_vertices(triangle.vertex_indices))
    ind+=1



print(len(a)/3.0)
# for triangle_index, triangle in enumerate(surface.triangles):
#     print(f'\ntriangle #{triangle_index}:')
#     print('vertex indices:', triangle.vertex_indices)
#     print('vertex coordinates:')
#     for vertex in surface.select_vertices(triangle.vertex_indices):
#         print((vertex.right, vertex.anterior, vertex.superior))
import time
count = 0
from datetime import datetime
start = datetime.now()

for triangle_index, triangle in enumerate(surface.triangles):
    for vertex in surface.select_vertices(triangle.vertex_indices):
        count+=1
        if count%10000 == 0:
            print((vertex.right, vertex.anterior, vertex.superior))

end = datetime.now()

print('start',start.strftime("%H:%M:%S"))
print('end',end.strftime("%H:%M:%S"))


print(distPointToPlane(nptriangles,[0,1,0],-10))
print('max',np.max(nptriangles,axis = 0))
print('min',np.min(nptriangles,axis = 0))
print('mean',np.mean(nptriangles,axis = 0))
print('triangles',len(surface.triangles))

from skspatial.objects import Plane
pl = Plane.from_points(np.max(nptriangles,axis = 0), np.min(nptriangles,axis = 0), np.mean(nptriangles,axis = 0))

distances = np.zeros(nptriangles.shape[0])
for i in range(nptriangles.shape[0]):
    distances[i] = pl.distance_point(nptriangles[i,:])

print(pl)
print(np.sum(distances<.100))

points = nptriangles[distances<.100,:]

import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2])
plt.savefig('fig.png',dpi=600)