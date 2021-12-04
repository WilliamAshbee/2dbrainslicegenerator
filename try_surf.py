from freesurfer_surface import Surface

surface = Surface.read_triangular('/data/hcp-plis/hdd05/101410/T1w/101410/surf/lh.pial')

for vertex in surface.vertices[:3]:
    print(vertex)

vertex_0 = surface.vertices[0]
print('coordinates of vertex #0:', (vertex_0.right, vertex_0.anterior, vertex_0.superior))

for triangle_index, triangle in enumerate(surface.triangles):
    print(f'\ntriangle #{triangle_index}:')
    print('vertex indices:', triangle.vertex_indices)
    print('vertex coordinates:')
    for vertex in surface.select_vertices(triangle.vertex_indices):
        print((vertex.right, vertex.anterior, vertex.superior))