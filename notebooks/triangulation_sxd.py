import os
import pickle

import numpy as np
import scipy
import einops
import skimage

import open3d as o3d


def _pad_arr(arr, pad_size=10):
    return np.pad(
        arr, 
        ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),   # pad size to each dimension, require tensor to have size (H,W, C)
        mode='constant', 
        constant_values=0)

def _to_o3d_mesh(verts: np.ndarray, faces: np.ndarray, color=None):
    verts = o3d.utility.Vector3dVector(verts)
    faces = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(verts, faces)
        
    if color is not None:
        if len(color) != len(verts): 
            color = np.array(color)[None].repeat(len(verts), axis=0)
        mesh.vertex_colors = o3d.utility.Vector3dVector(color)   
        
    return mesh


def _to_o3d_pc(xyz: np.ndarray, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    print('[_to_o3d_pc] color: ', pcd.points)
        
    if color is not None:
        if len(color) != len(xyz): 
            color = np.array(color)[None].repeat(len(xyz), axis=0)
        pcd.colors = o3d.utility.Vector3dVector(color)

    return pcd


def _make_grid(bb_min=[0,0,0], bb_max=[1,1,1], shape=[10,10,10], 
            mode='on', flatten=True, indexing="ij"):
    """ Make a grid of coordinates

    Args:
        bb_min (list or np.array): least coordinate for each dimension
        bb_max (list or np.array): maximum coordinate for each dimension
        shape (list or int): list of coordinate number along each dimension. If it is an int, the number
                            same for all dimensions
        mode (str, optional): 'on' to have vertices lying on the boundary and 
                              'in' for keeping vertices and its cell inside of the boundary
                              same as align_corners=True and align_corners=False
        flatten (bool, optional): Return as list of points or as a grid. Defaults to True.
        indexing (["ij" or "xy"]): default to "xy", see https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html

    Returns:
        np.array: return coordinates (XxYxZxD if flatten==False, X*Y*ZxD if flatten==True.
    """    
    coords=[]
    bb_min = np.array(bb_min)
    bb_max = np.array(bb_max)
    if type(shape) is int:
        shape = np.array([shape]*bb_min.shape[0])
    for i,si in enumerate(shape):
        if mode=='on':
            coord = np.linspace(bb_min[i], bb_max[i], si)
        elif mode=='in':
            offset = (bb_max[i]-bb_min[i])/2./si
            # 2*si*w=bmax-bmin
            # w = (bmax-bmin)/2/si
            # start, end = bmax+w, bmin-w
            coord = np.linspace(bb_min[i]+offset,bb_max[i]-offset, si)
        coords.append( coord )
    grid = np.stack(np.meshgrid(*coords,sparse=False, indexing=indexing), axis=-1)
    if flatten==True:
        grid = grid.reshape(-1,grid.shape[-1])
    return grid


def _prune_unused_vertices(vert, face):
    # Identify all unique vertex indices used in face
    unique_vert_ind = np.unique(face)
    mapper = np.zeros(vert.shape[0], dtype=int)
    mapper[unique_vert_ind] = np.arange(unique_vert_ind.shape[0])
    new_face = mapper[face]
    # Create the new vertices array using only the used vertices
    new_vert = vert[unique_vert_ind]
    
    return new_vert, new_face, unique_vert_ind


def _downsample_sparse_pooling(omg, factor=4, anti_aliasing=False, visualize=False):
    """ downsample input array with sparse pooling
    Args:
        omg: np.ndarray, (H, W, 4), omage tensor
        factor: int, downsample factor
        anti_aliasing: bool, whether to use anti_aliasing
        visualize: bool, whether to visualize the result
    Returns:
        dict, containing 'omg_down_star', 'omg_down', 'sov', 'edge_occ_down'
        'omg_down_star': np.ndarray, downsampled omg with edge snapping
        'omg_down': np.ndarray, downsampled omg without edge snapping
        'sov': np.ndarray, occupancy map with snapped boundaries highlighted
        'edge_occ_down': np.ndarray, edge occupancy map
    """
    # assuming omg is 1024 x 1024
    occ = omg[..., 3]>=.5
    eroded_mask = scipy.ndimage.binary_erosion(occ, structure=np.ones((3,3))) # sqaure strucure is needed to get the corners
    edge_occ = ~eroded_mask & occ
    edge_omg = omg.copy()
    edge_omg[edge_occ==0] = -1.
    
    edge_occ_patches = einops.rearrange(edge_occ, '(h1 h2) (w1 w2) -> h1 w1 h2 w2', h2=factor, w2=factor)
    edge_occ_down = edge_occ_patches.max(axis=-1).max(axis=-1)
    eod_0_count  = (edge_occ_patches==0).sum(axis=-1).sum(axis=-1)
    eod_1_count  = (edge_occ_patches==1).sum(axis=-1).sum(axis=-1)
    edge_omg_patches = einops.rearrange(edge_omg, '(h1 h2) (w1 w2) c-> h1 w1 h2 w2 c', h2=factor, w2=factor)
    edge_omg_down = edge_omg_patches.sum(axis=-2).sum(axis=-2) + eod_0_count[...,None]
    edge_omg_down = np.divide(edge_omg_down, eod_1_count[...,None], out=np.zeros_like(edge_omg_down), where=eod_1_count[...,None]!=0)

    omg_down = skimage.transform.resize(omg, (omg.shape[0]//factor,)*2, order=0, preserve_range=False, anti_aliasing=anti_aliasing)
    
    omg_down_star = edge_omg_down * (edge_occ_down[...,None]) + omg_down * (1-edge_occ_down[...,None])
    star_occ = (omg_down[...,3] >= .5) | edge_occ_down
    
    sov = (omg_down[...,3] >= .5) # for visualizaton
    sov = sov * .5 + edge_occ_down.astype(float)
    sov[sov>=1.] = 1.

    return dict(omg_down_star=omg_down_star, omg_down=omg_down,
            sov=sov, edge_occ_down=edge_occ_down, edge_occ=edge_occ)


def meshing_uv_map(occupancy):
    occ = occupancy.astype(bool)
    pixel_index = np.arange(occ.size).reshape(occ.shape)

    # Determine triangles' vertices
    is_tri_vert = occ & np.roll(occ, shift=-1, axis=0) & np.roll(occ, shift=-1, axis=1)
    verta = pixel_index
    vertb = np.roll(pixel_index, shift=-1, axis=1)
    vertc = np.roll(pixel_index, shift=-1, axis=0)
    face0 = np.stack([verta[is_tri_vert], vertb[is_tri_vert], vertc[is_tri_vert]], axis=1)
    
    # Determine the second set of triangles' vertices
    is_tri_vert = occ & np.roll(occ, shift=1, axis=0) & np.roll(occ, shift=1, axis=1)
    verta = pixel_index
    vertb = np.roll(pixel_index, shift=1, axis=1)
    vertc = np.roll(pixel_index, shift=1, axis=0)
    face1 = np.stack([verta[is_tri_vert], vertb[is_tri_vert], vertc[is_tri_vert]], axis=1)
    
    # Combine the two sets of faces
    face = np.concatenate([face0, face1], axis=0)

    return face


def grid_triangulation(
    occ, vert, 
    normal=None, 
    color=None, 
    prune_verts=True, 
    return_o3d_mesh=False,
    return_boundary_verts=False):
    """ Turn an omage into a mesh.
    Args:
        occ: np.ndarray of shape (H, W, 1) representing the occupancy map.
        vert: np.ndarray of shape (H, W, 3) representing the 3D position of each vertex.
        normal: np.ndarray of shape (H, W, 3) representing the normal.
        return_uv: bool, whether to return uv as well.
        prune_verts: bool, whether to remove unused vertices to reduce size.
    """    
    occ = _pad_arr(occ)
    vert = _pad_arr(vert)
    normal = _pad_arr(normal) if normal is not None else None
    
    # Generate pixel indices array
    vert         = vert.reshape(-1, 3)
    normal       = normal.reshape(-1, 3) if normal is not None else None
    
    face = meshing_uv_map(occ)
    if face.shape[0] == 0: # no face, return empty mesh
        meshing_ret = dict( vert = np.zeros((0,3)), face = np.zeros((0,3)).astype(int), uv = np.zeros((0,2)))
        return meshing_ret

    # flip faces with inconsistent objnormal vs face normal
    if normal is not None:
        face_normal = np.cross(vert[face[:,1]] - vert[face[:,0]], vert[face[:,2]] - vert[face[:,0]])
        flip_mask = np.einsum('ij,ij->i', face_normal, normal[face[:,0]]) < 0
        face[flip_mask] = face[flip_mask][:,[0,2,1]]
    
    uv = _make_grid([0,0], [1,1], shape=(occ.shape[0], occ.shape[1]), mode='on')
    uv[..., [0,1]] = uv[..., [1,0]] # swap x, y to match the image coordinate system
    
    
    # TODO: snap
    
    
    meshing_ret=dict( vert=vert, face=face, uv=uv)

    if prune_verts:
        vert, face, unique_vert_ind = _prune_unused_vertices(vert, face)
        uv = uv[unique_vert_ind]
        for key in meshing_ret:
            if key not in ['vert', 'face', 'uv']:
                print(key, meshing_ret[key].shape)
                meshing_ret[key] = meshing_ret[key][unique_vert_ind]
        meshing_ret['vert'], meshing_ret['face'], meshing_ret['uv'] = vert, face, uv
        if color is not None:
            meshing_ret['color'] = color[unique_vert_ind]
        
    if return_o3d_mesh:
        return _to_o3d_mesh(
            meshing_ret['vert'], 
            meshing_ret['face'], 
            color=color if color is not None else np.random.rand(3))
        
    if return_boundary_verts: pass
        
    return meshing_ret


# data_dir = "E:\\lry\\code\\style3d_gen\\resources\\examples\\outputs_2"
data_dir = "E:\\lry\\data\\AIGP\\demo_v2\\Q2\\brep_reso_1024"

for idx in range(1,6):
    
    with open(os.path.join(data_dir, '%05d.pkl'%(idx)), 'rb') as f: data = pickle.load(f)
    position = data['surf_wcs']
    occ = data['surf_mask']
    
    # with open(os.path.join(data_dir, 'position', 'xyz_%d.pkl'%(idx)), 'rb') as f: position = pickle.load(f)
    # with open(os.path.join(data_dir, 'occ', 'mask_%d.pkl'%(idx)), 'rb') as f: occ = pickle.load(f)
    # occ = occ[:, 0, :, :][..., None] 
    # print('*** occ: ', occ.shape, occ.min(), occ.max(), occ.mean())

    meshes = []
    boundary_verts = []

    for s_idx in range(position.shape[0]):        
        # OPTION 1 : grid triangulation        
        mesh_ret = grid_triangulation(
            occ[s_idx] > 0.85,
            position[s_idx][..., :3],
            return_o3d_mesh=True
        )
        meshes.append(mesh_ret)
        
        # # OPTION 2: del triangulation
        # coords = np.argwhere(occ[s_idx] > 0.85)
        # plt.imshow(occ[s_idx], cmap='gray')
        # plt.scatter(coords[:, 1], coords[:, 0])
        # print(coords.shape, coords[0])
        # break
        
        # print("*** ", s_idx, mesh_ret['vert'].shape, mesh_ret['vert'].dtype, mesh_ret['face'].shape,  mesh_ret['face'].dtype)
        # meshes.append(mesh_ret)
        # verts = o3d.utility.Vector3dVector(mesh_ret['vert'])
        # faces = o3d.utility.Vector3iVector(mesh_ret['face'])
        # meshes.append(_to_o3d_mesh(mesh_ret['vert'], mesh_ret['face'], color=np.random.rand(3)))

    # for mesh in meshes:
    #     # calculate boundary facets
    #     pass

    # o3d.visualization.draw_geometries(meshes)
    o3d.visualization.draw_geometries(meshes, mesh_show_wireframe=True)
    break
