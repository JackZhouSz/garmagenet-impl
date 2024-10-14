import pickle

import cv2

import open3d as o3d
import numpy as np

from sklearn.cluster import DBSCAN

import matplotlib as mpl
from matplotlib import pyplot as plt


def _to_o3d_pc(xyz: np.ndarray, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    print('[_to_o3d_pc] color: ', pcd.points)
        
    if color is not None:
        if len(color) != len(xyz): 
            color = np.array(color)[None].repeat(len(xyz), axis=0)
        pcd.colors = o3d.utility.Vector3dVector(color)
    else:
        pcd.colors = o3d.utility.Vector3dVector(0.771 * np.ones_like(xyz))

    return pcd


def _pad_arr(arr, pad_size=10):
    return np.pad(
        arr, 
        ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)),   # pad size to each dimension, require tensor to have size (H,W, C)
        mode='constant', 
        constant_values=0)


def resample_boundary(points, delta, outlier_thresh=0.05):

    points = np.asarray(points)
    
    # remove outliers
    deltas_prev = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
    deltas_next = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)    
        
    valid_pts = np.logical_and(deltas_prev < outlier_thresh, deltas_next < outlier_thresh)
    points = points[valid_pts, :]
        
    # Compute distances between consecutive points
    points = np.vstack([points, points[0]])  # Close the boundary
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    
    # Compute cumulative arc length
    arc_lengths = np.insert(np.cumsum(deltas), 0, 0)

    # Total length of the boundary
    total_length = arc_lengths[-1]

    # Number of new points
    num_points = int(np.ceil(total_length / delta)) + 1

    # New equally spaced arc lengths
    new_arc_lengths = np.linspace(0, total_length, num=num_points)

    # Interpolate to find new points
    new_points = np.zeros((num_points, 3))
    for i in range(3):  # For x, y, z coordinates
        new_points[:, i] = np.interp(new_arc_lengths, arc_lengths, points[:, i])

    return new_points


# load data
mask_fp = "..\\resources\\examples\\breps\\data\\mask\\mask_8.pkl"
geo_fp = mask_fp.replace('mask', 'xyz')
with open(geo_fp, 'rb') as f: geo_orig = pickle.load(f)
with open(mask_fp, 'rb') as f: mask = pickle.load(f)

geo_orig = _pad_arr(geo_orig, pad_size=5)
mask = _pad_arr(mask, pad_size=5)

# visualization buffers
surf_points = []
surf_colors = []
boundary_points = []
boundary_point_colors = []

# process data
for s_idx in range(mask.shape[0]):
    
    geo_dist = np.linalg.norm(geo_orig[s_idx], axis=-1)
    if geo_dist.min() < 1e-6 and geo_dist.max() < 1e-6: continue
        
    valid_pts = np.where(mask[s_idx, :, :, 0] > 0.5)    
    
    valid_pts = geo_orig[s_idx, valid_pts[0], valid_pts[1], :]    
    valid_pts = valid_pts[np.random.randint(0, valid_pts.shape[0], 1000), :]
    surf_points.append(valid_pts)
    surf_colors.append(np.zeros_like(valid_pts) + 0.5)
    
    # erode mask image 
    mask_img = (mask[s_idx] * 255.0).astype(np.uint8) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) 
    mask_img = cv2.erode(mask_img, kernel, iterations=1)
    mask_img[mask_img >= 150] = 255
    mask_img[mask_img < 150] = 0 
    
    # extract contours from mask image
    _, thresh = cv2.threshold(mask_img, 128, 255, cv2.THRESH_BINARY)        
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_pts = [np.squeeze(contour, axis=1) for contour in contours if contour.shape[0] > 16]

    for contour in contours:
        if contour.shape[0] < 16: continue  # custom threshold to remove small contours
        contour_pts = np.squeeze(contour, axis=1)
    
        # extract boundary points
        geo_arr = geo_orig[s_idx]
        geo_sample_pts = geo_arr[contour_pts[:, 1], contour_pts[:, 0], :]    
        geo_sample_pts = geo_sample_pts[np.linalg.norm(geo_sample_pts, axis=-1) > 0.01, :]
        
        # resample boundary points
        geo_sample_pts = resample_boundary(geo_sample_pts, 0.01, outlier_thresh=0.1)

        # save for visualization
        cmap = mpl.colormaps['rainbow']
        boundary_points.append(geo_sample_pts)
        boundary_point_colors.append(cmap(np.linspace(0, 1, len(geo_sample_pts)))[:, :3])
            

###################### visualization ########################
boundary_points = np.concatenate(boundary_points, axis=0).astype(np.float32)
boundary_point_colors = np.concatenate(boundary_point_colors, axis=0)

# filtering points that are too close to origin
close_to_origin = np.linalg.norm(boundary_points, axis=1) > 0.001
boundary_points = boundary_points[close_to_origin, :]
boundary_point_colors = boundary_point_colors[close_to_origin, :]
boundary_pcd = _to_o3d_pc(boundary_points, boundary_point_colors)

surf_points = np.concatenate(surf_points, axis=0).astype(np.float32)
surf_pcd = _to_o3d_pc(surf_points)    
            
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0., 0., 0.])
o3d.visualization.draw_geometries([
    mesh_frame, 
    boundary_pcd, 
    # surf_pcd
    ])

# o3d.visualization.draw_geometries([
#     mesh_frame, 
#     boundary_pcd, 
#     surf_pcd
#     ])
############################################################