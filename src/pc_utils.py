import math
import numpy as np


def farthest_point_sample(points, npoint, max_npoint=40000):
    # limit point num py uniform smaple
    if len(points) > max_npoint:
        points = points[::math.ceil(len(points)/max_npoint)]

    N, D = points.shape
    xyz = points[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    points = points[centroids.astype(np.int32)]
    return points


def normalize_pointcloud(pc: np.ndarray, range=1) -> np.ndarray:
    assert pc.ndim == 2 and pc.shape[1] == 3

    min_xyz = pc.min(axis=0)
    max_xyz = pc.max(axis=0)

    center = (min_xyz + max_xyz) / 2.0
    pc_centered = pc - center

    scale = (max_xyz - min_xyz).max()

    pc_normalized = pc_centered / (scale / 2.0) * range
    return pc_normalized