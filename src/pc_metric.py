
import os
import pickle
import random
import argparse
import warnings
from glob import glob
from tqdm import tqdm
from pathlib import Path

import torch
import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from plyfile import PlyData
import multiprocessing
from chamferdist import ChamferDistance

N_POINTS = 2000


def _denormalize_pts(pts, bbox):
    pos_dim =  pts.shape[-1]
    bbox_min = bbox[..., :pos_dim][:, None, ...]
    bbox_max = bbox[..., pos_dim:][:, None, ...]
    bbox_scale = np.max(bbox_max - bbox_min, axis=-1, keepdims=True) * 0.5
    bbox_offset = (bbox_max + bbox_min) / 2.0
    return pts * bbox_scale + bbox_offset

def find_files(folder, extension):
    return sorted([Path(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(extension)])


def read_ply(path):
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
        x = np.array(plydata['vertex']['x'])
        y = np.array(plydata['vertex']['y'])
        z = np.array(plydata['vertex']['z'])
        vertex = np.stack([x, y, z], axis=1)
    return vertex


def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


def _pairwise_CD(sample_pcs, ref_pcs, batch_size):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(N_sample)
    matched_gt = []
    pbar = tqdm(iterator)
    chamfer_dist = ChamferDistance()

    for sample_b_start in pbar:
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()
            
            dl, dr, idx1, idx2 = chamfer_dist(sample_batch_exp,ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        all_cd.append(cd_lst)

        hit = np.argmin(cd_lst.detach().cpu().numpy()[0])
        matched_gt.append(hit)
        pbar.set_postfix({"cov": len(np.unique(matched_gt)) * 1.0 / N_ref})

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref

    return all_cd


def compute_cov_mmd(sample_pcs, ref_pcs, batch_size):
    all_dist = _pairwise_CD(sample_pcs, ref_pcs, batch_size)
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)

    return {
        'MMD-CD': mmd.item(),
        'COV-CD': cov.item(),
    }


def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, in_unit_sphere, resolution=28):
    '''Computes the JSD between two sets of point-clouds, as introduced in the paper ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    '''
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False):
    '''Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    '''
    epsilon = 10e-4
    bound = 1 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        print(abs(np.max(pclouds)), abs(np.min(pclouds)))
        warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        p = 0.0
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    '''Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    '''
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1) * 2
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5 * 2
                grid[i, j, k, 1] = j * spacing - 0.5 * 2
                grid[i, j, k, 2] = k * spacing - 0.5 * 2

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[np.linalg.norm(grid, axis=1) <= 0.5]

    return grid, spacing


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    '''another way of computing JSD'''

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))


# def downsample_pc(points, n):
#     sample_idx = random.sample(list(range(points.shape[0])), n)
#     return points[sample_idx]
#
#
# def normalize_pc(points):
#     # normalize
#     mean = np.mean(points, axis=0)
#     points = (points - mean)
#     # fit to unit cube
#     scale = np.max(np.abs(points))
#     points = points / scale
#     return points
#
#
# def collect_pc(cad_folder):
#     pc_path = find_files(os.path.join(cad_folder, 'pcd'), 'final_pcd.ply')
#     if len(pc_path) == 0:
#         return []
#     pc_path = pc_path[-1] # final pcd
#     pc = read_ply(pc_path)
#     if pc.shape[0] > N_POINTS:
#         pc = downsample_pc(pc, N_POINTS)
#     pc = normalize_pc(pc)
#     return pc

# def collect_pc2(cad_folder):
#     pc = read_ply(cad_folder)
#     if pc.shape[0] > N_POINTS:
#         pc = downsample_pc(pc, N_POINTS)
#     pc = normalize_pc(pc)
#     return pc
#
# theta_x = np.radians(90)  # Rotation angle around X-axis
# theta_y = np.radians(90)  # Rotation angle around Y-axis
# theta_z = np.radians(180)  # Rotation angle around Z-axis
#
# # Create individual rotation matrices
# Rx = np.array([[1, 0, 0],
#                [0, np.cos(theta_x), -np.sin(theta_x)],
#                [0, np.sin(theta_x), np.cos(theta_x)]])
#
# Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
#                [0, 1, 0],
#                [-np.sin(theta_y), 0, np.cos(theta_y)]])
#
# Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
#                [np.sin(theta_z), np.cos(theta_z), 0],
#                [0, 0, 1]])
#
# rotation_matrix = np.dot(np.dot(Rz, Ry), Rx)
#
# def collect_pc3(cad_folder):
#     pc = read_ply(cad_folder)
#     if pc.shape[0] > N_POINTS:
#         pc = downsample_pc(pc, N_POINTS)
#     pc = normalize_pc(pc)
#     rotated_point_cloud = np.dot(pc, rotation_matrix.T).astype(np.float32)  # Transpose the rotation matrix to apply it correctly
#     return rotated_point_cloud

def load_data_with_prefix(root_folder, prefix):
    data_files = sorted(glob(os.path.join(root_folder,f"*.{prefix}")))
    return data_files


def random_split_integer(N, M, seed=None):
    q, r = divmod(N, M)
    result = [q + 1] * r + [q] * (M - r)
    if seed is not None:
        random.seed(seed)  # 可选，设定随机种子保证可复现
    random.shuffle(result)
    return result

def load_garment_pc(shape_path, sample_num=2000):
    with open(shape_path, 'rb') as f:
        data = pickle.load(f)

    if "surf_bbox_wcs" in data:
        bbox = data.get("surf_bbox_wcs")
    else:
        bbox = data.get("surf_bbox")
    points = data["surf_ncs"]
    point_masks = data["surf_mask"]
    n_surfs = len(points)
    sample_num = random_split_integer(sample_num, n_surfs)
    points = _denormalize_pts(points.reshape(-1, 65536, 3), bbox).reshape(-1, 256, 256, 3)
    sample_list = []
    for idx in range(n_surfs):
        cur_points, cur_points_mask = points[idx].reshape(-1, 3), point_masks[idx].reshape(-1)
        cur_points = cur_points[cur_points_mask, :]
        rand_idx = np.random.choice(cur_points.shape[0], sample_num[idx], replace=False)
        cur_points = cur_points[rand_idx, :]
        sample_list.append(cur_points)


    pc = np.concatenate(sample_list, axis=0)
    # pc = styleXD_denormalize(pc)
    # from src.vis import pointcloud_visualize
    # pointcloud_visualize(pc)
    return pc


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--fake", type=str)
#     parser.add_argument("--real", type=str)
#     parser.add_argument("--n_test", type=int, default=1000)
#     parser.add_argument("--multi", type=int, default=3)
#     parser.add_argument("--times", type=int, default=10)
#     parser.add_argument("--batch_size", type=int, default=64)
#     args = parser.parse_args()
#
#     print("n_test: {}, multiplier: {}, repeat times: {}".format(args.n_test, args.multi, args.times))
#
#     args.output = args.fake + '_results.txt'
#
#     # Load reference pcd
#     num_cpus = multiprocessing.cpu_count()
#     ref_pcs = []
#     shape_paths = load_data_with_prefix(args.real, '.ply')
#     load_iter = multiprocessing.Pool(num_cpus).imap(collect_pc2, shape_paths)
#     for pc in tqdm(load_iter, total=len(shape_paths)):
#         if len(pc) > 0:
#             ref_pcs.append(pc)
#     ref_pcs = np.stack(ref_pcs, axis=0)
#     print("real point clouds: {}".format(ref_pcs.shape))
#
#
#     # Load fake pcd
#     sample_pcs = []
#     shape_paths = load_data_with_prefix(args.fake, '.ply')
#     load_iter = multiprocessing.Pool(num_cpus).imap(collect_pc2, shape_paths)
#     for pc in tqdm(load_iter, total=len(shape_paths)):
#         if len(pc) > 0:
#             sample_pcs.append(pc)
#     sample_pcs = np.stack(sample_pcs, axis=0)
#
#     print("fake point clouds: {}".format(sample_pcs.shape))
#
#     # Testing
#     fp = open(args.output, "w")
#     result_list = []
#     for i in range(args.times):
#         print("iteration {}...".format(i))
#         select_idx = random.sample(list(range(len(sample_pcs))), int(args.multi * args.n_test))
#         rand_sample_pcs = sample_pcs[select_idx]
#
#         select_idx = random.sample(list(range(len(ref_pcs))), args.n_test)
#         rand_ref_pcs = ref_pcs[select_idx]
#
#         jsd = jsd_between_point_cloud_sets(rand_sample_pcs, rand_ref_pcs, in_unit_sphere=False)
#         with torch.no_grad():
#             rand_sample_pcs = torch.tensor(rand_sample_pcs).cuda()
#             rand_ref_pcs = torch.tensor(rand_ref_pcs).cuda()
#             result = compute_cov_mmd(rand_sample_pcs, rand_ref_pcs, batch_size=args.batch_size)
#         result.update({"JSD": jsd})
#
#         print(result)
#         print(result, file=fp)
#         result_list.append(result)
#     avg_result = {}
#     for k in result_list[0].keys():
#         avg_result.update({"avg-" + k: np.mean([x[k] for x in result_list])})
#     print("average result:")
#     print(avg_result)
#     print(avg_result, file=fp)
#     fp.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, default='/home/Ex1/画论文teasor的素材/生成的结果/第4批/surfz_e230000_1k')
    parser.add_argument("--test_dir", type=str, default='/home/Ex1/Datasets/S3D/brep_reso_256_edge_snap_with_caption')
    parser.add_argument("--data_list", default="/home/Ex1/ProjectFiles/Pycharm_MyPaperWork/style3d_gen/_LSR/gen_data_list/output/data_list.pkl")
    parser.add_argument("--cache_dir", type=str, default='log/pc_metric/cache')

    parser.add_argument("--n_sample", type=int, default=128)
    parser.add_argument("--n_test", type=int, default=128)
    parser.add_argument("--times", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    args.output = os.path.join(args.cache_dir, 'results.txt')

    args.data_list=None

    # Load reference pcd
    num_cpus = multiprocessing.cpu_count()
    ref_pcs = []
    if args.data_list is None:
        shape_paths = load_data_with_prefix(args.test_dir, 'pkl')[:128]
    else:  # 仅使用验证集
        with open(args.data_list, 'rb') as f:
            data_list = pickle.load(f)["val"]
        shape_paths = [os.path.join(args.test_dir, p.split("/")[-1]) for p in data_list]
    load_iter = multiprocessing.Pool(num_cpus).imap(load_garment_pc, shape_paths)
    for pc in tqdm(load_iter, total=len(shape_paths)):
        if len(pc) > 0:
            ref_pcs.append(pc)
    ref_pcs = np.stack(ref_pcs, axis=0)
    print("real point clouds: {}".format(ref_pcs.shape))

    # Load fake pcd
    sample_pcs = []
    shape_paths = load_data_with_prefix(args.sample_dir, 'pkl')[:128]
    load_iter = multiprocessing.Pool(num_cpus).imap(load_garment_pc, shape_paths)
    for pc in tqdm(load_iter, total=len(shape_paths)):
        if len(pc) > 0:
            sample_pcs.append(pc)
    sample_pcs = np.stack(sample_pcs, axis=0)
    print("fake point clouds: {}".format(sample_pcs.shape))

    # Testing
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fp:
        result_list = []
        for i in range(args.times):
            print("iteration {}...".format(i))
            select_idx = random.sample(list(range(len(sample_pcs))), int(args.n_sample))
            rand_sample_pcs = sample_pcs[select_idx]

            select_idx = random.sample(list(range(len(ref_pcs))), args.n_test)
            rand_ref_pcs = ref_pcs[select_idx]

            jsd = jsd_between_point_cloud_sets(rand_sample_pcs, rand_ref_pcs, in_unit_sphere=False)
            with torch.no_grad():
                rand_sample_pcs = torch.tensor(rand_sample_pcs).cuda()
                rand_ref_pcs = torch.tensor(rand_ref_pcs).cuda()
                result = compute_cov_mmd(rand_sample_pcs, rand_ref_pcs, batch_size=args.batch_size)
            result.update({"JSD": jsd})

            print(result)
            print(result, file=fp)
            result_list.append(result)
        avg_result = {}
        for k in result_list[0].keys():
            avg_result.update({"avg-" + k: np.mean([x[k] for x in result_list])})
        print("average result:")
        print(avg_result)
        print(avg_result, file=fp)

if __name__ == '__main__':
    main()
