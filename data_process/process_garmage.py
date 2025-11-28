
import os
import random
import argparse
import json, pickle
from glob import glob
from tqdm import tqdm

import torch
import scipy
import einops
import skimage
import numpy as np
import nvdiffrast.torch as dr

import igl
from geomdl import fitting, BSpline, utilities

from concurrent.futures import ThreadPoolExecutor, as_completed
from geometry_utils.obj import read_obj

torch.set_grad_enabled(False)


_GLOBAL_SCALE = 2000.0
_GLOBAL_OFFSET = np.array([0.0, 1000.0, 0.0])

_GLOBAL_SCALE_UV = 1500.0
_GLOBAL_OFFSET_UV = np.array([0.0, 1000.0])


def _get_bbox(point_sets, points_mask=None):
    num_sets = point_sets.shape[0]
    p_dim = point_sets.shape[-1]
    
    if points_mask is None: points_mask = np.ones_like(point_sets[..., :1]).astype(bool)
    
    bboxes = []
    for idx in range(num_sets):
        valid_pnts = point_sets[idx].reshape(-1, p_dim)[points_mask[idx].reshape(-1)]
        min_point, max_point = np.min(valid_pnts, axis=0), np.max(valid_pnts, axis=0)
        bboxes.append(np.concatenate([min_point, max_point]))
    
    return np.vstack(bboxes)


def _get_scale_offset(point_sets, points_mask):
    p_dim = point_sets.shape[-1]
    valid_pnts = point_sets.reshape(-1, p_dim)[points_mask.reshape(-1)]

    min_vals = np.min(valid_pnts, axis=0)
    max_vals = np.max(valid_pnts, axis=0)
    
    offset = min_vals + (max_vals - min_vals) / 2
    scale = max(max_vals - min_vals)
    
    assert scale != 0, 'scale is zero'
    
    return offset, scale


def _interpolate_feature_dr(rast, pos, tris, feat, antialias=False):

    out, _ = dr.interpolate(feat[None], rast, tris)
    if antialias: out = dr.antialias(out, rast, pos, tris)
    out = out.detach().cpu().numpy()

    return out


def _downsample_sparse_pooling(x, factor=16, anti_aliasing=False):
    """ downsample input array with sparse pooling
    Args:
        x: np.ndarray, (H, W, C + 1), original feature tensor of shape (H, W, C) with additional occupancy channel
        factor: int, downsample factor
        anti_aliasing: bool, whether to use anti_aliasing
    Returns:
        'feat_down': np.ndarray, downsampled feat with edge snapping
    """
    # assuming feat is 1024 x 1024
    occ = x[..., -1]>=.5
    eroded_mask = scipy.ndimage.binary_erosion(occ, structure=np.ones((3,3))) # sqaure strucure is needed to get the corners
    edge_occ = ~eroded_mask & occ
    edge_feat = x.copy()
    edge_feat[edge_occ==0] = -1.
    
    edge_occ_patches = einops.rearrange(edge_occ, '(h1 h2) (w1 w2) -> h1 w1 h2 w2', h2=factor, w2=factor)
    edge_occ_down = edge_occ_patches.max(axis=-1).max(axis=-1)
    eod_0_count  = (edge_occ_patches==0).sum(axis=-1).sum(axis=-1)
    eod_1_count  = (edge_occ_patches==1).sum(axis=-1).sum(axis=-1)
    edge_feat_patches = einops.rearrange(edge_feat, '(h1 h2) (w1 w2) c-> h1 w1 h2 w2 c', h2=factor, w2=factor)
    edge_feat_down = edge_feat_patches.sum(axis=-2).sum(axis=-2) + eod_0_count[...,None]
    edge_feat_down = np.divide(edge_feat_down, eod_1_count[...,None], out=np.zeros_like(edge_feat_down), where=eod_1_count[...,None]!=0)

    downsampled = skimage.transform.resize(
        x, (x.shape[0]//factor,)*2, order=0, 
        preserve_range=False, anti_aliasing=anti_aliasing)
    
    # edge snapping
    downsampled = edge_feat_down * (edge_occ_down[...,None]) + downsampled * (1-edge_occ_down[...,None])
        
    return downsampled

def prepare_surf_data(
    glctx,
    mesh_obj,
    reso=64          # original rasterization resolution
    # down_factor=-1    # downsample factor
):

    verts = torch.from_numpy(mesh_obj.points).to(torch.float32).to('cuda')
    uv = torch.from_numpy(mesh_obj.point_data['obj:vt']).to(
        torch.float32).to('cuda')
    normals = torch.from_numpy(mesh_obj.point_data['obj:vn']).to(torch.float32).to('cuda')

    panel_ids = []
    
    uv_local = uv.clone()               # projected pixel coordinate for each vertex

    tris = []                           # all triangles
    tri_ranges = []                     # triangle range for each panel,
    start_idx = 0
    
    for idx, panel_id in enumerate(mesh_obj.field_data['obj:group_tags']):
        # if panel_id not in panel_data: print(f"\t Skipping panel {panel_id}"); continue

        panel_ids.append(panel_id)

        panel_faces = mesh_obj.cells[idx].data
        vert_ids = np.unique(panel_faces)

        tris.append(torch.from_numpy(panel_faces).to('cuda'))
        tri_ranges.append([start_idx, len(panel_faces)])
        start_idx += len(panel_faces)

        panel_bbox2d = torch.cat([
            torch.min(uv_local[vert_ids, :], dim=0, keepdim=True)[0],
            torch.max(uv_local[vert_ids, :], dim=0, keepdim=True)[0]
        ])

        uv_local[vert_ids, :] = (uv_local[vert_ids, :] - panel_bbox2d[0]) / (
            panel_bbox2d[1] - panel_bbox2d[0] + 1e-6)      # normalize to [0, 1]

        # # check boundary vertices
        # boundary = igl.boundary_facets(panel_faces)
        # boundary_verts_idx = np.unique(boundary)
        # print('[SURF] boundary_verts_idx: ', boundary_verts_idx.shape)

        # boundary_verts_uv.append(uv_local[boundary_verts_idx, :].cpu().numpy())
        # boundary_verts.append(verts[boundary_verts_idx, :].cpu().numpy())
        
    tris = torch.cat(tris, dim=0).to(torch.int32)
    # panel triangle range#
    tri_ranges = torch.tensor(tri_ranges, dtype=torch.int32)

    # normalize to [-1, 1]
    uv_local = uv_local[..., :2] * 2.0 - 1.0
    uv_local = torch.cat([
        uv_local,
        torch.zeros_like(uv_local[:, :1]),
        torch.ones_like(uv_local[:, :1])], dim=1)

    rast, _ = dr.rasterize(
        glctx, uv_local, tris, resolution=[1024, 1024], 
        ranges=tri_ranges, grad_db=False)

    vert_feat = torch.cat([verts, uv[..., :2], normals], dim=-1)    # dim: (N, 8), xyz + uv + normal
    surf_feat = _interpolate_feature_dr(rast, uv_local, tris, vert_feat)
    surf_mask = (rast[..., 3:]>0).squeeze(0).cpu().numpy()
        
    down_factor = 1024 // reso
    if down_factor > 1:
        downsampled_feat = []
        for s_idx in range(surf_feat.shape[0]):
            downsampled_feat.append(
                _downsample_sparse_pooling(
                    np.concatenate([surf_feat[s_idx], surf_mask[s_idx]], axis=-1), 
                    factor=down_factor, anti_aliasing=False
                    )
                )
        
        surf_feat = np.stack(downsampled_feat, axis=0)
        surf_feat, surf_mask = surf_feat[..., :-1], surf_feat[..., -1:].astype(bool)
        
    # spliting surf_feat
    surf_pnts, surf_uvs, surf_norms = \
        surf_feat[..., :3], surf_feat[..., 3:5], surf_feat[..., 5:8]
                                    
    return panel_ids, surf_pnts, surf_uvs, surf_norms, surf_mask


def normalize(
    surf_pnts,  surf_masks=None,
    global_offset=None, global_scale=None):
    
    """
    Various levels of normalization 
    """
            
    # Global normalization to -1~1
    if global_offset is None or global_scale is None:
        global_offset, global_scale = _get_scale_offset(surf_pnts, surf_masks)
        assert global_scale != 0, 'scale is zero'

    surfs_wcs, surfs_ncs = [], []

    # Normalize surface
    if surf_masks is None: surf_masks = np.ones_like(surf_pnts[..., :1]).astype(bool)
    # print('[NORMALIZE] surf_mask: ', surf_pnts.shape, surf_masks.shape, np.unique(surf_masks))  
    for surf_idx in range(surf_pnts.shape[0]):
        surf_pnt, surf_mask = surf_pnts[surf_idx], surf_masks[surf_idx]
        surf_pnt_wcs = (
            surf_pnt - global_offset[np.newaxis, np.newaxis, :]) / (global_scale * 0.5)
        surf_pnt_wcs = surf_pnt_wcs * surf_mask.astype(surf_pnt_wcs.dtype)
        surfs_wcs.append(surf_pnt_wcs)
        
        local_offset, local_scale = _get_scale_offset(surf_pnt_wcs, surf_mask)
        pnt_ncs = (surf_pnt_wcs - local_offset[np.newaxis, np.newaxis, :]) / (local_scale * 0.5)
        pnt_ncs = pnt_ncs * surf_mask.astype(pnt_ncs.dtype)
        surfs_ncs.append(pnt_ncs)

    surfs_wcs = np.stack(surfs_wcs)
    surfs_ncs = np.stack(surfs_ncs)

    return surfs_wcs, surfs_ncs, global_offset, global_scale


def process_data(
        data_item: str,
        glctx=None,
        num_surf_samples=256,
        use_uni_norm=True):

    # print('*** Processing data: ', data_item)

    mesh_fp = data_item+".obj"
    # pattern_fp = os.path.join(data_item, 'pattern.json')

    mesh_obj = read_obj(mesh_fp)
    # with open(pattern_fp, 'r', encoding='utf-8') as f:
    #     pattern_spec = json.load(f)

    # surf_uv: panel 3D points [num_panels, num_samples, num_samples, 3]
    if glctx is None: glctx = dr.RasterizeCudaContext()

    panel_ids, surf_pnts, surf_uvs, surf_norms, surf_mask = prepare_surf_data(
        glctx, mesh_obj=mesh_obj,
        reso=num_surf_samples
    )

    surfs_wcs, surfs_ncs, global_offset, global_scale = normalize(
        surf_pnts, surf_masks=surf_mask,
        global_offset=_GLOBAL_OFFSET if use_uni_norm else None, 
        global_scale=_GLOBAL_SCALE if use_uni_norm else None
    )
    
    surfs_uv_wcs, surfs_uv_ncs, uv_offset, uv_scale = normalize(
        surf_uvs, surf_masks=surf_mask,
        global_offset=_GLOBAL_OFFSET_UV if use_uni_norm else None, 
        global_scale=_GLOBAL_SCALE_UV if use_uni_norm else None
    )
    
    result = {
        'data_fp': os.path.splitext(os.path.basename(data_item))[0],
        # xyz
        'surf_mask': surf_mask.astype(bool),
        'surf_wcs': surfs_wcs.astype(np.float32),
        'surf_ncs': surfs_ncs.astype(np.float32),
        'global_offset': global_offset.astype(np.float32),
        'global_scale': global_scale,
        # uv
        'surf_uv_wcs': surfs_uv_wcs.astype(np.float32),
        'surf_uv_ncs': surfs_uv_ncs.astype(np.float32),
        'uv_offset': uv_offset.astype(np.float32),
        'uv_scale': uv_scale,
        # normal
        'surf_normals': surf_norms.astype(np.float32),
    }
    
    # Calculate bounding boxes
    result['surf_bbox_wcs'] = _get_bbox(result['surf_wcs'], surf_mask)
    
    # Calculating UV bounding bbox
    result['surf_uv_bbox_wcs'] = _get_bbox(result['surf_uv_wcs'], surf_mask)
        
    return result


def process_item(data_idx, data_item, args, glctx):
    # try:
    os.makedirs(args.output, exist_ok=True)
    uuid = os.path.basename(data_item)
    output_fp = os.path.join(args.output, f'{uuid}.pkl')

    result = process_data(
        data_item, glctx=glctx,
        num_surf_samples=args.nf)
    
    with open(output_fp, 'wb') as f: pickle.dump(result, f)
    return True, data_item

    # except Exception as e:
    #     return False, f"{data_item} | [ERROR] {e}"


def main(args):
    glctx = dr.RasterizeCudaContext()
    print('[DONE] Init renderer.', type(glctx))

    data_root_dirs = args.input.split(',')
    print('Input directories: ', data_root_dirs)

    data_items = []
    for idx, data_root in enumerate(data_root_dirs):
        cur_data_items = sorted([x.replace(".obj", "") for x in glob(
            os.path.join(data_root, '*.obj'), recursive=True)])
        data_items += cur_data_items
        print('[%02d/%02d] Found %d items in %s.' % (idx + 1, len(data_root_dirs), len(cur_data_items), data_root))
    print('Total items: ', len(data_items))

    log_file = os.path.join(args.output, 'app.log')
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
            processed = [x.split("\t")[0] for x in lines if x.split("\t")[1].strip() == "1"]
            data_items = [x for x in data_items if x not in processed]

    if args.range is not None:
        if ',' in args.range:
            begin, end = args.range.split(",")
            begin, end = max(0, int(begin)), min(int(end), len(data_items))
            data_items = data_items[begin:end]
            print("Extracting range: %d %s" % (len(data_items), args.output))
        else:
            data_items = random.choices(data_items, k=int(args.range))
            print("Extracting random items: %d %s" % (len(data_items), args.output))

    os.makedirs(args.output, exist_ok=True)

    failed_items = []
    with open(log_file, 'a+') as f:
        with ThreadPoolExecutor(max_workers=1) as executor:  # 可以调整max_workers以改变并行度
            futures = {executor.submit(
                process_item, data_idx, data_item, args, glctx): data_item for data_idx, data_item in enumerate(data_items)}

            for future in tqdm(as_completed(futures), total=len(futures)):
                result, data_item = future.result()
                if not result:
                    data_item, err_code = data_item.split('|')
                    data_item = data_item.strip()
                    err_code = err_code.strip()
                    failed_items.append(data_item)
                    print('[ERROR] Failed to process data:', data_item, err_code)
                    f.write(f"{data_item}\t 0\n")
                else:
                    f.write(f"{data_item}\t 1\n")

    with open(os.path.join(args.output, 'failed_items.log'), 'w') as f:
        for item in failed_items: f.write(item + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="resources/examples/raw",
                        type=str, help="Input directories splited bt comma.")
    parser.add_argument("-o", "--output", default='resources/examples/garmage',
                        type=str, help="Output directory.")
    parser.add_argument("-r", "--range", default=None, type=str, 
                        help="Path to executable.")
    parser.add_argument('--nf', default=256, type=int, help='Number of surface samples.')

    args, cfg_cmd = parser.parse_known_args()

    main(args)