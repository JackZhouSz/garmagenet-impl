import json
import os
import argparse
from glob import glob
from tqdm import tqdm
import pickle
import numpy as np
import torch
import nvdiffrast.torch as dr

import random

import igl
from geomdl import fitting, BSpline, utilities
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.colors import to_rgba

torch.set_grad_enabled(False)

from geometry_utils.obj import read_obj  # type: ignore


_CMAP = {
    "帽": {"alias": "帽", "color": "#F7815D"},
    "领": {"alias": "领", "color": "#F9D26D"},
    "肩": {"alias": "肩", "color": "#F23434"},
    "袖片": {"alias": "袖片", "color": "#C4DBBE"},
    "袖口": {"alias": "袖口", "color": "#F0EDA8"},
    "衣身前中": {"alias": "衣身前中", "color": "#8CA740"},
    "衣身后中": {"alias": "衣身后中", "color": "#4087A7"},
    "衣身侧": {"alias": "衣身侧", "color": "#DF7D7E"},
    "底摆": {"alias": "底摆", "color": "#DACBBD"},
    "腰头": {"alias": "腰头", "color": "#DABDD1"},
    "裙前中": {"alias": "裙前中", "color": "#46B974"},
    "裙后中": {"alias": "裙后中", "color": "#6B68F5"},
    "裙侧": {"alias": "裙侧", "color": "#D37F50"},

    "橡筋": {"alias": "橡筋", "color": "#696969"},
    "木耳边": {"alias": "木耳边", "color": "#696969"},
    "袖笼拼条": {"alias": "袖笼拼条", "color": "#696969"},
    "荷叶边": {"alias": "荷叶边", "color": "#696969"},
    "绑带": {"alias": "绑带", "color": "#696969"}
}

_PANEL_CLS = [
    '帽', '领', '肩', '袖片', '袖口', '衣身前中', '衣身后中', '衣身侧', '底摆', '腰头', '裙前中', '裙后中', '裙侧', '橡筋', '木耳边', '袖笼拼条', '荷叶边', '绑带']

_PANEL_COLORS = np.array(
    [(0., 0., 0., 0.)] + [to_rgba(_CMAP[_PANEL_CLS[idx]]['color'])
                          for idx in range(len(_PANEL_CLS))]
)

_AVATAR_BBOX = np.array([
    [-449.00006104,  191.10876465, -178.34872437],      # min
    [447.45980835, 1831.29016113,  174.13575745]       # max
])

_GLOBAL_SCALE = 1000.0
_GLOBAL_OFFSET = np.array([0.0, 1000.0, 0.0])
_GLOBAL_SCALE_UV = 2500.0
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
    if antialias: out = dr.antialias(out, rast, pos[None], tris)
    out = out.detach().cpu().numpy()

    ############## visualize sample points ##############
    # out = out[:, ::-1, :, :]     # flip image coordinate
    # for idx in range(out.shape[0]):
    #     plt.imshow(out[idx])
    #     plt.show()
    #     plt.imshow(out[idx])
    #     plt.show()

    return out


def _sample_curve_points_2d(edge_spec, num_samples=32):
    bezierPts = np.asarray(edge_spec['bezierPoints'])[:, :2]
    ctrlPts = np.asarray(edge_spec['controlPoints'])[:, :2]

    if np.any(bezierPts) and len(ctrlPts) == 2:
        if not np.any(bezierPts[1]):
            # print('quadratic bezier')
            bezierPts[1] = 2.0 / 3.0 * \
                (bezierPts[0] + ctrlPts[0] - ctrlPts[1])
            bezierPts[0] = 2.0 / 3.0 * bezierPts[0]

        bezierPts = ctrlPts + bezierPts
        curve = BSpline.Curve()
        curve.degree = 3
        curve.ctrlpts = [ctrlPts[0].tolist()] + \
            bezierPts.tolist() + [ctrlPts[1].tolist()]
        curve.knotvector = utilities.generate_knot_vector(
            curve.degree, len(curve.ctrlpts))
        curve.sample_size = num_samples
        curve.evaluate()

        evalpts = np.array(curve.evalpts)

    else:
        if ctrlPts.shape[0] <= 2:
            evalpts = ctrlPts
            t_values = np.linspace(0, 1, num_samples)
            evalpts = np.outer(
                1 - t_values, evalpts[0]) + np.outer(t_values, evalpts[1])
        else:
            curve = fitting.interpolate_curve(
                ctrlPts.tolist(), degree=2 if ctrlPts.shape[0] < 5 else 3)
            curve.sample_size = num_samples
            curve.evaluate()
            evalpts = np.array(curve.evalpts)

    return evalpts


def _project_curve_to_line_segments(curve_pts, line_segments, line_features):
    """
    Project curve points to the boundary edges of a triangle mesh, returen its 3d positions.

    Parameters:
    - curve_pts: query points, A numpy array of shape (num_points, 2)
    - line_segments: line segments a numpy array of shape (num_edges, 2, 2)
    - line_features: line features a numpy array of shape (num_edges, 2, feature_dim)

    Returns:
    - An array of shape (num_points, feature_dim), where each elements is the projected feature of the query point.
    """

    # find the belonging line segment
    query_dist_start = np.linalg.norm(
        curve_pts[:, None, :] - line_segments[:, 0, :][None], axis=-1)       # (num_sample, num_edges)
    query_dist_end = np.linalg.norm(
        curve_pts[:, None, :] - line_segments[:, 1, :][None], axis=-1)       # (num_sample, num_edges)
    query_dist = query_dist_start + query_dist_end

    belonging_segments = np.argmin(query_dist, axis=-1)

    # start_pt = line_segments[belonging_segments, 0, :]
    # end_pt = line_segments[belonging_segments, 1, :]

    start_feat = line_features[belonging_segments, 0, :]
    end_feat = line_features[belonging_segments, 1, :]

    dist_start = query_dist_start[np.arange(
        query_dist_start.shape[0]), belonging_segments]
    dist_end = query_dist_end[np.arange(
        query_dist_end.shape[0]), belonging_segments]
    dist_total = query_dist[np.arange(query_dist.shape[0]), belonging_segments]

    assert np.allclose(dist_total, dist_start +
                       dist_end), "Invalid distance calculation!"

    t_start = dist_start / dist_total
    t_end = dist_end / dist_total

    query_feat = t_start[:, None] * start_feat + t_end[:, None] * end_feat

    return belonging_segments, query_feat


def prepare_edge_data(
    mesh_obj,
    pattern_spec,
    reso=64
):

    verts = mesh_obj.points
    uv = mesh_obj.point_data['obj:vt']
    normals = mesh_obj.point_data['obj:vn']

    panel_data = dict([(x['id'], x) for x in pattern_spec['panels']])
    edge_ids, edge_wcs, edge_uvs, edge_normals = [], [], [], []
    corner_wcs, corner_uvs, corner_normals = [], [], []
    
    faceEdge_adj = {}

    for idx, panel_id in enumerate(mesh_obj.field_data['obj:group_tags']):
        if panel_id not in faceEdge_adj: faceEdge_adj[panel_id] = []
        panel_faces = mesh_obj.cells[idx].data
        panel_spec = panel_data[panel_id]

        panel_center = np.asarray(panel_spec['center'])

        boundary = igl.boundary_facets(panel_faces)
        boundary_uv = np.stack([
            (uv[boundary[:, 0], :]-panel_center)[:, :2],
            (uv[boundary[:, 1], :]-panel_center)[:, :2]],
            axis=1
        )
        boundary_pos = np.stack(
            [verts[boundary[:, 0], :], verts[boundary[:, 1], :]], axis=1)
        boundary_normals = np.stack(
            [normals[boundary[:, 0], :], normals[boundary[:, 1], :]], axis=1)

        panel_corner_uvs = []

        for seg_edge in panel_spec['seqEdges']:
            # skip inner line, only consider sew line
            if seg_edge['type'] != 3: continue
            
            panel_corner_uvs.append(np.asarray(
                seg_edge['vertices'], dtype=np.float32))

            # processing edge
            for edge in seg_edge['edges']:
                edge_samples = _sample_curve_points_2d(edge, reso)

                edge_bbox = (np.min(edge_samples, axis=0)-5.0,
                             np.max(edge_samples, axis=0)+5.0)

                # find boundary vertices within the edge bounding box
                within_bbox = np.all(
                    (boundary_uv[:, 0, :] >= edge_bbox[0]) &
                    (boundary_uv[:, 0, :] <= edge_bbox[1]) &
                    (boundary_uv[:, 1, :] >= edge_bbox[0]) &
                    (boundary_uv[:, 1, :] <= edge_bbox[1]), axis=-1)

                if within_bbox.sum() == 0: continue
                
                _, sample_pos = _project_curve_to_line_segments(
                    edge_samples, boundary_uv[within_bbox, :, :], 
                    np.concatenate([
                        boundary_pos[within_bbox, :, :], 
                        boundary_normals[within_bbox, :, :]], 
                        axis=-1)
                    )
                
                sample_pos, sample_normal = sample_pos[:, :3], sample_pos[:, 3:]
                edge_samples = edge_samples + panel_center[None, :2]

                edge_ids.append(edge['id'])
                edge_wcs.append(sample_pos)
                edge_uvs.append(edge_samples)
                edge_normals.append(sample_normal)

                faceEdge_adj[panel_id].append(edge['id'])

        # processing corner
        panel_corner_uvs = np.concatenate(panel_corner_uvs, axis=0)
        boundary_verts = np.unique(boundary)
        boundary_uv = uv[boundary_verts, :] - panel_center[None]
        boundary_pos = verts[boundary_verts, :]
        boundary_normals = normals[boundary_verts, :]

        panel_corner_vert_dists = np.linalg.norm(
            panel_corner_uvs[:, None, :] - boundary_uv[None], axis=-1)
        panel_corner_vert_idx = np.argmin(panel_corner_vert_dists, axis=-1)
        panel_corner_wcs = boundary_pos[panel_corner_vert_idx, :]
        panel_corner_normals = boundary_normals[panel_corner_vert_idx, :]
        panel_corner_uvs = (panel_corner_uvs + panel_center[None])[..., :2]
        
        # # change to [start, end] tuples
        # panel_corner_wcs = np.stack(
        #     [panel_corner_wcs, panel_corner_wcs[1:, :]]
        # )
        
        # global verts coordinates
        corner_wcs.append(panel_corner_wcs)
        corner_uvs.append(panel_corner_uvs)
        corner_normals.append(panel_corner_normals)

    edge_wcs = np.stack(edge_wcs, axis=0)
    edge_uvs = np.stack(edge_uvs, axis=0)
    edge_normals = np.stack(edge_normals, axis=0)
    
    corner_uvs = np.concatenate(corner_uvs, axis=0)
    corner_wcs = np.concatenate(corner_wcs, axis=0)
    corner_normals = np.concatenate(corner_normals, axis=0)

    return edge_ids, edge_wcs, edge_uvs, edge_normals, corner_wcs, corner_uvs, corner_normals, faceEdge_adj


def prepare_surf_data(
    glctx,
    mesh_obj,
    pattern_spec,
    reso=64
):

    verts = torch.from_numpy(mesh_obj.points).to(torch.float32).to('cuda')
    uv = torch.from_numpy(mesh_obj.point_data['obj:vt']).to(
        torch.float32).to('cuda')
    normals = torch.from_numpy(mesh_obj.point_data['obj:vn']).to(torch.float32).to('cuda')
    # print('*** before normalization: ',
    #       verts.min(dim=0)[0], verts.max(dim=0)[0])

    panel_data = dict([(x['id'], x) for x in pattern_spec['panels']])

    panel_ids = []
    panel_cls = []
    
    uv_local = uv.clone()               # projected pixel coordinate for each vertex

    tris = []                           # all triangles
    tri_ranges = []                     # triangle range for each panel,
    start_idx = 0

    for idx, panel_id in enumerate(mesh_obj.field_data['obj:group_tags']):
        panel_ids.append(panel_id)

        panel_seg_id = panel_data[panel_id]['label'].strip()
        panel_seg_id = _PANEL_CLS.index(panel_seg_id) + 1 if panel_seg_id in _PANEL_CLS else 0.
        panel_cls.append(panel_seg_id)

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
        glctx, uv_local, tris, resolution=[reso, reso], ranges=tri_ranges)

    surf_pnts = _interpolate_feature_dr(rast, uv_local, tris, verts)
    surf_uvs = _interpolate_feature_dr(rast, uv_local, tris, uv)[..., :2]
    surf_norms = _interpolate_feature_dr(rast, uv_local, tris, normals)
    surf_mask = (rast[..., 3:]>0).squeeze(0).cpu().numpy()
    
    return panel_ids, panel_cls, surf_pnts, surf_uvs, surf_norms, surf_mask


def face_edge_adj(json_content: dict, panel_ids: list, edge_ids: list):
    """
    TODO:Extract adjacent information between face and edge

    Args:
        json_content (dict): pattern.json content
        panel_ids(list): list of panel uuids
        edge_ids(list): list of edge uuids

    Returns:
        faceEdge_adj: A list of N sublist, where each sublist represents the adjacent edge IDs to a face
    """
    panel_ids = np.array(panel_ids)
    edge_ids = np.array(edge_ids)

    id_info = {}

    panel_edge_dict = {}
    for panel in json_content['panels']:
        try:
            panel_id = np.where(panel_ids == panel['id'])[0][0]
        except Exception as e:
            print(f"Wrong faces | panel_id: {panel['id']}, {np.where(panel_ids == panel['id'])},{e}")
            continue
        panel_edge_dict[panel_id] = []
        id_info[panel['id']] = []
        for edge_seq in panel['seqEdges']:
            for edge in edge_seq['edges']:
                id_info[panel['id']].append(edge['id'])
                try:
                    edge_id = np.where(edge_ids == edge['id'])[0][0]
                except Exception as e:
                    print(f"Wrong edges |panel_id: {panel['id']}, edge_id: {edge['id']}, {np.where(edge_ids == edge['id'])},{e}")
                    continue
                panel_edge_dict[panel_id].append(int(edge_id))
                panel_edge_dict[panel_id] = sorted(list(set(panel_edge_dict[panel_id])))

    faceEdge_adj = []
    for panel_id in range(len(panel_ids)):
        faceEdge_adj.append(panel_edge_dict[panel_id])

    # final = {'id_info':id_info, 'panel_ids':panel_ids.tolist(), 'edge_ids':edge_ids.tolist(),'faceEdge_adj':faceEdge_adj}

    # with open('test.json', 'w') as f:
    #     json.dump(final, f, indent=4)

    return faceEdge_adj

def face_edge_to_id(faceEdge_adj: dict, panel_ids_, edge_ids):
    adj_uuid_id = {}
    
    face_uuid_id, edge_uuid_id = {}, {}
    for idx, panel_id in enumerate(panel_ids_):
        face_uuid_id[panel_id] = idx
    
    for idx, edge_id in enumerate(edge_ids):
        edge_uuid_id[edge_id] = idx

    for face_id, edge_ids in faceEdge_adj.items():
        adj_uuid_id[face_uuid_id[face_id]] = [edge_uuid_id[edge_id] for edge_id in edge_ids]
    
    return adj_uuid_id


def normalize(
    surf_pnts, edge_pnts, corner_pnts, 
    surf_masks=None, edge_masks=None, corner_masks=None,
    global_offset=None, global_scale=None):
    
    """
    Various levels of normalization 
    """
            
    # Global normalization to -1~1
    if global_offset is None or global_scale is None:
        global_offset, global_scale = _get_scale_offset(surf_pnts, surf_masks)
        assert global_scale != 0, 'scale is zero'

    surfs_wcs, edges_wcs, surfs_ncs, edges_ncs = [], [], [], []

    # Normalize corner
    if corner_masks is None: corner_masks = np.ones_like(corner_pnts[..., :1]).astype(bool)
    corner_wcs = (corner_pnts - global_offset[np.newaxis, :]) / (global_scale * 0.5) * corner_masks.astype(corner_pnts.dtype)

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

    # Normalize edge
    if edge_masks is None: edge_masks = np.ones_like(edge_pnts[..., :1]).astype(bool)
    # print('[NORMALIZE] edge_mask: ', edge_pnts.shape, surf_masks.shape, np.unique(surf_masks)) 
    for edge_idx in range(edge_pnts.shape[0]):
        edge_pnt, edge_mask = edge_pnts[edge_idx], edge_masks[edge_idx]
        edge_pnt_wcs = (edge_pnt - global_offset[np.newaxis, :]) / (global_scale * 0.5)
        edge_pnt_wcs = edge_pnt_wcs * edge_mask.astype(edge_pnt_wcs.dtype)
        edges_wcs.append(edge_pnt_wcs)
        
        local_offset, local_scale = _get_scale_offset(edge_pnt_wcs, edge_mask)
        pnt_ncs = (edge_pnt_wcs - local_offset[np.newaxis, :]) / (local_scale * 0.5)
        pnt_ncs = pnt_ncs * edge_mask.astype(pnt_ncs.dtype)
        edges_ncs.append(pnt_ncs)

    surfs_wcs = np.stack(surfs_wcs)
    surfs_ncs = np.stack(surfs_ncs)
    edges_wcs = np.stack(edges_wcs)
    edges_ncs = np.stack(edges_ncs)

    return surfs_wcs, edges_wcs, surfs_ncs, edges_ncs, corner_wcs, global_offset, global_scale


def process_data(
        data_item: str,
        glctx=None,
        num_samples=64,
        use_uni_norm=True):

    # print('*** Processing data: ', data_item)

    mesh_fp = os.path.join(data_item, os.path.basename(data_item)+'.obj')
    pattern_fp = os.path.join(data_item, 'pattern.json')

    mesh_obj = read_obj(mesh_fp)
    with open(pattern_fp, 'r', encoding='utf-8') as f:
        pattern_spec = json.load(f)

    # surf_uv: panel 3D points [num_panels, num_samples, num_samples, 3]
    if glctx is None: glctx = dr.RasterizeCudaContext()

    panel_ids, panel_cls, surf_pnts, surf_uvs, surf_norms, surf_mask = prepare_surf_data(
        glctx, mesh_obj=mesh_obj, pattern_spec=pattern_spec, reso=num_samples
    )

    # edge && corner
    edge_ids, edge_pnts, edge_uvs, edge_normals, corner_pnts, corner_uvs, corner_normals, faceEdge_adj = prepare_edge_data(
        mesh_obj, pattern_spec, reso=num_samples
    )

    faceEdge_adj = face_edge_to_id(faceEdge_adj, panel_ids, edge_ids)

    surfs_wcs, edges_wcs, surfs_ncs, edges_ncs, corner_wcs, global_offset, global_scale = normalize(
        surf_pnts, edge_pnts, corner_pnts, surf_masks=surf_mask, 
        global_offset=_GLOBAL_OFFSET if use_uni_norm else None, 
        global_scale=_GLOBAL_SCALE if use_uni_norm else None)
    
    surfs_uv_wcs, edges_uv_wcs, surfs_uv_ncs, edges_uv_ncs, corner_uv_wcs, uv_offset, uv_scale = normalize(
        surf_uvs, edge_uvs, corner_uvs, surf_masks=surf_mask, 
        global_offset=_GLOBAL_OFFSET_UV if use_uni_norm else None, 
        global_scale=_GLOBAL_SCALE_UV if use_uni_norm else None
        )
    
    result = {
        'data_fp': data_item,
        # xyz
        'surf_cls': np.array(panel_cls, dtype=np.int32),
        'surf_mask': surf_mask.astype(bool),
        'surf_wcs': surfs_wcs.astype(np.float32),
        'edge_wcs': edges_wcs.astype(np.float32),
        'surf_ncs': surfs_ncs.astype(np.float32),
        'edge_ncs': edges_ncs.astype(np.float32),
        'corner_wcs': corner_wcs.astype(np.float32),
        'global_offset': global_offset.astype(np.float32),
        'global_scale': global_scale,
        # uv
        'surf_uv_wcs': surfs_uv_wcs.astype(np.float32),
        'edge_uv_wcs': edges_uv_wcs.astype(np.float32),
        'surf_uv_ncs': surfs_uv_ncs.astype(np.float32),
        'edge_uv_ncs': edges_uv_ncs.astype(np.float32),
        'corner_uv_wcs': corner_uv_wcs.astype(np.float32),
        'uv_offset': uv_offset.astype(np.float32),
        'uv_scale': uv_scale,
        # normal
        'surf_normals': surf_norms.astype(np.float32),
        'edge_normals': edge_normals.astype(np.float32),
        'corner_normals': corner_normals.astype(np.float32),
        # adj
        'faceEdge_adj': faceEdge_adj
    }
    
    # Calculate bounding boxes
    result['surf_bbox_wcs'] = _get_bbox(result['surf_wcs'], surf_mask)
    result['edge_bbox_wcs'] = _get_bbox(result['edge_wcs'])
    
    # Calculating UV bounding bbox
    result['surf_uv_bbox_wcs'] = _get_bbox(result['surf_uv_wcs'], surf_mask)
    result['edge_uv_bbox_wcs'] = _get_bbox(result['edge_uv_wcs'])
        
    return result


def process_item(data_idx, data_item, args, glctx):
    try:
        os.makedirs(args.output, exist_ok=True)
        output_fp = os.path.join(args.output, '%04d.pkl' % (data_idx))    
        
        result = process_data(
            data_item, glctx=glctx, 
            num_samples=64)
        
        with open(output_fp, 'wb') as f: pickle.dump(result, f)
        return True, data_item
        
    except Exception as e:
        return False, f"{data_item} | [ERROR] {e}"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Draw panel bbox 3D")
    parser.add_argument("-i", "--input", default="./resources/examples",
                        type=str, help="Input directories splited bt comma.")
    parser.add_argument("-o", "--output", default='./resources/examples/processed',
                        type=str, help="Output directory.")
    parser.add_argument("-r", "--range", default=None, type=str, 
                        help="Path to executable.")
    parser.add_argument("--use_uni_norm", action='store_true',
                        help="Whether to apply universal normalization for the output data. Use predefined global scale and offset for the whole dataset if specified otherwise calculate from each data item.")

    args, cfg_cmd = parser.parse_known_args()

    glctx = dr.RasterizeCudaContext()
    print('[DONE] Init renderer.', type(glctx))

    data_root_dirs = args.input.split(',')
    print('Input directories: ', data_root_dirs)
    
    data_items = []
    for idx, data_root in enumerate(data_root_dirs):
        cur_data_items = sorted([os.path.dirname(x) for x in glob(
            os.path.join(data_root, '**', 'pattern.json'), recursive=True)])
        data_items += cur_data_items
        print('[%02d/%02d] Found %d items in %s.'%(idx, len(data_root_dirs), len(cur_data_items), data_root))
    print('Total items: ', len(data_items))

    log_file = os.path.join(args.output, 'app.log')
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            processed = [x.split("\t")[0] for x in lines if x.split("\t")[1].strip() == "1"]
            data_items = [x for x in data_items if x not in processed]

    if args.range is not None:
        begin, end = args.range.split(",")
        begin, end = max(0, int(begin)), min(int(end), len(data_items))
        data_items = data_items[begin:end]
        print("Extracting range: %d %s" % (len(data_items), args.output))

    os.makedirs(args.output, exist_ok=True)

    failed_items = []
    with open(log_file, 'a+') as f:
        with ThreadPoolExecutor(max_workers=8) as executor:  # 可以调整max_workers以改变并行度
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
        for item in failed_items: f.write(item+'\n')