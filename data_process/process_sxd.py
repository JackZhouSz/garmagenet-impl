import json
import os
import sys
import argparse
from glob import glob
from tqdm import tqdm
import pickle
import numpy as np
import torch
import nvdiffrast.torch as dr
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

_AVATAR_BBOX = [
    [-449.00006104,  191.10876465, -178.34872437],      # min
    [447.45980835, 1831.29016113,  174.13575745]       # max
]


def _interpolate_feature_dr(rast, pos, tris, feat):
    """
    Interpolate feature from rasterized image to 3D points.
    
    """
    if rast.dim() == 4 and rast.shape[0] == 1:
        feat, pos = feat[None], pos[None]
    if rast.dim() == 4 and rast.shape[0] == 1:
        feat, pos = feat[None], pos[None]
    out, _ = dr.interpolate(feat, rast, tris)
    out = dr.antialias(out, rast, pos, tris)
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

    start_feat = line_features[belonging_segments, 0, :]
    end_feat = line_features[belonging_segments, 1, :]

    dist_start = query_dist_start[np.arange(
        query_dist_start.shape[0]), belonging_segments]
    dist_end = query_dist_end[np.arange(
        query_dist_end.shape[0]), belonging_segments]
    dist_total = query_dist[np.arange(query_dist.shape[0]), belonging_segments]

    assert np.allclose(dist_total, dist_start + dist_end), "Invalid distance calculation!"

    t_start = dist_start / dist_total
    t_end = dist_end / dist_total

    query_feat = t_start[:, None] * start_feat + t_end[:, None] * end_feat

    return belonging_segments, query_feat


def _face_edge_to_id(faceEdge_adj: dict, panel_ids_, edge_ids):
    adj_uuid_id = {}
    
    face_uuid_id, edge_uuid_id = {}, {}
    for idx, panel_id in enumerate(panel_ids_):
        face_uuid_id[panel_id] = idx
    
    for idx, edge_id in enumerate(edge_ids):
        edge_uuid_id[edge_id] = idx

    for face_id, edge_ids in faceEdge_adj.items():
        adj_uuid_id[face_uuid_id[face_id]] = [edge_uuid_id[edge_id] for edge_id in edge_ids]
    
    return adj_uuid_id


def _normalize_pnts(surf_pnts, edge_pnts, corner_pnts):
    """
    Various levels of normalization for 3D points
    """
    
    p_dim = surf_pnts[0].shape[-1]
    
    # Global normalization to -1~1
    total_points = np.array(surf_pnts).reshape(-1, p_dim)
    
    min_vals = np.min(total_points, axis=0)
    max_vals = np.max(total_points, axis=0)
    
    offset = min_vals + (max_vals - min_vals)/2 
    scale = max(max_vals - min_vals)
    
    assert scale != 0, 'scale is zero'

    surfs_wcs, edges_wcs, surfs_ncs, edges_ncs = [],[],[],[]

    # Normalize corner 
    corner_wcs = (corner_pnts - offset[np.newaxis,:]) / (scale * 0.5)

    # Normalize surface
    for surf_pnt in surf_pnts:    
        # Normalize CAD to WCS
        surf_pnt_wcs = (surf_pnt - offset[np.newaxis,np.newaxis,:]) / (scale * 0.5)
        surfs_wcs.append(surf_pnt_wcs)
        # Normalize Surface to NCS
        min_vals = np.min(surf_pnt_wcs.reshape(-1, p_dim), axis=0)
        max_vals = np.max(surf_pnt_wcs.reshape(-1, p_dim), axis=0)
        
        local_offset = min_vals + (max_vals - min_vals)/2 
        local_scale = max(max_vals - min_vals)
        pnt_ncs = (surf_pnt_wcs - local_offset[np.newaxis,np.newaxis,:]) / (local_scale * 0.5)
        surfs_ncs.append(pnt_ncs)
       
    # Normalize edge
    for edge_pnt in edge_pnts:    
        # Normalize CAD to WCS
        edge_pnt_wcs = (edge_pnt - offset[np.newaxis,:]) / (scale * 0.5)
        edges_wcs.append(edge_pnt_wcs)
        # Normalize Edge to NCS
        min_vals = np.min(edge_pnt_wcs.reshape(-1,p_dim), axis=0)
        max_vals = np.max(edge_pnt_wcs.reshape(-1,p_dim), axis=0)
        local_offset = min_vals + (max_vals - min_vals)/2 
        local_scale = max(max_vals - min_vals)
        pnt_ncs = (edge_pnt_wcs - local_offset) / (local_scale * 0.5)
        edges_ncs.append(pnt_ncs)
        assert local_scale != 0, 'scale is zero'

    surfs_wcs = np.stack(surfs_wcs).astype(np.float32)
    surfs_ncs = np.stack(surfs_ncs).astype(np.float32)
    edges_wcs = np.stack(edges_wcs).astype(np.float32)
    edges_ncs = np.stack(edges_ncs).astype(np.float32)
    corner_wcs = corner_wcs.astype(np.float32)
    offset = offset
    scale = scale
    
    return surfs_wcs, edges_wcs, surfs_ncs, edges_ncs, corner_wcs, offset, scale


def prepare_edge_data(
    mesh_obj,
    pattern_spec,
    reso=64,
    global_bbox=None
):

    verts = mesh_obj.points
    uv = mesh_obj.point_data['obj:vt']

    # global normalization to (-1, 1) for the whole
    if global_bbox is not None:
        global_bbox = np.array(global_bbox, dtype=np.float32)
        global_offset = (global_bbox[0, :] + global_bbox[1, :]) / 2.0
        global_scale = np.max(global_bbox[1, :] - global_bbox[0, :])
        verts = (verts - global_offset) / (global_scale * 0.5)

    panel_data = dict([(x['id'], x) for x in pattern_spec['panels']])

    edge_ids, edge_pnts, edge_uvs = [], [], []
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

        verts_colors = _PANEL_COLORS[_PANEL_CLS.index(
            panel_data[panel_id]['label'].strip())+1, :3][None]
        verts_colors = np.ones_like(boundary_pos[:, 0, :]) * verts_colors

        for seg_edge in panel_spec['seqEdges']:
            
            if seg_edge['type'] != 3 or seg_edge['circleType'] != 0: continue
            
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

                if within_bbox.sum() == 0:
                    continue   # TODO:内部线问题？（待验证）

                _, sample_pos = _project_curve_to_line_segments(
                    edge_samples, boundary_uv[within_bbox, :, :], boundary_pos[within_bbox, :, :])

                edge_ids.append(edge['id'])
                edge_pnts.append(sample_pos)
                edge_uvs.append(edge_samples + panel_center[..., :2][None])
                faceEdge_adj[panel_id].append(edge['id'])
            
    edge_pnts = np.stack(edge_pnts, axis=0)
    edge_uvs = np.stack(edge_uvs, axis=0)
    
    print('*** edge_pnts: ', edge_pnts.shape, edge_pnts.reshape(-1, 3).min(0), edge_pnts.reshape(-1, 3).max(0))
    print('*** edge_uvs: ', edge_uvs.shape, edge_uvs.reshape(-1, 2).min(0), edge_uvs.reshape(-1, 2).max(0))

    corner_pnts = edge_pnts[:, [0, -1], :]
    corner_uvs = edge_uvs[:, [0, -1], :]

    return edge_ids, edge_pnts, edge_uvs, corner_pnts, corner_uvs, faceEdge_adj


def prepare_surf_data(
    glctx,
    mesh_obj,
    pattern_spec,
    reso=64,
    global_bbox=None
):
    
    _ = pattern_spec

    verts = torch.from_numpy(mesh_obj.points).to(torch.float32).to('cuda')
    uv = torch.from_numpy(mesh_obj.point_data['obj:vt']).to(torch.float32).to('cuda')

    if global_bbox is not None:
        global_bbox = torch.tensor(global_bbox).to(torch.float32).to ('cuda')
        global_offset = (global_bbox[0, :] + global_bbox[1, :]) / 2.0
        global_scale = torch.max(global_bbox[1, :] - global_bbox[0, :])
        verts = (verts - global_offset) / (global_scale * 0.5)

    panel_ids = []
    uv_local = uv.clone()               # projected pixel coordinate for each vertex
    panel_bbox2ds = []

    tris = []                           # all triangles
    tri_ranges = []                     # triangle range for each panel,
    start_idx = 0

    for idx, panel_id in enumerate(mesh_obj.field_data['obj:group_tags']):
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
        
        panel_offset = (panel_bbox2d[0] + panel_bbox2d[1]) / 2.0
        panel_scale = (panel_bbox2d[1] - panel_bbox2d[0]) / 2.0
                
        uv_local[vert_ids, :] = (uv_local[vert_ids, :] - panel_offset) / (panel_scale + 1e-6)      # normalize to [-1, 1]
        
        panel_bbox2ds.append(panel_bbox2d)

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

    return surf_pnts, surf_uvs, panel_ids


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


def process_data(
        data_item: str,
        glctx=None,
        num_samples=64,
        use_global_bbox=False,
        output_dir=None):

    print('*** Processing data: ', data_item)

    mesh_fp = os.path.join(data_item, os.path.basename(data_item)+'.obj')
    pattern_fp = os.path.join(data_item, 'pattern.json')

    mesh_obj = read_obj(mesh_fp)
    with open(pattern_fp, 'r', encoding='utf-8') as f:
        pattern_spec = json.load(f)

    # surf_uv: panel 3D points [num_panels, num_samples, num_samples, 3]
    if glctx is None:
        glctx = dr.RasterizeCudaContext()

    surf_pnts, surf_uvs, panel_ids = prepare_surf_data(
        glctx, mesh_obj=mesh_obj, pattern_spec=pattern_spec, reso=num_samples,
        global_bbox=_AVATAR_BBOX if use_global_bbox else None
    )

    # edge && corner
    edge_ids, edge_pnts, edge_uvs, corner_pnts, corner_uvs, faceEdge_adj = prepare_edge_data(
        mesh_obj, pattern_spec, reso=num_samples,
        global_bbox=_AVATAR_BBOX if use_global_bbox else None
    )

    # faceEdge_adj = face_edge_adj(pattern_spec, panel_ids_, edge_ids)
    faceEdge_adj = _face_edge_to_id(faceEdge_adj, panel_ids, edge_ids)
        
    surfs_wcs, edges_wcs, surfs_ncs, edges_ncs, corner_wcs, global_offset, global_scale = _normalize_pnts(
        surf_pnts, edge_pnts, corner_pnts)
    # surf_bbox_wcs = ...
    # edge_bbox_wcs = ...
    # corner_unique = ...
    
    surf_uvs_wcs, edge_uvs_wcs, surf_uvs_ncs, edge_uvs_ncs, corner_uv_wcs, uv_offset, uv_scale = _normalize_pnts(
        surf_uvs, edge_uvs, corner_uvs)
    # surf_uv_bbox_wcs = ...
    # edge_uv_bbox_wcs = ...
    # corner_uv_unique = ...
    
    result = {
        'surf_wcs': np.concatenate([surfs_wcs, surf_uvs_wcs], axis=-1).astype(np.float32),          # (num_panels, num_samples, num_samples, 5) -> x, y, z, u, v
        'edge_wcs': np.concatenate([edges_wcs, edge_uvs_wcs], axis=-1).astype(np.float32),          # (num_edges,  num_samples, 5) -> x, y, z, u, v
        'surf_ncs': np.concatenate([surfs_ncs, surf_uvs_ncs], axis=-1).astype(np.float32),          # normalized surf_wcs
        'edge_ncs': np.concatenate([edges_ncs, edge_uvs_ncs], axis=-1).astype(np.float32),          # normalized edge_wcs
        'corner_wcs': np.concatenate([corner_wcs, corner_uv_wcs], axis=-1).astype(np.float32),      # (num_edges, 2, 5) -> x, y, z, u, v of start/end point for each edge 
        'edgeFace_adj': None,
        'edgeCorner_adj':None,
        'faceEdge_adj': faceEdge_adj,
        'surf_bbox_wcs': None,
        'edge_bbox_wcs': None,
        'corner_unique': None,
        'global_offset': np.concatenate([global_offset, uv_offset], axis=-1).astype(np.float32),    # array of shape (5,) (offset_x, offset_y, offset_z, offset_u, offset_v)
        'global_scale': np.array([global_scale, uv_scale], dtype=np.float32),                       # array of shape (2,) (scale_xyz, scale_uv)
    }

    if output_dir:
        with open(os.path.join(output_dir, '%s.pkl' % (os.path.basename(output_dir))), 'wb') as f:
            pickle.dump(result, f)

    return result


def process_item(data_idx, data_item, args, glctx):
    # try:
    output_dir = os.path.join(args.output, '%04d' % (data_idx))
    out_folder = data_item.replace(args.input, args.output)
    os.makedirs(out_folder, exist_ok=True)
    result = process_data(
        data_item, glctx=glctx, num_samples=64, 
        use_global_bbox=False, output_dir=out_folder)
    return None, data_item
    # except Exception as e:
    #     return str(e), data_item




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Draw panel bbox 3D")
    parser.add_argument("-i", "--input", default="./resources/examples",
                        type=str, help="Input directory.")
    parser.add_argument("-o", "--output", default='./resources/examples/processed',
                        type=str, help="Output directory.")

    parser.add_argument("--render_uv", action='store_true',
                        help="Render 2D UV image.")
    parser.add_argument("--render_seg", action='store_true',
                        default=True, help="Render 2D UV image.")

    args, cfg_cmd = parser.parse_known_args()

    glctx = dr.RasterizeCudaContext()
    print('[DONE] Init renderer.', type(glctx))

    data_root = args.input
    data_items = sorted([os.path.dirname(x) for x in glob(
        os.path.join(data_root, '**', 'pattern.json'), recursive=True)])
    
    print('Total num items: ', len(data_items))
    
    wrong_files = 'wrong_edge_face_normalize_adj.txt'
    failed_items = []

    with open(wrong_files, 'a+') as wrong_fp:
        with ThreadPoolExecutor(max_workers=1) as executor:  # 可以调整max_workers以改变并行度
            futures = {executor.submit(process_item, data_idx, data_item, args, glctx): data_item for data_idx, data_item in enumerate(data_items)}

            for future in tqdm(as_completed(futures), total=len(futures)):
                error, data_item = future.result()
                if error:
                    print('[ERROR] Failed to process data:', data_item)
                    failed_items.append(data_item)
                    wrong_fp.write(f"{data_item}, {error}\n")