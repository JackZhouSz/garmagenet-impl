import math
import os
import random
from glob import glob

import numpy as np
import pickle

from plotly.subplots import make_subplots
import plotly.graph_objects as go


def min_max_normalize(points, bbox):
    # 计算每一列的最小值和最大值
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # 移动到中心

    bbox_normalized = bbox.reshape(-1, 3)
    points = points - (max_vals + min_vals)/2
    bbox_normalized = bbox_normalized - (max_vals + min_vals)/2
    a=1
    # 计算每一列的最小值和最大值
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # 防止除零
    ranges = np.array([1]*points.shape[-1]) * np.max(max_vals - min_vals)
    ranges[ranges == 0] = 1

    # 归一化到 [-0.5, 0.5]
    normalized_points = points/ranges
    bbox_normalized = bbox_normalized/ranges
    bbox_normalized = bbox_normalized.reshape(-1, 6)

    return normalized_points, bbox_normalized

import gc

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
    "木耳边": {"alias": "木耳边", "color": "#A8D4D2"},
    "袖笼拼条": {"alias": "袖笼拼条", "color": "#696969"},
    "荷叶边": {"alias": "荷叶边", "color": "#A8D4D2"},
    "绑带": {"alias": "绑带", "color": "#696969"}
}

_PANEL_CLS = [
    '帽', '领', '肩', '袖片', '袖口', '衣身前中', '衣身后中', '衣身侧', '底摆', '腰头', '裙前中', '裙后中', '裙侧', '橡筋', '木耳边', '袖笼拼条', '荷叶边', '绑带']


def _create_bounding_box_lines(min_point, max_point, color, width=2, opacity=1):
    
    # Create the 12 lines of the bounding box
    x_lines = []
    y_lines = []
    z_lines = []
    
    # List of all 8 corners of the box
    x0, y0, z0 = min_point
    x1, y1, z1 = max_point

    corners = np.array([
        [x0, y0, z0],  # 0
        [x1, y0, z0],  # 1
        [x1, y1, z0],  # 2
        [x0, y1, z0],  # 3
        [x0, y0, z1],  # 4
        [x1, y0, z1],  # 5
        [x1, y1, z1],  # 6
        [x0, y1, z1]   # 7
    ])

    # Pairs of corners between which to draw lines
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    for edge in edges:
        start = corners[edge[0]]
        end = corners[edge[1]]
        x_lines.extend([start[0], end[0], None])  # None to break the line
        y_lines.extend([start[1], end[1], None])
        z_lines.extend([start[2], end[2], None])

    line_trace = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        line=dict(
            color=color,
            width=width
        ),
        opacity=opacity,
        showlegend=False
    )
    return line_trace


def _create_bounding_box_mesh(min_point, max_point, color, opacity=0.2):
    # List of all 8 corners of the box
    x0, y0, z0 = min_point
    x1, y1, z1 = max_point

    corners = np.array([
        [x0, y0, z0],  # 0
        [x1, y0, z0],  # 1
        [x1, y1, z0],  # 2
        [x0, y1, z0],  # 3
        [x0, y0, z1],  # 4
        [x1, y0, z1],  # 5
        [x1, y1, z1],  # 6
        [x0, y1, z1]   # 7
    ])

    # Define the triangles composing the surfaces of the box
    # Each face is composed of two triangles
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 1, 5], [0, 5, 4],  # Side face
        [1, 2, 6], [1, 6, 5],  # Side face
        [2, 3, 7], [2, 7, 6],  # Side face
        [3, 0, 4], [3, 4, 7]   # Side face
    ])

    x = corners[:, 0]
    y = corners[:, 1]
    z = corners[:, 2]

    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
        name='Bounding Box',
        showlegend=False,
        flatshading=True
    )

    return mesh


def draw_bbox_geometry(
    bboxes,
    bbox_colors,
    points=None,
    point_masks=None,
    point_colors=None,
    num_point_samples=1000,
    show_bbox=True,
    show_num=False,
    all_bboxes=None,
    export_data_config=None,
    output_fp = None,
    visatt_dict={}
):
    annotations = []

    fig = go.Figure()

    for idx in range(len(bboxes)):
        # visuzlize point clouds if given
        if points is not None:
            cur_points, cur_points_mask = points[idx].reshape(-1, 3), point_masks[idx].reshape(-1)
            cur_points = cur_points[cur_points_mask, :]
            if cur_points.shape[0] > num_point_samples:
                rand_idx = np.random.choice(cur_points.shape[0], num_point_samples, replace=False)

                # rand_idx = np.linspace(0, cur_points.shape[0] - 1, num_point_samples, dtype=int)

                cur_points = cur_points[rand_idx, :]
            fig.add_trace(go.Scatter3d(
                x=cur_points[:, 0],
                y=cur_points[:, 1],
                z=cur_points[:, 2],
                mode='markers',
                marker=dict(size=2, color=point_colors[idx]),
                name=f'Point Cloud {idx+1}'
            ))

        # Add the bounding box lines
        min_point, max_point = bboxes[idx, :3], bboxes[idx, 3:]
        bbox_lines = _create_bounding_box_lines(min_point, max_point, color=bbox_colors[idx])
        fig.add_trace(bbox_lines)
        # Add the bounding box surfaces with transparency
        bbox_mesh = _create_bounding_box_mesh(min_point, max_point, color=bbox_colors[idx], opacity=0.05)
        fig.add_trace(bbox_mesh)

    if show_bbox:
        for idx in range(len(bboxes)):
            # Add the bounding box lines
            min_point, max_point = bboxes[idx, :3], bboxes[idx, 3:]
            bbox_lines = _create_bounding_box_lines(min_point, max_point, color=bbox_colors[idx],
                                                    width=visatt_dict.get("bboxline_width", 2),
                                                    opacity=visatt_dict.get("bboxline_opacity", 1))
            fig.add_trace(bbox_lines)
            # Add the bounding box surfaces with transparency
            bbox_mesh = _create_bounding_box_mesh(min_point, max_point, color=bbox_colors[idx],
                                                  opacity=visatt_dict.get("bboxmesh_opacity", 0.2))
            fig.add_trace(bbox_mesh)

            if show_num:
                # Add annotation (always on top)
                center = (min_point + max_point) / 2
                annotations.append(dict(
                    showarrow=False,
                    x=center[0],
                    y=center[1],
                    z=center[2],
                    text=f'<b>{idx}</b>',
                    font=dict(color='black', size=14),
                    xanchor='left',
                    yanchor='bottom',
                    bgcolor='rgba(255,255,255,0.7)',  # Optional: white semi-transparent background
                    bordercolor='black',
                    borderwidth=1,
                    opacity=1
                ))


    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=visatt_dict.get("camera_eye_z", 2))
    )

    # 计算所有 bounding boxes 的全局范围
    if all_bboxes is None:
        all_mins = np.min(bboxes[:, :3], axis=0)
        all_maxs = np.max(bboxes[:, 3:], axis=0)
    else:
        all_mins = np.min(all_bboxes[:, :3], axis=0)
        all_maxs = np.max(all_bboxes[:, 3:], axis=0)
    # 计算中心点和最大跨度以保持 aspect 比例
    center = (all_mins + all_maxs) / 2
    max_range = np.max(all_maxs - all_mins) / 2

    x_range = [center[0] - max_range, center[0] + max_range]
    y_range = [center[1] - max_range, center[1] + max_range]
    z_range = [center[2] - max_range, center[2] + max_range]

    # Update layout
    fig.update_layout(
        scene=dict(
            annotations=annotations,
            camera=camera,
            xaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
                title='',
                range=x_range
            ),
            yaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
                title='',
                range=y_range
            ),
            zaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
                title='',
                range=z_range
            ),
            aspectmode='cube',  # Keep the aspect ratio of data
        ),
        width=800,
        height=800,
        margin=dict(r=0, l=0, b=0, t=0),
        showlegend=False,
        title=dict(text="", automargin=True),
        scene_camera=camera,
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
        # scene_dragmode = False  # 禁止自动调整场景大小
    )

    # 显示图形
    if export_data_config is None and output_fp is None:
        fig.show("browser")

    if output_fp is not None and export_data_config is None:
        fig.write_image(output_fp, width=1080, height=1080, scale=2)

    # 绕着人旋转一圈，并保存图片 --------------------------------------------------------------------------------------------
    if export_data_config:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='X Axis',
                    range=[min_val, max_val],
                    showgrid = False,  # 隐藏网格线
                    showticklabels = False,  # 隐藏刻度标签
                    zeroline = False,
                    visible=False  # 隐藏以上所有，和其它的
                ),
                yaxis=dict(
                    title='Y Axis',
                    range=[min_val, max_val],
                    showgrid = False,  # 隐藏网格线
                    showticklabels = False,  # 隐藏刻度标签
                    zeroline = False,
                    visible=False  # 隐藏以上所有，和其它的
                ),
                zaxis=dict(
                    title='Z Axis',
                    range=[min_val, max_val],
                    showgrid = False,  # 隐藏网格线
                    showticklabels = False,  # 隐藏刻度标签
                    zeroline=False,
                    visible=False  # 隐藏以上所有，和其它的
                ),
                aspectmode='cube'  # 确保各个轴的比例相同
            ),
            title=""
        )
        camera = dict(
            up=dict(x=0, y=1, z=0),
            center=dict(
                x=0,
                y=0,
                z=0),
            eye=dict(x=0, y=0, z=1.5)
        )
        fig.update_layout(scene_camera=camera)

        pic_num = export_data_config["pic_num"]
        export_path = export_data_config["export_path"]
        os.makedirs(export_data_config["export_path"], exist_ok=True)
        for i in range(export_data_config["pic_num"]+1):
            img_path = os.path.join(export_path, f"{i}".zfill(3)+".png")

            math.sin(pic_num)
            eye_z = math.cos(2 * math.pi * i / pic_num) * 1.5
            eye_x = math.sin(2 * math.pi * i / pic_num) * 1.5
            camera = dict(
                up=dict(x=0, y=1, z=0),
                center=dict(
                    x=0,
                    y=0,
                    z=0),
                eye=dict(x=eye_x, y=0, z=eye_z)
            )
            # Update layout
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        visible=False,
                        showbackground=False,
                        showgrid=False,
                        showline=False,
                        showticklabels=False,
                        title=''
                    ),
                    yaxis=dict(
                        visible=False,
                        showbackground=False,
                        showgrid=False,
                        showline=False,
                        showticklabels=False,
                        title=''
                    ),
                    zaxis=dict(
                        visible=False,
                        showbackground=False,
                        showgrid=False,
                        showline=False,
                        showticklabels=False,
                        title=''
                    ),
                    aspectmode='cube'  # Keep the aspect ratio of data
                ),
                width=800,
                height=800,
                margin=dict(r=0, l=0, b=0, t=0),
                showlegend=False,
                title=dict(text="", automargin=True),
                scene_camera=camera,
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
                paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper background
            )
            fig.write_image(img_path, width=1920, height=1920, scale=2)
            # fig.show()


# def draw_bbox_geometry2D(
#     bboxes,
#     bbox_colors,
#     points=None,
#     point_masks=None,
#     point_colors=None,
#     num_point_samples=1000,
#     output_fp = None
# ):
#     annotations = []
#
#     fig = go.Figure()
#
#     # 2D ===
#     for idx in range(len(bboxes[1])):
#         # 可视化点云
#         if points is not None and points[1] is not None:
#             cur_points, cur_points_mask = points[1][idx].reshape(-1, 3), point_masks[idx].reshape(-1)
#             cur_points = cur_points[cur_points_mask, :]
#             if cur_points.shape[0] > num_point_samples:
#                 rand_idx = np.random.choice(cur_points.shape[0], num_point_samples, replace=False)
#                 cur_points = cur_points[rand_idx, :]
#             trace = go.Scatter3d(
#                 x=cur_points[:, 0],
#                 y=cur_points[:, 1],
#                 z=cur_points[:, 2],
#                 mode='markers',
#                 marker=dict(size=2, color=point_colors[idx]),
#                 name=f'Point Cloud {idx + 1}',
#                 showlegend=False
#             )
#             fig.add_trace(trace)
#
#         # 添加包围盒线条和表面
#         min_point, max_point = bboxes[1][idx, :3], bboxes[1][idx, 3:]
#         bbox_lines = _create_bounding_box_lines(min_point, max_point, color=bbox_colors[idx])
#         fig.add_trace(bbox_lines)
#         bbox_mesh = _create_bounding_box_mesh(min_point, max_point, color=bbox_colors[idx], opacity=0.05)
#         fig.add_trace(bbox_mesh)
#
#     camera = dict(
#         up=dict(x=0, y=1, z=0),
#         center=dict(x=0, y=0, z=0),
#         eye=dict(x=0, y=0, z=2.5)
#     )
#
#     # Update layout
#     fig.update_layout(
#         scene=dict(
#             annotations=annotations,
#             xaxis=dict(
#                 visible=False,
#                 showbackground=False,
#                 showgrid=False,
#                 showline=False,
#                 showticklabels=False,
#                 title=''
#             ),
#             yaxis=dict(
#                 visible=False,
#                 showbackground=False,
#                 showgrid=False,
#                 showline=False,
#                 showticklabels=False,
#                 title=''
#             ),
#             zaxis=dict(
#                 visible=False,
#                 showbackground=False,
#                 showgrid=False,
#                 showline=False,
#                 showticklabels=False,
#                 title=''
#             ),
#             aspectmode='data'  # Keep the aspect ratio of data
#         ),
#         width=800,
#         height=800,
#         margin=dict(r=0, l=0, b=0, t=0),
#         showlegend=False,
#         title=dict(text="", automargin=True),
#         scene_camera=camera,
#         plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
#         paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper background
#     )
#
#     # 显示图形
#     if  output_fp is None:
#         fig.show("browser")
#
#     if output_fp is not None:
#         fig.write_image(output_fp, width=1920, height=1920, scale=2)


def draw_bbox_geometry_3D2D(
    bboxes,
    bbox_colors,
    points=None,
    point_masks=None,
    point_colors=None,
    num_point_samples=1000,
    title='',
    output_fp=None,
    show_num=False,
    visatt_dict = {}
):
    """
    Args:
        bboxes:
        bbox_colors:
        points:
        point_masks:
        point_colors:
        num_point_samples:
        title:
        output_fp:
        show_num:
        fig_show:

    Returns:

    Usage Example:
        fig = draw_bbox_geometry_3D2D(
        bboxes=[data["surf_bbox_wcs"],surf_uv_bbox_wcs],
        bbox_colors=colors,
        points=[data["surf_wcs"], points_uv],
        point_masks=data["surf_mask"],
        point_colors=colors,
        num_point_samples=1000,
        title=f"{os.path.basename(data_path)}: {data['caption']}",
        # output_fp=output_fp.replace('.pkl', '_pointcloud.png'),
        show_num=True,
        fig_show="browser"
        )
    """
    # 创建两个子图，两个都是3D视图
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        horizontal_spacing=0.02,
        subplot_titles=("View 1", "View 2")
    )

    # 3D ===
    # normalize ===
    if points and points[0] is not None:
        cur_points_list = []
        for idx in range(len(bboxes[0])):
            # visuzlize point clouds if given
            cur_points, cur_points_mask = points[0][idx].reshape(-1, 3), point_masks[idx].reshape(-1)
            cur_points = cur_points[cur_points_mask, :]
            cur_points_list.append(cur_points)

        pcs_num_cumsum = np.cumsum([len(n) for n in cur_points_list])
        cur_points_normalized, bboxes[0] = min_max_normalize(np.concatenate(cur_points_list), bboxes[0])

        for i in range(len(pcs_num_cumsum)):
            if i == 0:
                st = 0
            else:
                st = pcs_num_cumsum[i - 1]
            ed = pcs_num_cumsum[i]
            cur_points_list[i] = cur_points_normalized[st:ed]

        for idx in range(len(bboxes[0])):
            # visuzlize point clouds if given
            cur_points, cur_points_mask = points[0][idx].reshape(-1, 3), point_masks[idx].reshape(-1)
            points[0][idx][cur_points_mask, :] = cur_points_list[idx]
    # END normalize ===

    # 可视化点云
    traces1 = []
    annotations1 = []
    for idx in range(len(bboxes[0])):
        # 可视化点云
        if points and points[0] is not None:
            cur_points, cur_points_mask = points[0][idx].reshape(-1, 3), point_masks[idx].reshape(-1)
            cur_points = cur_points[cur_points_mask, :]
            if cur_points.shape[0] > num_point_samples:
                rand_idx = np.random.choice(cur_points.shape[0], num_point_samples, replace=False)
                cur_points = cur_points[rand_idx, :]
            trace = go.Scatter3d(
                x=cur_points[:, 0],
                y=cur_points[:, 1],
                z=cur_points[:, 2],
                mode='markers',
                marker=dict(size=2, color=point_colors[idx]),
                name=f'Point Cloud {idx+1}',
                showlegend=False,
                opacity=visatt_dict.get("point_opacity", 1)
            )
            traces1.append(trace)

        # 添加包围盒线条和表面
        min_point, max_point = bboxes[0][idx, :3], bboxes[0][idx, 3:]
        bbox_lines = _create_bounding_box_lines(min_point, max_point, color=bbox_colors[idx],
                                                width=visatt_dict.get("bboxline_width", 2),
                                                opacity=visatt_dict.get("bboxline_opacity", 1))
        bbox_mesh = _create_bounding_box_mesh(min_point, max_point, color=bbox_colors[idx],
                                              opacity=visatt_dict.get("bboxmesh_opacity", 0.2))
        traces1.extend([bbox_lines, bbox_mesh])

        if show_num:
            center = (min_point + max_point) / 2
            annotations1.append(dict(
                showarrow=False,
                x=center[0], y=center[1], z=center[2],
                text=f'<b>{idx}</b>',
                font=dict(color='black', size=14),
                xanchor='left',
                yanchor='bottom',
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='black',
                borderwidth=1,
                opacity=1
            ))

    # 把所有 traces 添加到两个 scene
    for trace in traces1:
        fig.add_trace(trace, row=1, col=1)
        # fig.add_trace(trace, row=1, col=2)

    # 2D ===
    # # normalize ===
    # cur_points_list = []
    # for idx in range(len(bboxes[1])):
    #     # visuzlize point clouds if given
    #     cur_points, cur_points_mask = points[1][idx].reshape(-1, 3), point_masks[idx].reshape(-1)
    #     cur_points = cur_points[cur_points_mask, :]
    #     cur_points_list.append(cur_points)
    #
    # pcs_num_cumsum = np.cumsum([len(n) for n in cur_points_list])
    # cur_points_normalized, bboxes[1] = min_max_normalize(np.concatenate(cur_points_list), bboxes[1])
    #
    # for i in range(len(pcs_num_cumsum)):
    #     if i == 0:
    #         st = 0
    #     else:
    #         st = pcs_num_cumsum[i - 1]
    #     ed = pcs_num_cumsum[i]
    #     cur_points_list[i] = cur_points_normalized[st:ed]
    #
    # for idx in range(len(bboxes[1])):
    #     # visuzlize point clouds if given
    #     cur_points, cur_points_mask = points[1][idx].reshape(-1, 3), point_masks[idx].reshape(-1)
    #     points[1][idx][cur_points_mask, :] = cur_points_list[idx]
    # # END normalize ===

    traces2 = []
    annotations2 = []
    for idx in range(len(bboxes[1])):
        # 可视化点云
        if points is not None and points[1] is not None:
            cur_points, cur_points_mask = points[1][idx].reshape(-1, 3), point_masks[idx].reshape(-1)
            cur_points = cur_points[cur_points_mask, :]
            if cur_points.shape[0] > num_point_samples:
                rand_idx = np.random.choice(cur_points.shape[0], num_point_samples, replace=False)
                cur_points = cur_points[rand_idx, :]
            trace = go.Scatter3d(
                x=cur_points[:, 0],
                y=cur_points[:, 1],
                z=cur_points[:, 2],
                mode='markers',
                marker=dict(size=2, color=point_colors[idx]),
                name=f'Point Cloud {idx+1}',
                showlegend=False,
                opacity=visatt_dict.get("point_opacity", 1)
            )
            traces2.append(trace)

        # 添加包围盒线条和表面
        min_point, max_point = bboxes[1][idx, :3], bboxes[1][idx, 3:]
        bbox_lines = _create_bounding_box_lines(min_point, max_point, color=bbox_colors[idx],
                                                width=visatt_dict.get("bboxline_width", 2),
                                                opacity=visatt_dict.get("bboxline_opacity", 1))
        bbox_mesh = _create_bounding_box_mesh(min_point, max_point, color=bbox_colors[idx],
                                              opacity=visatt_dict.get("bboxmesh_opacity", 0.2))
        traces2.extend([bbox_lines, bbox_mesh])

        if show_num:
            center = (min_point + max_point) / 2
            annotations2.append(dict(
                showarrow=False,
                x=center[0], y=center[1], z=center[2],
                text=f'<b>{idx}</b>',
                font=dict(color='black', size=14),
                xanchor='left',
                yanchor='bottom',
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='black',
                borderwidth=1,
                opacity=1
            ))

    # 把所有 traces 添加到两个 scene
    for trace in traces2:
        # fig.add_trace(trace, row=1, col=1)
        fig.add_trace(trace, row=1, col=2)


    # 相机设置
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=2.5)
    )
    camera2 = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=-1, y=0, z=0),
        eye=dict(x=-1, y=0, z=2.5)
    )

    # min_val_scene1 = np.min(bboxes[0])
    # max_val_scene1 = np.max(bboxes[0])
    # min_val_scene2 = np.min(bboxes[1][:, [0, 1, 3, 4]])
    # max_val_scene2 = np.max(bboxes[1][:, [0, 1, 3, 4]])

    min_val_scene1 = -1
    max_val_scene1 = 1
    min_val_scene2 = -1
    max_val_scene2 = 1

    # 更新 layout
    fig.update_layout(
        scene=dict(
            camera=camera,
            annotations=annotations1,
            aspectmode='cube',
            xaxis=dict(
                visible=False,
                range=[min_val_scene1, max_val_scene1]
            ),
            yaxis=dict(
                visible=False,
                range=[min_val_scene1, max_val_scene1]
            ),
            zaxis=dict(
                visible=False,
                range=[min_val_scene1, max_val_scene1]
            ),
        ),
        scene2=dict(
            camera=camera2,
            annotations=annotations2,
            aspectmode='cube',
            xaxis=dict(
                visible=False,
                range=[min_val_scene2, max_val_scene2]
            ),
            yaxis=dict(
                visible=False,
                range=[min_val_scene2, max_val_scene2]
            ),
            zaxis=dict(
                visible=False,
                range=[min_val_scene2, max_val_scene2]
            ),
        ),
        # scene_camera=camera,
        # scene2_camera=camera2,
        width=1600,
        height=800,
        margin=dict(r=0, l=0, b=0, t=40),
        showlegend=False,
        title=dict(text=title, automargin=True),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # 显示图形
    if output_fp is None:
        fig.show("browser")

    elif output_fp is not None:
        fig.write_image(output_fp, width=1920, height=960, scale=2)

    return fig
