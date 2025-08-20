"""
一个比较原始但是能跑通的版本，用于批量生成数据
"""

import os

import numpy as np
from PIL import Image
from matplotlib.colors import to_rgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def _pad_arr(arr, pad_size=10, pad_value=0):
    return np.pad(
        arr,
        ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),   # pad size to each dimension, require tensor to have size (H,W, C)
        mode='constant',
        constant_values=pad_value)


def _create_bounding_box_lines(min_point, max_point, color):
    
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
        line=dict(color=color, width=2),
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
    title='',
    output_fp=None,
    show_num=False,
    fig_show=None
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
        eye=dict(x=-1.0, y=0, z=2.5)
    )

    # Update layout
    fig.update_layout(
        scene=dict(
            annotations=annotations,
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
            aspectmode='data'  # Keep the aspect ratio of data
        ),
        width=800,
        height=800,
        margin=dict(r=0, l=0, b=0, t=0),
        showlegend=False,
        title=dict(text=title, automargin=True),
        scene_camera=camera,
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper background
    )
    print(output_fp)
    if output_fp is not None: fig.write_image(output_fp, format='png', engine='kaleido')
    else: fig.show(fig_show)

    if output_fp is not None and fig_show is not None:
        fig.show(fig_show)

def draw_geometry(surf_pos, surf_ncs):
    pass




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
    fig_show=None,
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
                showlegend=False
            )
            traces1.append(trace)

        # 添加包围盒线条和表面
        min_point, max_point = bboxes[0][idx, :3], bboxes[0][idx, 3:]
        bbox_lines = _create_bounding_box_lines(min_point, max_point, color=bbox_colors[idx])
        bbox_mesh = _create_bounding_box_mesh(min_point, max_point, color=bbox_colors[idx], opacity=0.05)
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
                showlegend=False
            )
            traces2.append(trace)

        # 添加包围盒线条和表面
        min_point, max_point = bboxes[1][idx, :3], bboxes[1][idx, 3:]
        bbox_lines = _create_bounding_box_lines(min_point, max_point, color=bbox_colors[idx])
        bbox_mesh = _create_bounding_box_mesh(min_point, max_point, color=bbox_colors[idx], opacity=0.05)
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
        eye=dict(x=-1.0, y=0, z=2.5)
    )
    camera2 = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=2.5)
    )

    # 更新 layout
    fig.update_layout(
        scene=dict(
            annotations=annotations1,
            aspectmode='data',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        scene2=dict(
            annotations=annotations2,
            aspectmode='data',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        scene_camera=camera,
        scene2_camera=camera2,
        width=1600,
        height=800,
        margin=dict(r=0, l=0, b=0, t=40),
        showlegend=False,
        title=dict(text=title, automargin=True),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    if output_fp is not None:
        fig.write_image(output_fp, format='png')
    else:
        fig.show(fig_show)

    if output_fp is not None and fig_show is not None:
        fig.show(fig_show)

    return fig


def pointcloud_visualize(vertices:np.array):
    # 场景
    fig = go.Figure()

    all_coords = np.concatenate(vertices, axis=0)

    min_val = np.min(all_coords)
    max_val = np.max(all_coords)

    # x, y, z坐标
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    # 在场景中添加点云
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=3,
            opacity=1
        )
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='X Axis',
                range=[min_val, max_val]
            ),
            yaxis=dict(
                title='Y Axis',
                range=[min_val, max_val]
            ),
            zaxis=dict(
                title='Z Axis',
                range=[min_val, max_val]
            ),
            aspectmode='cube'  # 确保各个轴的比例相同
        ),
    )

    fig.show("browser")


# 可视化作为 condition 的 PointCloud
def pointcloud_condition_visualize(vertices: np.ndarray, output_fp=None):
    assert vertices.ndim == 2 and vertices.shape[1] == 3, "vertices 应为 (N, 3) 的 numpy 数组"

    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    color = "#717388"  # 用 z 来着色
    xrange = x.max() - x.min()
    yrange = y.max() - y.min()
    zrange = z.max() - z.min()
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=color,
                colorscale='Viridis',
                opacity=1,
                showscale=False  # 不显示 colorbar
            ),
            showlegend=False  # 不显示图例
        )
    ])

    # 隐藏坐标轴、网格、背景等
    axis_style = dict(
        showbackground=False,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False,
        visible=False  # 最直接隐藏整个轴
    )
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=2)
    )
    fig.update_layout(
        scene=dict(
            xaxis=axis_style,
            yaxis=axis_style,
            zaxis=axis_style,
            aspectmode='manual',
            aspectratio=dict(
                x=xrange,
                y=yrange,
                z=zrange
            )
        ),
        scene_camera=camera,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    RESO = 800
    if output_fp:
        # fig.write_html(output_fp.replace(".pkl", "") + "_pcCond_vis.html")
        fig.write_image(output_fp.replace(".pkl", "") + "_pcCond.png", width=RESO, height=RESO, scale=2.5)

def draw_per_panel_geo_imgs(surf_ncs, surf_mask, colors, pad_size=5, out_dir=''):
    n_surfs = surf_ncs.shape[0]
    reso = int(surf_ncs.shape[1] ** 0.5)

    framed_imgs = []

    _surf_ncs = surf_ncs.reshape(n_surfs, reso, reso, 3)
    _surf_mask = surf_mask.reshape(n_surfs, reso, reso, 1)

    # 限定图像范围防止越界
    _surf_ncs[_surf_ncs>1] = 1
    _surf_ncs[_surf_ncs<-1] = -1

    for idx in range(n_surfs):
        mask_img = _surf_mask[idx, ...].astype(np.float32)
        _inv_mask_img = 1.0 - mask_img

        _padded_mask = _pad_arr(_inv_mask_img * 0.33, pad_size=pad_size, pad_value=1.0)

        _cur_color = colors[idx]
        if type(_cur_color) is str: _cur_color = to_rgb(_cur_color)

        _bg_img = np.zeros_like(_padded_mask.repeat(3, axis=-1)) + np.asarray(_cur_color)[None, None, :3]
        _bg_img = np.concatenate([_bg_img * _padded_mask, _padded_mask], axis=-1)

        _fg_img = np.concatenate([(np.clip(_surf_ncs[idx, ...], -1.0, 1.0) + 1.0) * 0.5, _surf_mask[idx, ...]], axis=-1)
        _fg_img = _pad_arr(_fg_img, pad_size=pad_size, pad_value=0.0)

        fused_img = _bg_img + _fg_img

        framed_imgs.append(fused_img)

        fused_pil_img = Image.fromarray((fused_img * 255).astype(np.uint8))

        os.makedirs(out_dir, exist_ok=True)
        fused_pil_img.save(os.path.join(out_dir, f'surf_{idx:02d}.png'))

    return framed_imgs


def get_visualization_steps():
    """
    get which step should be visualize during training

    Args:
        total_steps:

    Returns:

    """
    total_steps = 1000
    steps = [999]

    for i in range(total_steps - 1, -1, -1):
        if i >= 200:
            if i % 40 == 0:
                if i not in steps:
                    steps.append(i)
        elif i >= 100:
            if i % 10 == 0:
                steps.append(i)
        elif i >= 50:
            if i % 5 == 0:
                steps.append(i)
        elif i >= 20:
            if i % 1 == 0:
                steps.append(i)
        else:
            steps.append(i)

    return steps