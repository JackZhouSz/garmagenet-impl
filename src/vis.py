import os
import random
from glob import glob

import numpy as np
import pickle

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from matplotlib.colors import to_rgb

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
        line=dict(color=color, width=5),
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


def draw_geometry(surf_pos, surf_ncs):
    pass
