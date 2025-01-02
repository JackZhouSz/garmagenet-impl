import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add horizontal lines with arrows
def add_horizontal_arrow(y_position, x_start, x_end, text, color="black"):
    plt.annotate(
        '',
        xy=(x_end, y_position),
        xytext=(x_start, y_position),
        arrowprops=dict(arrowstyle='<->', color=color),
    )
    plt.text((x_start + x_end) / 2, y_position + 50, text, ha='center', fontsize=10, color=color)


def draw_hist_comp(data_id, src_data, trgt_data, src_label="stylexd", trgt_label="garmentcode", x_lim=None, num_samples=10000):
    
    if isinstance(src_data, pd.Series): src_data = src_data.values
    if isinstance(trgt_data, pd.Series): trgt_data = trgt_data.values
    
    src_data = np.random.choice(src_data, num_samples)
    trgt_data = np.random.choice(trgt_data, num_samples)
            
    data_range = (
        min(min(src_data), min(trgt_data)),
        max(max(src_data), max(trgt_data)))
    
    bins = np.linspace(data_range[0], data_range[1], 50)
        
    plt.style.use("seaborn-v0_8")
    
    data_hist_src = plt.hist(src_data, bins=bins, color="skyblue", alpha=0.5, label=src_label, edgecolor="white", range=data_range)
    data_hist_trgt = plt.hist(trgt_data, bins=bins, color="red", alpha=0.5, label=trgt_label, edgecolor="white", range=data_range)
    
    src_label_pos = max(data_hist_src[0]) * 0.75
    trgt_label_pos = max(data_hist_trgt[0])
    
    src_mean = src_data.mean()
    src_std = src_data.std()
    trgt_mean = trgt_data.mean()
    trgt_std = trgt_data.std()
    
    plt.axvline(src_mean, color="blue", linestyle="--", linewidth=1)
    plt.axvline(src_mean + src_std, color="blue", linestyle=":", linewidth=1)
    plt.axvline(src_mean - src_std, color="blue", linestyle=":", linewidth=1)
    plt.text(src_mean, src_label_pos, f"Mean = {src_mean:.2f}", color="blue", ha="center", fontsize=10)
    add_horizontal_arrow(src_label_pos-100*(src_label_pos//1000.0 + 1.0), src_mean - src_std, src_mean + src_std, f"Std = {src_std:.2f}", color="blue")
    
    plt.axhline(0, color="black", linewidth=1)
    
    plt.axvline(trgt_mean, color="red", linestyle="--", linewidth=1)
    plt.axvline(trgt_mean + trgt_std, color="red", linestyle=":", linewidth=1)
    plt.axvline(trgt_mean - trgt_std, color="red", linestyle=":", linewidth=1)
    plt.text(trgt_mean, trgt_label_pos, f"Mean = {trgt_mean:.2f}", color="red", ha="center", fontsize=10)
    add_horizontal_arrow(trgt_label_pos-100*(src_label_pos//1000.0 + 1.0), trgt_mean - trgt_std, trgt_mean + trgt_std, f"Std = {trgt_std:.2f}", color="red")

    plt.xlim(0, x_lim if x_lim is not None else data_range[1])
    # plt.ylim(0, max(src_y_max, trgt_y_max) * 1.2)

    plt.legend()

    plt.xlabel(f"Number of {data_id.split('/')[0].capitalize()} per {data_id.split('/')[1].capitalize()}")
    plt.ylabel(f"{data_id.split('/')[1].capitalize()} Count")
    # plt.title("Distribution of Number of Edges per Panel")
    # plt.show()
    plt.savefig(f"results/{data_id.replace('/', '_per_')}.png", dpi=300, bbox_inches='tight')
    plt.clf()
    
    print('[Done] saving results/' + data_id.replace('/', '_per_') + '.png')
            

# Number of edges per panel
stylexd_data = pd.read_csv("styxd_data_edge.csv")["edge_num"]
garmentcode_data = pd.read_csv("garmentcode_data_edge.csv")["edge_num"]
draw_hist_comp("edges/panel", stylexd_data, garmentcode_data, x_lim=50)

# Number of panels per pattern
stylexd_data = pd.read_csv("styxd_data_panel_stitch.csv")["panel_num"]
garmentcode_data = pd.read_csv("garmentcode_data_panel_stitch.csv")["panel_num"]
draw_hist_comp("panels/pattern", stylexd_data, garmentcode_data, x_lim=30)

# Number of stitches per pattern
stylexd_data = pd.read_csv("styxd_data_panel_stitch.csv")["stitch_num"]
garmentcode_data = pd.read_csv("garmentcode_data_panel_stitch.csv")["stitch_num"]
draw_hist_comp("stitches/panel", stylexd_data, garmentcode_data, x_lim=125)

# Number of vertices per garment
stylexd_data = pd.read_csv("stylexd_verts_faces.csv")["vertex_num"]
garmentcode_data = pd.read_csv("garmentcode_data_panel_stitch.csv")["vertex_num"]
draw_hist_comp("vertices/garment", stylexd_data, garmentcode_data, x_lim=100000)

# Number of faces per garment
stylexd_data = pd.read_csv("stylexd_verts_faces.csv")["face_num"]
garmentcode_data = pd.read_csv("garmentcode_data_panel_stitch.csv")["face_num"]
draw_hist_comp("faces/garment", stylexd_data, garmentcode_data, x_lim=60000)
