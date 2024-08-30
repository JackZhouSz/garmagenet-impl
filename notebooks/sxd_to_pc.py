# convert obj to point cloud
import os
import sys

import json
import random

import argparse

from glob import glob
from tqdm import tqdm

import numpy as np

sys.path.insert(0, os.path.abspath('..'))

from data_process.geometry_utils.obj import read_obj
from matplotlib.colors import to_rgb

from tqdm.contrib.concurrent import process_map

_CMAP = {
    "帽": {"alias": "hat", "color": "#F7815D"},
    "领": {"alias": "collar", "color": "#F9D26D"},
    "肩": {"alias": "shoulder", "color": "#F23434"},
    "袖片": {"alias": "sleeve", "color": "#C4DBBE"},
    "袖口": {"alias": "cuff", "color": "#F0EDA8"},
    "衣身前中": {"alias": "body front", "color": "#8CA740"},
    "衣身后中": {"alias": "body back", "color": "#4087A7"},
    "衣身侧": {"alias": "body side", "color": "#DF7D7E"},
    "底摆": {"alias": "hem", "color": "#DACBBD"},
    "腰头": {"alias": "belt", "color": "#DABDD1"},
    
    "裙前中": {"alias": "skirt front", "color": "#46B974"},
    "裙后中": {"alias": "skirt back", "color": "#6B68F5"},
    "裙侧": {"alias": "skirt side", "color": "#D37F50"},
    
    "裤前中": {"alias": "pelvis front", "color": "#46B974"},
    "裤后中": {"alias": "pelvis back", "color": "#6B68F5"},
    "裤侧": {"alias": "pelvis side", "color": "#D37F50"},

    "橡筋": {"alias":"ruffles", "color": "#A8D4D2"},
    "木耳边": {"alias":"ruffles", "color": "#A8D4D2"},
    "袖笼拼条": {"alias":"ruffles", "color": "#A8D4D2"},
    "荷叶边": {"alias":"ruffles", "color": "#A8D4D2"},
    "绑带": {"alias":"ruffles", "color": "#A8D4D2"}
}

_GLOBAL_SCALE = 1000.0


def process_data(data_dir, output_dir):
    obj_fp = os.path.join(data_dir, os.path.basename(data_dir)+'.obj')
    pattern_fp = os.path.join(data_dir, 'pattern.json')
    
    with open(pattern_fp, 'r', encoding='utf-8') as f: pattern_spec = json.load(f)
    
    out_panel_names = dict([(
        x['id'], 
        _CMAP[x['label'].split('|')[0].strip()]['alias'].replace(' ', '')+'_'+x['id'].split('-')[0]) \
        for x in pattern_spec['panels']])
    
    out_panel_names_sorted = {}
    for out_item in sorted(out_panel_names.values()):
        panel_cls, panel_id = out_item.split('_')
        if panel_cls not in out_panel_names_sorted: out_panel_names_sorted[panel_cls] = []
        out_panel_names_sorted[panel_cls].append(panel_id)
        
    for out_item in out_panel_names:
        panel_cls, panel_id = out_panel_names[out_item].split('_')
        panel_idx = out_panel_names_sorted[panel_cls].index(panel_id)
        out_panel_names[out_item] = '%s_%d'%(panel_cls, panel_idx+1)
        
    mesh_obj = read_obj(obj_fp)
    
    verts = mesh_obj.points
    normals = mesh_obj.point_data['obj:vn']
    uv = mesh_obj.point_data['obj:vt']
    
    for idx, panel_id in enumerate(mesh_obj.field_data['obj:group_tags']):
        if panel_id not in out_panel_names: continue
        
        panel_faces = mesh_obj.cells[idx].data        
        panel_verts_idx = np.unique(panel_faces.flatten())
        
        panel_verts = verts[panel_verts_idx, :] / _GLOBAL_SCALE
        panel_normals = normals[panel_verts_idx, :]
        panel_uvs = uv[panel_verts_idx, :2] / _GLOBAL_SCALE     # global uv
        
        panel_output = np.concatenate([panel_verts, panel_normals, panel_uvs], axis=1)
        panel_out_fp = os.path.join(output_dir, out_panel_names[panel_id]+'.txt')
    
        with open(panel_out_fp, 'w') as f:
            f.writelines('# x y z nx ny nz u v\n')
            np.savetxt(f, panel_output, fmt='%.6f')
            
    return len(out_panel_names)

    
def process_item(data_idx, data_dir, args):
    try:
        output_dir = os.path.join(args.output, 'garment_%05d'%data_idx)
        os.makedirs(output_dir, exist_ok=True)       
        result = process_data(data_dir, output_dir)        
        return result > 0, data_dir
    except Exception as e:
        return False, f"{data_dir} | [ERROR] {e}"
    
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Draw panel bbox 3D")
    parser.add_argument("-i", "--input", default="./resources/examples",
                        type=str, help="Input directories splited bt comma.")
    parser.add_argument("-o", "--output", default='./resources/examples/processed',
                        type=str, help="Output directory.")
    parser.add_argument("-r", "--range", default=None, type=str, 
                        help="Path to executable.")
    
    args, cfg_cmd = parser.parse_known_args()
    
    data_root_dirs = args.input.split(',')
    print('Input directories: ', data_root_dirs)
    
    data_items = []
    for idx, data_root in enumerate(data_root_dirs):
        cur_data_items = sorted([os.path.dirname(x) for x in glob(
            os.path.join(data_root, '**', 'pattern.json'), recursive=True)])
        data_items += cur_data_items
        print('[%02d/%02d] Found %d items in %s'%(idx+1, len(data_root_dirs), len(cur_data_items), data_root))
    print('Total items: ', len(data_items))
        
    log_file = os.path.join(args.output, 'app.log')
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            processed = [x.split("\t")[0] for x in lines if x.split("\t")[1].strip() == "1"]
            data_items = [x for x in data_items if x not in processed]
        
    if args.range is not None:
        if ',' in args.range:
            begin, end = args.range.split(",")
            begin, end = max(0, int(begin)), min(int(end), len(data_items))
            data_items = data_items[begin:end]
            print("Extracting range: %d" % (len(data_items)))
        else:
            data_items = random.choices(data_items, k=int(args.range))
            print("Extracting random items: %d" % (len(data_items)))
    
    results = process_map(
        process_item, 
        list(range(len(data_items))),
        data_items, 
        [args] * len(data_items), 
        max_workers=8)
    
    failed_items = []
    with open(log_file, 'a+') as f: 
        for result, data_item in results:
            if not result:
                    data_item, err_code = data_item.split('|')
                    data_item = data_item.strip()
                    err_code = err_code.strip()
                    failed_items.append(data_item)
                    print('[ERROR] Failed to process data:', data_item, err_code)
                    f.write(f"{data_item}\t 0\n")
            else:
                f.write(f"{data_item}\t 1\n")