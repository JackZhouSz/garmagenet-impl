import random
import os
import pickle
import json
import numpy as np

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing.pool import Pool

data_dir = "/data/AIGP/brep_reso_256_edge_snap"
caption_dir = "/data/AIGP/brep_reso_256_edge_snap_with_caption/patterns_with_caption_english"

out_caption_dir = "/data/AIGP/brep_reso_256_edge_snap_with_caption/processed"
out_missed_dir = "/data/AIGP/brep_reso_256_edge_snap_with_caption/missed"
os.makedirs(out_caption_dir, exist_ok=True)
os.makedirs(out_missed_dir, exist_ok=True)

server_root = "\\\\192.168.29.222\Share"

def _process_item(data_fp):
    
    if os.path.isfile(os.path.join(out_caption_dir, data_fp)): return
    if os.path.isfile(os.path.join(out_missed_dir, data_fp)): return
    
    try:
        with open(os.path.join(data_dir, data_fp), 'rb') as f:
            data = pickle.load(f)
        query_key = data['data_fp'].replace(server_root, '').strip().replace('objs\\', '')
        
        if query_key in name_caption_dict:
            data['caption'] = name_caption_dict[query_key]['eng_caption'].lower()
            # print(query_key, data['caption'])
            with open(os.path.join(out_caption_dir, data_fp), 'wb') as f: pickle.dump(data, f)
        
        else:
            print('Captions: ', random.choice(list(name_caption_dict.keys())))
            print('Missing Key: ', data_fp, ' | ', data['data_fp'], query_key)
            with open(os.path.join(out_missed_dir, data_fp), 'wb') as f: pickle.dump(data, f)
    
    except Exception as e:
        print('[ERROR] identify caption %s'%(data_fp), str(e))
    
    
if not os.path.exists('name_caption_dict.pkl'):
    name_caption_dict = {}
    for data_fp in tqdm(os.listdir(caption_dir)):
        try:
            with open(os.path.join(caption_dir, data_fp), 'rb') as f:
                pattern_spec = json.load(f)
            
            item_key = pattern_spec['raw_data_fp'].strip().replace('objs\\', '').replace('sprojs\\', '')
            
            name_caption_dict[item_key] = {
                'caption': pattern_spec['caption'],
                'eng_caption': pattern_spec['caption']
            }
        except Exception as e:
            print('[ERROR] %s: '%(data_fp), str(e))
            
    with open('name_caption_dict.pkl', 'wb') as f: pickle.dump(name_caption_dict, f)

else:
    with open('name_caption_dict.pkl', 'rb') as f: name_caption_dict = pickle.load(f)


print('Caption dict: ', len(list(name_caption_dict.items())))

print('Mapping captions...')
r = process_map(_process_item, os.listdir(data_dir), max_workers=os.cpu_count(), chunksize=1)
