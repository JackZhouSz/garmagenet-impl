import os
import math
import random
import pickle 
import torch
import numpy as np 
from tqdm import tqdm
import random
from multiprocessing.pool import Pool
from utils import (
    rotate_point_cloud,
    bbox_corners,
    rotate_axis,
    get_bbox,
    pad_repeat,
    pad_zero,
)

# furniture class labels
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
    "绑带": {"alias": "绑带", "color": "#696969"},
}

_CMAP_LINE = {
    "领窝线": {"alias": "领窝线", "color": "#8000FF"},          
    "袖笼弧线": {"alias": "袖笼弧线", "color": "#00B5EB"},      # interfaces between sleeve panels and bodice panels (belongs to bodice panels)
    "袖山弧线": {"alias": "袖山弧线", "color": "#00B5EB"},      # interfaces between sleeve panels and bodice panels (belongs to sleeve panels)
    "腰线": {"alias": "腰线", "color": "#80FFB4"},
    "袖口线": {"alias": "袖口线", "color": "#FFB360"},
    "底摆线": {"alias": "底摆线", "color": "#FF0000"},
    
    "省": {"alias": "省道", "color": "#FF3333"},
    "褶": {"alias": "褶", "color": "#33FF33"},
}
 
_PANEL_CLS = [
    '帽', '领', '肩', '袖片', '袖口', '衣身前中', '衣身后中', '衣身侧', '底摆', '腰头', 
    '裙前中', '裙后中', '裙侧', '橡筋', '木耳边', '袖笼拼条', '荷叶边', '绑带']


def _bbox_to_offset_scale(bbox):
    bbox_dim = bbox.shape[-1] // 2
    bbox_min, bbox_max = bbox[:, :bbox_dim], bbox[:, bbox_dim:] 
    
    offset = (bbox_min + bbox_max) * 0.5
    scale = (bbox_max - bbox_min) * 0.5
    
    return np.concatenate([offset, scale], axis=-1)


class SurfData(torch.utils.data.Dataset):
    """ Surface VAE Dataloader """
    def __init__(self, 
                 input_data, 
                 input_list, 
                 data_fields=['surf_ncs'], 
                 validate=False, 
                 aug=False, 
                 chunk_size=-1): 
        
        self.validate = validate
        self.aug = aug

        self.data_root = input_data
        self.data_fields = data_fields   

        print('Loading %s data...'%('validation' if validate else 'training'))
        with open(input_list, "rb") as tf: self.data_list = pickle.load(tf)['val' if validate else 'train']
        print("Total items: ", len(self.data_list))

        self.chunk_size = chunk_size if chunk_size > 0 and chunk_size < len(self.data_list) else len(self.data_list)
        if self.validate: self.chunk_size = self.chunk_size // 8
        self.chunk_idx = -1        
        self.data_chunks = [self.data_list[i:i+self.chunk_size] for i in range(0, len(self.data_list), self.chunk_size)]
        print('Data chunks: num_chunks=%d, chunk_size=%d.'%(len(self.data_chunks), self.chunk_size))

        self._next_chunk(lazy=False)
        
        
    def _next_chunk(self, lazy=False):
        
        if lazy and np.random.rand() < 0.5: return
        
        chunk_idx = (self.chunk_idx + 1) % len(self.data_chunks)        
        if self.chunk_idx == chunk_idx: return  
        else: self.chunk_idx = chunk_idx
        print('Switching to chunk %d/%d'%(self.chunk_idx, len(self.data_chunks)))
        
        cache = []
        for uid in tqdm(self.data_chunks[self.chunk_idx]):
            path = os.path.join(self.data_root, uid)
            
            try:
                with open(path, "rb") as tf: data = pickle.load(tf)
                
                data_pack = []
                for data_field in self.data_fields:
                    if data_field == 'surf_mask':
                        # scaling mask from [0, 1] to [-1, 1]
                        data_pack.append(data[data_field].astype(np.float32)*2.0-1.0)
                    else:
                        data_pack.append(data[data_field])    
                                        
                data_pack = np.concatenate(data_pack, axis=-1)
                cache.append(data_pack)
            
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
            
        self.cache = np.vstack(cache)
        self.num_channels = self.cache.shape[-1]
        self.resolution = self.cache.shape[1]

        print('Load chunk [%03d/%03d]: '%(self.chunk_idx, len(self.data_chunks)), self.cache.shape)
        
        
    def update(self): self._next_chunk(lazy=True)
    
    def __len__(self): return len(self.cache) * 10
    
    def __getitem__(self, index):
        return torch.FloatTensor(self.cache[index%len(self.cache)])
        # surf_uv = self.data[index]
        # N, H, W, C = surf_uv.shape
        # if np.random.rand() > 0.5 and self.aug:
        #     for axis in ['x', 'y', 'z']:
        #         angle = random.choice([90, 180, 270])
        #         surf_uv[..., :3] = rotate_point_cloud(surf_uv[..., :3].reshape(-1, 3), angle, axis).reshape(H, W, 3)
        # return torch.FloatTensor(surf_uv)


class SurfPosData(torch.utils.data.Dataset):
    """ Surface position (3D bbox) Dataloader """
    def __init__(self, input_data, input_list, validate=False, aug=False, pad_mode='repeat', args=None): 
        
        self.max_face = args.max_face
        self.bbox_scaled = args.bbox_scaled
        
        self.validata = validate
        self.aug = aug
        
        self.pad_mode = pad_mode
        
        # Load data
        self.data_root = input_data
        self.data_fields = args.data_fields  
        self.pos_dim = self.__get_pos_dim__()
        self.padding = args.padding
        
        print('Loading %s data...'%('validation' if validate else 'training'))
        with open(os.path.join(self.data_root, input_list), 'rb') as f: 
            self.data_list = pickle.load(f)['val' if self.validata else 'train']
            self.data_list = [x for x in self.data_list if os.path.exists(os.path.join(self.data_root, x))]
        print('Total items: ', len(self.data_list))


    def __get_pos_dim__(self):
        pos_dim = 0
        if 'surf_bbox_wcs' in self.data_fields: pos_dim += 3
        if 'surf_uv_bbox_wcs' in self.data_fields: pos_dim += 2
        return pos_dim


    def __len__(self): return len(self.data_list)


    def __getitem__(self, index):
        
        data_fp = os.path.join(self.data_root, self.data_list[index])        
        with open(data_fp, 'rb') as f: data = pickle.load(f)
                
        # Load surf pos
        surf_pos = []
        if 'surf_bbox_wcs' in self.data_fields: surf_pos.append(data['surf_bbox_wcs'].astype(np.float32))
        if 'surf_uv_bbox_wcs' in self.data_fields: surf_pos.append(data['surf_uv_bbox_wcs'].astype(np.float32))
        if 'surf_cls' in self.data_fields: surf_pos.append(data['surf_cls'][..., None].astype(np.float32))
        surf_pos = np.concatenate(surf_pos, axis=-1)
                
        # padding && shuffling
        if self.padding == 'repeat':
            surf_pos, _ = pad_repeat(surf_pos, self.max_face)            
        else:
            surf_pos, _ = pad_zero(surf_pos, self.max_face, shuffle=True, return_mask=True)
        
        return (
            torch.FloatTensor(surf_pos[..., :self.pos_dim*2]),
            torch.LongTensor(surf_pos[..., -1:]) if 'surf_cls' in self.data_fields else \
                torch.LongTensor(np.zeros((self.max_face, 1))-1),
            data['caption'] if 'caption' in self.data_fields else ''   
        )
    

class SurfZData(torch.utils.data.Dataset):
    """ Surface latent geometry Dataloader """
    def __init__(self, input_data, input_list, validate=False, aug=False, pad_mode='repeat', args=None): 
        
        self.max_face = args.max_face
        self.bbox_scaled = args.bbox_scaled
        
        self.validata = validate
        self.aug = aug
        
        self.pad_mode = pad_mode
        
        # Load data
        self.data_root = input_data
        self.data_fields = args.data_fields  
        self.pos_dim = self.__get_pos_dim__()
        self.padding = args.padding
        
        print('Loading %s data...'%('validation' if validate else 'training'))
        with open(os.path.join(self.data_root, input_list), 'rb') as f: 
            self.data_list = pickle.load(f)['val' if self.validata else 'train']
            self.data_list = [x for x in self.data_list if os.path.exists(os.path.join(self.data_root, x))]
        print('Total items: ', len(self.data_list))


    def __len__(self): return len(self.data_list)


    def __getitem__(self, index):

        data_fp = os.path.join(self.data_root, self.data_list[index])        
        with open(data_fp, 'rb') as f: data = pickle.load(f)
        
        # Load surfpos
        surf_pos = []
        if 'surf_bbox_wcs' in self.data_fields: surf_pos.append(data['surf_bbox_wcs'].astype(np.float32))
        if 'surf_uv_bbox_wcs' in self.data_fields: surf_pos.append(data['surf_uv_bbox_wcs'].astype(np.float32))
        if 'surf_cls' in self.data_fields: surf_pos.append(data['surf_cls'][..., None].astype(np.float32))
        surf_pos = np.concatenate(surf_pos, axis=-1)
        
        # Load surfncs
        surf_ncs = []
        if 'surf_ncs' in self.data_fields: surf_ncs.append(data['surf_ncs'].astype(np.float32))
        if 'surf_uv_ncs' in self.data_fields: surf_ncs.append(data['surf_uv_ncs'].astype(np.float32))
        if 'surf_mask' in self.data_fields: surf_ncs.append(data['surf_mask'].astype(np.float32)*2.0-1.0)
        surf_ncs = np.concatenate(surf_ncs, axis=-1)
        
        n_surfs, n_pads = surf_pos.shape[0], self.max_face-surf_pos.shape[0]
        if self.padding == 'repeat':
            pad_idx = np.random.permutation(np.concatenate([
                np.arange(n_surfs), np.random.choice(n_surfs, n_pads, replace=True)
                ], axis=0))
            surf_pos, surf_ncs = surf_pos[pad_idx, ...], surf_ncs[pad_idx, ...]
            pad_mask = np.random.rand(self.max_face) > 0.5
        else:
            # Zero-padding
            pad_idx = np.random.permutation(self.max_face)
            pad_mask = np.array([True]*n_surfs+[False]*n_pads, dtype=bool)[pad_idx]
            surf_pos = np.concatenate([
                surf_pos, np.zeros((n_pads, *surf_pos.shape[1:]))
            ], axis=0)[pad_idx, ...]
            surf_ncs = np.concatenate([
                surf_ncs, np.zeros((n_pads, *surf_ncs.shape[1:]))
            ], axis=0)[pad_idx, ...]
            
        return (
            torch.FloatTensor(surf_pos[...,:self.pos_dim*2]),
            torch.FloatTensor(surf_ncs),
            torch.BoolTensor(pad_mask),
            torch.LongTensor(surf_pos[..., -1:]) if 'surf_cls' in self.data_fields else \
                torch.LongTensor(np.zeros((self.max_face, 1))-1),
            data['caption'] if 'caption' in self.data_fields else ''
        )