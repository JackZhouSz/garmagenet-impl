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
                 chunksize=-1): 
        
        self.validate = validate
        self.aug = aug

        self.data_root = input_data
        self.data_fields = data_fields   

        print('Loading %s data...'%('validation' if validate else 'training'))
        with open(input_list, "rb") as tf: self.data_list = pickle.load(tf)['val' if validate else 'train']
        print("Total items: ", len(self.data_list))

        self.chunksize = chunksize if chunksize > 0 and chunksize < len(self.data_list) else len(self.data_list)
        if self.validate: self.chunksize = self.chunksize // 8
        self.chunk_idx = -1        
        self.data_chunks = [self.data_list[i:i+self.chunksize] for i in range(0, len(self.data_list), self.chunksize)]
        print('Data chunks: num_chunks=%d, chunksize=%d.'%(len(self.data_chunks), self.chunksize))

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
                
                surf_ncs = []
                if 'surf_ncs' in self.data_fields: surf_ncs.append(data['surf_ncs'].astype(np.float32))
                if 'surf_uv_ncs' in self.data_fields: surf_ncs.append(data['surf_uv_ncs'].astype(np.float32))
                if 'surf_mask' in self.data_fields: surf_ncs.append(data['surf_mask'].astype(np.float32)*2.0-1.0)
                                        
                surf_ncs = np.concatenate(surf_ncs, axis=-1)
                cache.append(surf_ncs)
            
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
        
        self.validate = validate
        self.aug = aug
        
        self.pad_mode = pad_mode
        
        # Load data
        self.data_root = input_data
        self.data_fields = args.data_fields  
        self.pos_dim = self.__get_pos_dim__()
        self.padding = args.padding
        
        print('Loading %s data...'%('validation' if validate else 'training'))
        with open(input_list, 'rb') as f: 
            self.data_list = pickle.load(f)['val' if self.validate else 'train']
            self.data_list = [x for x in self.data_list if os.path.exists(os.path.join(self.data_root, x))]
            if self.validate: self.data_list = random.choices(self.data_list, k=128)
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
    def __init__(self, input_data, input_list, z_encoder=None, validate=False, aug=False, pad_mode='repeat', args=None): 
        
        self.z_encoder = z_encoder
        
        self.max_face = args.max_face
        self.bbox_scaled = args.bbox_scaled
        
        self.validate = validate
        self.aug = aug
        
        self.pad_mode = pad_mode
        
        # Load data
        self.data_root = input_data
        self.data_fields = args.data_fields  
        self.padding = args.padding
        
        print('Loading %s data...'%('validation' if validate else 'training'))
        with open(input_list, 'rb') as f: self.data_list = pickle.load(f)['val' if self.validate else 'train']
        self.data_list = [x for x in self.data_list if os.path.exists(os.path.join(self.data_root, x))]
        print('Total items: ', len(self.data_list))

        # Config data chunks
        self.chunksize = args.chunksize if args.chunksize > 0 and args.chunksize < len(self.data_list) else len(self.data_list)
        if self.validate: self.chunksize = self.chunksize // 8
        self.chunk_idx = -1        
        self.data_chunks = [self.data_list[i:i+self.chunksize] for i in range(0, len(self.data_list), self.chunksize)]
        print('Data chunks: num_chunks=%d, chunksize=%d.'%(len(self.data_chunks), self.chunksize))

        self.cache = {}
        self._next_chunk(lazy=False)


    def __len__(self): return len(self.data_chunks[self.chunk_idx]) * 256

    
    def init_encoder(self, z_encoder): 
        print("Init z_encoder...")
        self.z_encoder = z_encoder
        if 'latent' not in self.cache: self.__encode_all__()
        
    def update(self): self._next_chunk(lazy=True)

    def __load_one__(self, data_fp):
        
        with open(data_fp, 'rb') as f: data = pickle.load(f)
        
        # Load surfpos
        surf_pos = []
        if 'surf_bbox_wcs' in self.data_fields: surf_pos.append(data['surf_bbox_wcs'].astype(np.float32))
        if 'surf_uv_bbox_wcs' in self.data_fields: surf_pos.append(data['surf_uv_bbox_wcs'].astype(np.float32))
        surf_pos = np.concatenate(surf_pos, axis=-1) * self.bbox_scaled
        
        # Load surfncs
        surf_ncs = []
        if 'surf_ncs' in self.data_fields: surf_ncs.append(data['surf_ncs'].astype(np.float32))
        if 'surf_uv_ncs' in self.data_fields: surf_ncs.append(data['surf_uv_ncs'].astype(np.float32))
        if 'surf_mask' in self.data_fields: surf_ncs.append(data['surf_mask'].astype(np.float32)*2.0-1.0)
        surf_ncs = np.concatenate(surf_ncs, axis=-1)
        
        # Load semantics
        surf_cls = data['surf_cls'][..., None] if 'surf_cls' in self.data_fields \
            else np.zeros((self.max_face, 1)) - 1

        # Load caption
        caption = data['caption'] if 'caption' in self.data_fields else ''
        
        if not hasattr(self, 'num_channels'): self.num_channels = surf_ncs.shape[-1]
        if not hasattr(self, 'resolution'): self.resolution = surf_ncs.shape[1]
        if not hasattr(self, 'pos_dim'): self.pos_dim = surf_pos.shape[-1] // 2
        if not hasattr(self, 'num_classes'): self.num_classes = len(_PANEL_CLS) if 'surf_cls' in self.data_fields else 0
        
        return (
            torch.FloatTensor(surf_pos),
            torch.FloatTensor(surf_ncs),
            torch.LongTensor(surf_cls),
            caption
        )         


    def __pad_raw__(self, surf_pos, surf_ncs, surf_cls):
        n_surfs, n_pads = surf_pos.shape[0], self.max_face-surf_pos.shape[0]
        
        if self.padding == 'repeat':
            pad_idx = torch.permute(torch.cat([
                torch.arange(n_surfs), torch.randint(n_surfs, n_pads)
                ], dim=0))
            surf_pos, surf_ncs, surf_cls = surf_pos[pad_idx, ...], surf_ncs[pad_idx, ...], surf_cls[pad_idx, ...]
            pad_mask = torch.rand(self.max_face) > 0.5
        elif self.padding == 'zero':
            # Zero-padding
            pad_idx = torch.permute(torch.arange(self.max_face))
            pad_mask = torch.cat([
                torch.ones(n_surfs, dtype=bool), torch.zeros(n_pads, dtype=bool)
            ], dim=0)[pad_idx]
            surf_pos = torch.cat([
                surf_pos, torch.zeros((n_pads, *surf_pos.shape[1:]), dtype=surf_pos.dtype, device=surf_pos.device)
            ], dim=0)[pad_idx, ...]
            surf_ncs = torch.cat([
                surf_ncs, torch.zeros((n_pads, *surf_ncs.shape[1:]), dtype=surf_ncs.dtype, device=surf_ncs.device)
            ], dim=0)[pad_idx, ...]
            surf_cls = torch.cat([
                surf_cls, torch.zeros((n_pads, *surf_cls.shape[1:]), dtype=surf_cls.dtype, device=surf_cls.device)-1
            ], dim=0)[pad_idx, ...]
        else:
            pad_mask = torch.ones(self.max_face, dtype=bool)
        
        return surf_pos, surf_ncs, surf_cls, pad_mask
        

    def __pad_latents__(self, surf_pos, surf_latents, surf_cls):
        
        n_surfs, n_pads = surf_pos.shape[0], self.max_face-surf_pos.shape[0]
        
        if self.padding == 'repeat':
            pad_idx = torch.cat([
                torch.arange(n_surfs), torch.randint(0, n_surfs, (n_pads,))
                ], dim=0)[torch.randperm(self.max_face)]
            surf_pos, surf_latents, surf_cls = surf_pos[pad_idx, ...], surf_latents[pad_idx, ...], surf_cls[pad_idx, ...]
            pad_mask = torch.rand(self.max_face) > 0.5
        elif self.padding == 'zero':
            # Zero-padding
            pad_idx = torch.randperm(self.max_face)
            pad_mask = torch.cat([
                torch.ones(n_surfs, dtype=bool), torch.zeros(n_pads, dtype=bool)
            ], dim=0)[pad_idx]
            surf_pos = torch.cat([
                surf_pos, torch.zeros((n_pads, *surf_pos.shape[1:]), dtype=surf_pos.dtype, device=surf_pos.device)
            ], dim=0)[pad_idx, ...]
            surf_latents = torch.cat([
                surf_latents, self.zero_latent.repeat((n_pads, 1))
            ], dim=0)[pad_idx, ...]
            surf_cls = torch.cat([
                surf_cls, torch.zeros((n_pads, *surf_cls.shape[1:]), dtype=surf_cls.dtype, device=surf_cls.device)-1
            ], dim=0)[pad_idx, ...]
        else:
            pad_mask = torch.ones(self.max_face, dtype=bool)
            
        
        return surf_pos, surf_latents, surf_cls, pad_mask
        

    def __encode_all__(self):
        if not (hasattr(self, 'z_encoder') and self.z_encoder is not None): return

        try:
            z_device = self.z_encoder.module.parameters().__next__().device
        except:
            z_device = self.z_encoder.parameters().__next__().device   
                    
        _surf_ncs = self.cache['surf_ncs']
        _surf_ncs = _surf_ncs.permute(0, 3, 1, 2).to(z_device)
        
        _surf_latents = []
        _encode_bsz = 256
        
        with torch.no_grad():
            if not hasattr(self, 'zero_latent'):
                self.zero_latent = self.z_encoder(torch.zeros_like(_surf_ncs[0:1, ...])).flatten(start_dim=1).detach().cpu()
            for i in range(0, len(_surf_ncs), _encode_bsz):
                z = self.z_encoder(_surf_ncs[i:i+_encode_bsz])
                _surf_latents.append(z.flatten(start_dim=1))
                
        self.cache['latent'] = torch.cat(_surf_latents, dim=0).detach().cpu()        
        del self.cache['surf_ncs']
        
        print('Encode all data: ', self.cache['latent'].shape, self.cache.keys())

        
    def _next_chunk(self, lazy=False):
        
        if lazy and np.random.rand() < 0.5: return
        
        chunk_idx = (self.chunk_idx + 1) % len(self.data_chunks)        
        if self.chunk_idx == chunk_idx: return  
        else: self.chunk_idx = chunk_idx
        print('Switching to chunk %d/%d'%(self.chunk_idx, len(self.data_chunks)))
        
        del self.cache
        
        cache = {
            'surf_pos': [], 
            'surf_ncs': [], 
            'surf_cls': [], 
            'caption': [],
            'item_idx': []  # start and end index of each item in the cache
            }
        
        start_idx, end_idx = 0, 0
        for uid in tqdm(self.data_chunks[self.chunk_idx]):
            data_fp = os.path.join(self.data_root, uid)

            # try:
            surf_pos, surf_ncs, surf_cls, caption = self.__load_one__(data_fp)
            start_idx = end_idx
            end_idx = start_idx + surf_pos.shape[0]
            cache['surf_pos'].append(surf_pos)
            cache['surf_ncs'].append(surf_ncs)
            cache['surf_cls'].append(surf_cls)
            cache['caption'].append(caption)
            cache['item_idx'].append((start_idx, end_idx))
            
            # except Exception as e:
            #     print(f"Error loading {data_fp}: {e}")
            #     continue
        
        cache['surf_pos'] = torch.cat(cache['surf_pos'], dim=0)
        cache['surf_ncs'] = torch.cat(cache['surf_ncs'], dim=0)
        cache['surf_cls'] = torch.cat(cache['surf_cls'], dim=0)
                
        self.cache = cache
        self.__encode_all__()
        
        print('Load chunk [%03d/%03d]: '%(self.chunk_idx, len(self.data_chunks)))
        for key in self.cache.keys(): print(key, self.cache[key].shape \
            if torch.is_tensor(self.cache[key]) else len(self.cache[key]))
        
        
    def __getitem__(self, index):
        
        start_idx, end_idx = self.cache['item_idx'][index%len(self.cache['item_idx'])]
        caption = self.cache['caption'][index%len(self.cache['item_idx'])]

        surf_pos = self.cache['surf_pos'][start_idx:end_idx, ...]
        surf_latents = self.cache['latent'][start_idx:end_idx, ...]
        surf_cls = self.cache['surf_cls'][start_idx:end_idx, ...]
        
        surf_pos, surf_latents, surf_cls, pad_mask = self.__pad_latents__(
            surf_pos, surf_latents, surf_cls)
                
        return (
            surf_pos, surf_latents, pad_mask, surf_cls, caption
        )