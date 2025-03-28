import os
import sys

import argparse

import numpy as np
import torch

from PIL import Image
from glob import glob

from matplotlib import pyplot as plt

from pymilvus import MilvusClient

# Local environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.llm_utils.gme_inference import GmeQwen2VL


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database", type=str, 
        default='resources/examples/test_annotation/qwen_annotations', 
        help="Data folder path")
    parser.add_argument(
        "--collection", type=str, 
        default='stylexd_simple_retrieval', 
        help="Data collection")
    parser.add_argument(
        "--query", type=str, default='resources/examples/test_annotation/qwen_annotations_query', 
        help="Choose between dataset option [abc/deepcad/furniture] (default: abc)")
    parser.add_argument(
        "--model", type=str, default='/data/lry/models/gme-Qwen2-VL-2B-Instruct', 
        help="Embedder model path, default to gme-Qwen2-VL-2B-Instruct")
    args = parser.parse_args()    

    print('Model: ', args.model)

    gme_embedder = GmeQwen2VL(model_path=args.model, device='cuda')
    embedder_dim = 1536 if '2B' in args.model else 3584

    database_fp = args.database if args.database.endswith('.db') else args.database+'.db'

    if os.path.exists(database_fp):
        print('Loading database: ', database_fp)
        client = MilvusClient(database_fp)
    
    if not os.path.exists(database_fp):
        print('Creating Milvus collection from: ', args.database)
        
        data = []
        for idx, ref_data_id in enumerate([os.path.basename(x).split('.')[0] for x in glob(os.path.join(args.database, '*.md'))]):

            ref_sketch_fp = os.path.join(args.database, ref_data_id + '_sil.png')
            ref_image_fp = os.path.join(args.database, ref_data_id + '.png')

            ref_sketch = Image.open(ref_sketch_fp).resize((800, 1024))
            ref_image = Image.open(ref_image_fp)

            ref_desc_fp = os.path.join(args.database, ref_data_id + '.md')
            with open(ref_desc_fp, 'r') as f: ref_desc = f.read()

            emb_vec = gme_embedder.get_fused_embeddings(texts=[ref_desc], images=[ref_sketch])
            print(idx, 'emb_vec: ', emb_vec.shape)

            data.append({
                "id": idx,
                "vector": emb_vec.squeeze().detach().cpu().numpy().astype(np.float32),
                "data_id": ref_data_id,
                "desc_fp": ref_desc_fp,
                "sketch_fp": ref_sketch_fp,
                "img_fp": ref_image_fp
            })

        client = MilvusClient(database_fp)
        client.create_collection(collection_name="stylexd_simple_retrieval", dimension=embedder_dim)
        res = client.insert(collection_name="stylexd_simple_retrieval", data=data)
        print('[DONE] create miluvs database.')

    # Query
    query_embs = {
        'ids': [],
        'embs': []
        }

    for idx, query_data_id in enumerate([os.path.basename(x).split('.')[0].split('_')[0] for x in glob(os.path.join(args.query, '*.md'))]):
        query_sketch = Image.open(os.path.join(args.query, f'{query_data_id}_sketch.jpg'))
        query_desc_fp = os.path.join(args.query, f'{query_data_id}.md')

        with open(query_desc_fp, 'r') as f: query_desc = f.read()
        emb_vec = gme_embedder.get_fused_embeddings(texts=[query_desc], images=[query_sketch])
        
        query_embs['ids'].append(query_data_id)
        query_embs['embs'].append(emb_vec.squeeze().detach().cpu().numpy().astype(np.float32))

    query_res = client.search(
        collection_name="stylexd_simple_retrieval", 
        data=query_embs['embs'], 
        limit=5,
        output_fields=["vector", "data_id", "desc_fp", "sketch_fp", "img_fp"],
        search_params={"metric_type": "COSINE"}
        )

    top_k = {'1': 0, '3': 0, '5': 0}
    for idx, res in enumerate(query_res):

        num_results = min(len(res), 5)
        num_subplots = num_results + 2

        query_data_id = query_embs['ids'][idx]
        query_image_fp = os.path.join(args.query, f'{query_data_id}_img.jpg')
        query_sketch_fp = os.path.join(args.query, f'{query_data_id}_sketch.jpg')

        # collect top k
        all_ans_ids = [x['entity']['data_id'] for x in res]

        if query_data_id == all_ans_ids[0]: top_k['1'] += 1
        if query_data_id in all_ans_ids[:min(len(all_ans_ids), 3)]: top_k['3'] += 1
        if query_data_id in all_ans_ids[:min(len(all_ans_ids), 5)]: top_k['5'] += 1

        # visualize images
        fig = plt.figure(figsize=(num_subplots*5, 5))

        plt.subplot(1, num_subplots, 1)
        plt.axis('off')
        plt.imshow(Image.open(query_image_fp))
        plt.title('Query Image', fontsize=12)

        plt.subplot(1, num_subplots, 2)
        plt.axis('off')
        plt.imshow(Image.open(query_sketch_fp))
        plt.title('Query Sketch', fontsize=12)

        for iidx in range(num_results):
            plt.subplot(1, num_subplots, 3+iidx)
            plt.axis('off')
            plt.imshow(Image.open(query_res[idx][iidx]['entity']['sketch_fp']))
            query_dist = query_res[idx][iidx]['distance']
            plt.title(f'Top {iidx+1}, dist={query_dist}', fontsize=12)

        plt.tight_layout()
        fig.savefig(query_image_fp.replace('_img.jpg', '_result.jpg'), dpi=fig.dpi)

    print(
        'Top 1: ', float(top_k['1']) / len(query_res), 
        'Top 3: ', float(top_k['3']) / len(query_res), 
        'Top 5: ', float(top_k['5']) / len(query_res)
        )
