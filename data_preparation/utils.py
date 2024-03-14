"""This script is to load the dataset and log to a csv file from the coco format"""

import os



import pandas as pd
import json
from tqdm import tqdm



def get_meta_info(json_path, img_path):
    # empty dataframe
    df = pd.DataFrame()


    # log all the image and the boxes and the id of the object in the csv file
    meta_info = json.load(open(json_path))
    for img in tqdm(meta_info['images']):

        img_name = img['file_name']
        for ann in meta_info['annotations']:
            if ann['image_id'] == img['id']:
                box = ann['bbox']
                labels = ann['category_id']
                ids = ann['id']
                x_min, y_min, width, height = box

                op = dict(img_name=img_name, labels=labels, ids=ids, x_min=x_min, y_min=y_min, width=width, height=height)
                df = df._append(op, ignore_index=True)                   

                # df = df._append({'img_name': img_name, 'labels': labels, 'ids': ids,
                #                     'x_min': x_min, 'y_min': y_min, 'width': width, 'height': height
                #                 }, ignore_index=True)
                

    df['img_path'] = img_path + '/' + df['img_name']
    return df



# for json_path, img_path in zip(jsons[::-1], imgs_path[::-1]):

def prepare_csv(json_path, img_path):

    os.makedirs('./csv_info', exist_ok=True)
    df = get_meta_info(json_path, img_path)
    # print(df.head())
    df.to_csv(f'./csv_info/{json_path.split("/")[-1].split(".")[0]}.csv', index=False)



if __name__ == '__main__':

    # jsons = ['/data/tmp_teja/datacv/data/source_pool/bdd_annotation.json', 
    #         '/data/tmp_teja/datacv/data/source_pool/detrac_annotation.json',
    #         '/data/tmp_teja/datacv/data/source_pool/kitti_annotation.json', 
    #         '/data/tmp_teja/datacv/data/source_pool/coco_annotation.json', 
    #         '/data/tmp_teja/datacv/data/source_pool/ade_annotation.json', 
    #         '/data/tmp_teja/datacv/data/source_pool/cityscapes_annotation.json', 
    #         '/data/tmp_teja/datacv/data/source_pool/voc_annotation.json'
    # ]


    # imgs_path  = ['/data/tmp_teja/datacv/data/source_pool/bdd_train', 
    #             '/data/tmp_teja/datacv/data/source_pool/detrac_train',
    #             '/data/tmp_teja/datacv/data/source_pool/kitti_train', 
    #             '/data/tmp_teja/datacv/data/source_pool/coco_train', 
    #             '/data/tmp_teja/datacv/data/source_pool/ade_train', 
    #             '/data/tmp_teja/datacv/data/source_pool/cityscapes_train', 
    #             '/data/tmp_teja/datacv/data/source_pool/voc_train',
    # ]



    jsons = [ '/data/tmp_teja/datacv/data/source_pool/voc_annotation.json']
    imgs_path  = ['/data/tmp_teja/datacv/data/source_pool/voc_train']


    for json_path, img_path in zip(jsons[::-1], imgs_path[::-1]):
        prepare_csv(json_path, img_path)

    # cpu_count = os.cpu_count()
    # from multiprocessing import Pool
    # with Pool(cpu_count) as p:
    #     p.starmap(prepare_csv, zip(jsons, imgs_path))