import pandas as pd
import os
import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
from logzero import logger
import fiftyone.brain as fob

def get_pruned_dataset(csv_paths, 
                       output_path, 
                       total_images,
                       visdrone_img_paths,
                       c_num,
                       n_neighbors= 5,
                       result_dir = 'clipNN_output'):
    
    os.makedirs(result_dir, exist_ok=True)


    all_data = []

    for csv_path in csv_paths:
        data = pd.read_csv(csv_path)
        all_data.append(data)

    data = pd.concat(all_data)
    unq_img_paths = data['img_path'].unique()

    try:
        # load all the visdrone images to fiftyone
        visdrone_51 = fo.Dataset.from_images(visdrone_img_paths,    
                                            name="visdrone")

        visdrone_51.persistent = True  
        visdrone_51.save()
        # save the 

    except Exception as e:
        # load from the cache
        visdrone_51  = fo.load_dataset("visdrone")

    # check if the np file exists
    embeddings_path = "embeddings/visdrone_embeddings_clip.npy"
    os.makedirs("embeddings", exist_ok=True)


    # visdrone_51 = visdrone_51.limit(100)

    if not os.path.exists(embeddings_path):
        # compute the inception embeddings
        model = foz.load_zoo_model("clip-vit-base32-torch")
        target_feature = visdrone_51.compute_embeddings(model)
        # save the embedding as np file
        np.save(embeddings_path, target_feature)

    else:
        target_feature = np.load(embeddings_path)




    # dataset loading
    try:
        # load all the visdrone images to fiftyone
        source_dataset = fo.Dataset.from_images(unq_img_paths,    
                                            name="source_dataset")

        source_dataset.persistent = True  
        source_dataset.save()
        # save the 

    except Exception as e:
        # load from the cache
        source_dataset  = fo.load_dataset("source_dataset")


    # source_dataset = source_dataset.limit(100)

    # embeddings loading
    feature_path = "embeddings/feature_clip.npy"
    if not os.path.exists(feature_path):
        # compute the embeddings
        model = foz.load_zoo_model("clip-vit-base32-torch")
        feature_infer = source_dataset.compute_embeddings(model)
        # save the embeddings
        np.save(feature_path, feature_infer)

    else:
        feature_infer = np.load(feature_path)



    dataset = fo.Dataset()
    dataset.merge_samples(source_dataset)
    dataset.merge_samples(visdrone_51)

    all_embds = np.concatenate((feature_infer, target_feature), axis=0)


    umap_embs_path = "embeddings/umap_embs.npy"

    if not os.path.exists(umap_embs_path):
        import umap
        embs = fob.compute_visualization(
                                            dataset,
                                            embeddings=all_embds,
                                            num_dims=2,
                                            method="umap",
                                            brain_key="mnist_test",
                                            verbose=True,
                                            seed=51,
                                        )
        np.save(umap_embs_path, embs.points)

    else:
        embs = np.load(umap_embs_path)

    
    source_embs = embs[:len(feature_infer)]
    target_embs = embs[len(feature_infer):]

    logger.info("getting the nearest neighbours")

    # get the nearest neighbours
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=n_neighbors)

    model = neigh.fit(source_embs)

    # get all the nearest neighbours from the source to the target
    distances, indices = model.kneighbors(target_embs)

    # get the indices of the nearest neighbours
    indices = indices.flatten()

    # get the unique indices
    indices = np.unique(indices)
    img_names=unq_img_paths[indices]

    img_names = img_names[:total_images]


    final_df = data[data['img_path'].isin(img_names)]

    # check if final unq images are equal to the total images
    if len(final_df['img_path'].unique()) != total_images:
        logger.warn("The number of images are not equal to the total images")
        logger.warn("Images in final df: " + str( len(final_df['img_path'].unique())))
        logger.warn("Total images: " + str(total_images))
        # logger.error(len(final_df['img_path'].unique()), total_images)
        pass

    csv_op_path = os.path.join(result_dir, 'clipNN_pruned.csv')
    final_df.to_csv(csv_op_path, index=False)
    logger.info("The pruned dataset is saved at: " + csv_op_path)
    return

if __name__ == '__main__':
    csv_paths = [
        'csv_info/coco_annotation.csv',
        'csv_info/ade_annotation.csv',
        'csv_info/cityscapes_annotation.csv',
        'csv_info/voc_annotation.csv'
    ]


    from glob import glob
    visdrone_paths = '/data/tmp_teja/datacv/final/visdrone/VisDrone2019-VID-train/sequences/*/*'
    visdrone_img_paths = glob(visdrone_paths)
    visdrone_img_paths.sort()


    # no of clusters
    c_num = len(glob('/data/tmp_teja/datacv/final/visdrone/VisDrone2019-VID-train/sequences/*'))


    output_path = 'csv_info'
    total_images = 10
    nearest_neighbours = 10
    get_pruned_dataset(csv_paths, output_path, total_images,
                          visdrone_img_paths, c_num, n_neighbors = nearest_neighbours)

