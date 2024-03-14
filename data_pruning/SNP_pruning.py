# Search and Prune algorithm

import pandas as pd
import fiftyone as fo
from logzero import logger
import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import os
from sklearn.cluster import KMeans
from tqdm import tqdm
import scipy


import copy


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)














def get_pruned_dataset(csv_paths, 
                       output_path, 
                       total_images,
                       visdrone_img_paths,
                       c_num,
                       result_dir = 'snp_output'):
    
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
    embeddings_path = "embeddings/visdrone_embeddings.npy"
    os.makedirs("embeddings", exist_ok=True)

    if not os.path.exists(embeddings_path):
        visdrone_51 = visdrone_51.limit(100)
        # compute the inception embeddings
        model = foz.load_zoo_model("inception-v3-imagenet-torch")
        target_feature = visdrone_51.compute_embeddings(model)
        # save the embedding as np file
        np.save(embeddings_path, target_feature)

    else:
        target_feature = np.load(embeddings_path)

    
    # mean of the embeddings
    m1 = np.mean(target_feature, axis=0)
    # standard deviation of the embeddings
    s1 = np.cov(target_feature, rowvar=False)
    sum_eigen_val1 = (s1.diagonal()).sum()


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


    feature_path = "embeddings/feature.npy"
    if not os.path.exists(feature_path):

        source_dataset = source_dataset.limit(100)

        # compute the embeddings
        model = foz.load_zoo_model("inception-v3-imagenet-torch")
        feature_infer = source_dataset.compute_embeddings(model)
        # save the embeddings
        np.save(feature_path, feature_infer)

    else:
        feature_infer = np.load(feature_path)



    # clustering ids based on ids' mean feature
    # if not os.path.exists(result_dir + '/label_cluster_'+str(c_num)+'_img.npy'):
    if True:
        print('=========== clustering ===========')
        estimator = KMeans(n_clusters=c_num)
        # print (c_num, int(np.shape (feature_infer)[0] / c_num ))
        # estimator = KMeansConstrained(n_clusters=c_num, size_min=int(np.shape (feature_infer)[0] / c_num )-1, size_max=int(np.shape (feature_infer)[0] / c_num)+1)
        estimator.fit(feature_infer)
        label_pred = estimator.labels_
        np.save(result_dir + '/label_cluster_'+str(c_num)+'_img.npy',label_pred)
    else:
        label_pred = np.load(result_dir  + '/label_cluster_'+str(c_num)+'_img.npy')
    
    
    if not os.path.exists(result_dir + '/cluster_fid_div_by_'+str(c_num)+'_img.npy'):
        cluster_feature = []
        cluster_fid = []
        cluster_mmd = []
        cluster_var_gap = []
        cluster_div = []

        for k in tqdm(range(c_num)):
            initial_feature_infer = feature_infer[label_pred==k]

            if len(initial_feature_infer) == 1:
                continue

            cluster_feature.append(initial_feature_infer)
            
            mu = np.mean(initial_feature_infer, axis=0)
            sigma = np.cov(initial_feature_infer, rowvar=False)

            fea_corrcoef = np.corrcoef(initial_feature_infer)
            fea_corrcoef = np.ones(np.shape(fea_corrcoef)) - fea_corrcoef
            diversity_sum = np.sum(np.sum(fea_corrcoef)) - np.sum(np.diagonal(fea_corrcoef))
            current_div = diversity_sum / (np.shape (fea_corrcoef)[0] ** 2 - np.shape (fea_corrcoef)[0])

            # caculating variance
            current_var_gap = np.abs((sigma.diagonal()).sum() - sum_eigen_val1)

            current_fid = calculate_frechet_distance(m1, s1, mu, sigma)

            cluster_fid.append(current_fid)
            cluster_div.append(current_div)
            cluster_var_gap.append(current_var_gap)

    
        np.save(result_dir + '/cluster_fid_div_by_'+str(c_num)+'_img.npy', np.c_[np.array(cluster_fid), np.array(cluster_div)])
    else:
        cluster_fid_var=np.load(result_dir + '/cluster_fid_div_by_'+str(c_num)+'_img.npy')
        cluster_fid=cluster_fid_var[:,0]
        cluster_div=cluster_fid_var[:,1]
    
    
    cluster_fida = np.array(cluster_fid)
    score_fid = scipy.special.softmax(-cluster_fida)
    sample_rate = score_fid

    c_num_len = []
    id_score = []
    for kk in range(c_num):
        c_num_len_k = np.sum (label_pred == kk)
        c_num_len.append (c_num_len_k)

    for jj in range(len(label_pred)):
        id_score.append( sample_rate[label_pred[jj]] / c_num_len[label_pred[jj]])


    if True:
        lowest_fd = float('inf')
        lowest_img_list = []
        if not os.path.exists(result_dir + '/domain_seletive_'+str(c_num)+'_img.npy'):
            cluster_rank = np.argsort(cluster_fida)
            current_list = []
            cluster_feature_aggressive = []
            for k in tqdm(cluster_rank):
                img_list = np.where (label_pred==k)[0]
                initial_feature_infer = feature_infer[label_pred==k]
                cluster_feature_aggressive.extend(initial_feature_infer)
                cluster_feature_aggressive_fixed = cluster_feature_aggressive
                target_feature_fixed = target_feature
                if len (cluster_feature_aggressive) > len (target_feature):
                    cluster_idx = np.random.choice(range(len (cluster_feature_aggressive)), len(target_feature), replace=False)
                    cluster_feature_aggressive_fixed = np.array([cluster_feature_aggressive[ii] for ii in cluster_idx])
                if len (cluster_feature_aggressive) < len (target_feature):
                    cluster_idx = np.random.choice(range(len(target_feature)), len (cluster_feature_aggressive), replace=False)
                    target_feature_fixed = target_feature[cluster_idx]
                mu = np.mean(cluster_feature_aggressive_fixed, axis=0)
                sigma = np.cov(cluster_feature_aggressive_fixed, rowvar=False)
                current_fid = calculate_frechet_distance(m1, s1, mu, sigma)
                current_list.extend (list (img_list))
                print (current_fid)
                if lowest_fd > current_fid:
                    lowest_fd = current_fid
                    lowest_img_list = copy.deepcopy(current_list)
            np.save(result_dir + '/domain_seletive_'+str(c_num)+'_img.npy', lowest_img_list)
        else:
            lowest_img_list = np.load(result_dir + '/domain_seletive_'+str(c_num)+'_img.npy')
        # print (len (lowest_img_list))
        selected_data_ind = lowest_img_list
    
    if len (selected_data_ind) > total_images:
        final_selected_img_ind = list(np.sort(np.random.choice(selected_data_ind, opt.n_num, replace=False)))
    else:
        final_selected_img_ind = selected_data_ind

    result_feature = feature_infer[final_selected_img_ind]
    # print (np.shape (result_feature))

    mu = np.mean(result_feature, axis=0)
    sigma = np.cov(result_feature, rowvar=False)
    current_fid = calculate_frechet_distance(m1, s1, mu, sigma)

    # print (current_fid)
    # print(final_selected_img_ind)


    # get the img_path of the final selected images
    unq_img_paths = np.array(unq_img_paths)
    final_selected_img_path = unq_img_paths[final_selected_img_ind]

    # get the df of the final selected images
    pruned_data = data[data['img_path'].isin(final_selected_img_path)]

    csv_output = 'csv_info/snp_pruned_data.csv'
    pruned_data.to_csv(csv_output, index=False)
    logger.info(f"Pruned data saved to {csv_output}")

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

    c_num = len(glob('/data/tmp_teja/datacv/final/visdrone/VisDrone2019-VID-train/sequences/*'))
    
    c_num = 5

    output_path = 'csv_info/random_pruned_data.csv'
    total_images = 5000
    get_pruned_dataset(csv_paths, output_path, total_images,
                          visdrone_img_paths, c_num)