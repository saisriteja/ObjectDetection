# Data Pruning

1. Random data pruning.
2. SNP data pruning.
3. ClipNN based data pruning.

## dataset preparation
1. Download the dataset from the datacv competition
2. Get the JSON( coco format ) and make a csv file with information
```
img_name,labels,ids,x_min,y_min,width,height,img_path
VOC2007/JPEGImages/000007.jpg,1,5,140,49,359,280,/data/tmp_teja/datacv/data/source_pool/voc_train/VOC2007/JPEGImages/000007.jpg
```

3. Prepare the vis drone dataset.
Download the vis drone dataset.

The first one is train and the second is test, from official page of visdrone dataset.
Both of them have the ground truth labels for different categories.

```
gdown 1-BEq--FcjshTF1UwUabby_LHhYj41os5
gdown 1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V
```

```python
python data_pruning/clipNN_pruning.py
python data_pruning/random_pruning.py
python data_pruning/SNP_pruning.py
```