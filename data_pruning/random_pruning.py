import pandas as pd

def get_pruned_dataset(csv_paths, output_path, total_images):

    all_data = []

    for csv_path in csv_paths:
        data = pd.read_csv(csv_path)
        all_data.append(data)

    data = pd.concat(all_data)
    unq_img_paths = data['img_path'].unique()

    # sample total_images number of images
    unq_img_paths = unq_img_paths[:total_images]

    # get all the rows with the sampled images
    pruned_data = data[data['img_path'].isin(unq_img_paths)]

    # assert if the unq img_path is equal to the total_images
    assert len(pruned_data['img_path'].unique()) == total_images

    # save the pruned data to a csv file
    pruned_data.to_csv(output_path, index=False)

    print(f"Pruned data saved to {output_path}")


if __name__ == '__main__':
    csv_paths = [
        'csv_info/coco_annotation.csv',
        'csv_info/ade_annotation.csv',
        'csv_info/cityscapes_annotation.csv',
        'csv_info/voc_annotation.csv'
    ]

    output_path = 'csv_info/random_pruned_data.csv'
    total_images = 5000
    get_pruned_dataset(csv_paths, output_path, total_images)