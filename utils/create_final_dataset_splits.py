import sys
sys.path.append(".")

import pickle
import numpy as np
import json

from constants import *
from meta_data.meta_data import load_meta_data
from utils.dataset_splits.split_dataset import split_data


image_area_threshold = 150000
sampling_number = 6895
split_percentages = (0.96, 0.02, 0.02) # (train_split_percentage, eval_split_percentage, test_split_percentage)
meta_data_save_path = "meta_data/final_mixed_meta.json"
dataset_split_save_path = "utils/dataset_splits/final_mixed_split.json"

path = "dataset_analysis/all_imgres_info.pkl"
with open(path, 'rb') as handle:
    resolution_information_dict = pickle.load(handle)

def get_indices_with_target_area_larger_than_threshold(l, threshold):
    indices = []
    for i in range(len(l)):
        if l[i] >= threshold:
            indices.append(i)
    return indices

# creating meta data file
meta_data_dict = {}
dataset_names = ["CHAMELEON", "FAKECLUE", "MIRAGENEWS", "RU-AI"]
all_datasets_meta_data_files = [load_meta_data(CHAMELEON_META_DATA_PATH), 
                                load_meta_data(FAKECLUE_META_DATA_PATH), 
                                load_meta_data(MIRAGENEWS_META_DATA_PATH), 
                                load_meta_data(RUAI_META_DATA_PATH)]
for i in range(len(dataset_names)):
    real_ids = resolution_information_dict[dataset_names[i]]["real_dict"]["img_id"]
    fake_ids = resolution_information_dict[dataset_names[i]]["fake_dict"]["img_id"]
    real_areas = resolution_information_dict[dataset_names[i]]["real_dict"]["img_area_list"]
    fake_areas = resolution_information_dict[dataset_names[i]]["fake_dict"]["img_area_list"]
    real_indices_after_filter =  get_indices_with_target_area_larger_than_threshold(real_areas, image_area_threshold)
    fake_indices_after_filter =  get_indices_with_target_area_larger_than_threshold(fake_areas, image_area_threshold)
    real_samples_indices = list(np.random.choice(np.array(real_indices_after_filter), sampling_number, replace=False))
    fake_samples_indices = list(np.random.choice(np.array(fake_indices_after_filter), sampling_number, replace=False))
    for j in real_samples_indices:
        meta_data_dict[real_ids[j]] = all_datasets_meta_data_files[i][real_ids[j]]
    for j in fake_samples_indices:
        meta_data_dict[fake_ids[j]] = all_datasets_meta_data_files[i][fake_ids[j]]
json_object = json.dumps(meta_data_dict, indent=4)
with open(meta_data_save_path, "w") as outfile:
    outfile.write(json_object)

# creating dataset split
split_data(meta_data_save_path, split_percentages, dataset_split_save_path)
