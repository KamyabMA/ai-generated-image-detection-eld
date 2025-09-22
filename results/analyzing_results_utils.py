import numpy as np
import pandas as pd
import re

from constants import *
from meta_data.meta_data import load_meta_data


def get_accuracy_matrix(test_result_save_dir):

    all_datasets_name = ["chameleon",
                         "mirage_news",
                         "fakeclue", 
                         "ru_ai", 
                         "wild_fake"]
    result = {}
    result["TEST"] = []
    for i in range(len(all_datasets_name)):
        result["TEST"].append(f"{all_datasets_name[i]}_TEST")
    for i in range(len(all_datasets_name)):
        result[f"{all_datasets_name[i]}_TRAIN"] = []
        for j in range(len(all_datasets_name)):
            try:
                output_dict_path = f"{test_result_save_dir}/trained_on_{all_datasets_name[i]}_tested_on_{all_datasets_name[j]}_output.json"
                output_dict = load_meta_data(output_dict_path)
            except FileNotFoundError:
                if re.search(r"eld", test_result_save_dir) != None:
                    output_dict_path = f"{test_result_save_dir}/trained_on_{all_datasets_name[i]}_tested_on_{all_datasets_name[j]}_eld_output.json"
                    output_dict = load_meta_data(output_dict_path)
                else:
                    raise
            result[f"{all_datasets_name[i]}_TRAIN"].append(output_dict["accuracy_score"])
    result = pd.DataFrame(result)
    result = result.set_index('TEST')
    result.rename_axis(None, inplace=True)
    return result.transpose()


def average_diagonal(arr_matrix, d=3):
    vals = []
    for i in range(arr_matrix.shape[0]):
        vals.append(arr_matrix[i][i])
    return round(np.mean(vals), d)


def cross_dataset_score(arr_matrix, d=3):
    """
    Average Off-Diagonal Accuracy (AODA)
    """
    vals = []
    for i in range(arr_matrix.shape[0]):
        for j in range(arr_matrix.shape[1]):
            if i != j:
                vals.append(arr_matrix[i][j])
    return round(np.mean(vals), d)


def percentage_change(before, after, d=3):
    return round((after - before)/before, d)


def numpy_to_matrix(arr):
    df = pd.DataFrame(arr)
    df = df.set_axis(["chameleon_TEST", "mirage_news_TEST", "fakeclue_TEST", "ru_ai_TEST", "wild_fake_TEST"], axis=1)
    df = df.set_axis(["chameleon_TRAIN", "mirage_news_TRAIN", "fakeclue_TRAIN", "ru_ai_TRAIN", "wild_fake_TRAIN"], axis=0)
    return df
    