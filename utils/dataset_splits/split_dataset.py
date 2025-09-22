import sys
sys.path.append(".")

import json
from sklearn.model_selection import train_test_split

from meta_data.meta_data import load_meta_data
from constants import *


def load_data_split(path: str):
    with open(path, "r") as openfile:
        obj = json.load(openfile)
    return obj


def save_data_split(obj, path: str):
    json_object = json.dumps(obj, indent=4)
    with open(path, "w") as outfile:
	    outfile.write(json_object)


def split_data(meta_data_path,
               splits=(0.95, 0.025, 0.025),
               save_path=None):
    """
    spilts: (train_split_percentage, eval_split_percentage, test_split_percentage)
    """
    split_sum = splits[0] + splits[1] + splits[2]
    assert split_sum == 1, f"splits do not sum up to 1 (currently they sum to {split_sum})"
    meta_data_dict = load_meta_data(meta_data_path)
    ids = list(meta_data_dict.keys())

    train_and_eval_ids, test_ids = train_test_split(ids, train_size=splits[0]+splits[1])
    train_ids, eval_ids = train_test_split(train_and_eval_ids, train_size=round(splits[0]*len(ids)))

    print(f"len(train_ids): {len(train_ids)}")
    print(f"len(eval_ids): {len(eval_ids)}")
    print(f"len(test_ids): {len(test_ids)}")
    
    if save_path == None:
        return train_ids, eval_ids, test_ids
    else:
        save_data_split({
             "train_ids": train_ids,
             "eval_ids": eval_ids,
             "test_ids": test_ids
             },
             save_path)


if __name__ == "__main__":
    # split_data(CHAMELEON_META_DATA_PATH,
    #             splits=(0.92, 0.04, 0.04),
    #             save_path="utils/dataset_splits/chameleon_split.json")
    pass