import sys
sys.path.append(".")

import os
import json
import uuid
import pandas as pd
from PIL import Image
import io
from pathlib import Path
from constants import *


def load_meta_data(path: str) -> dict:
    with open(path, "r") as openfile:
        meta_data_dict = json.load(openfile)
    return meta_data_dict


def merge_meta_data(path_list: list[str], save_path: str):
    merged_meta_data = {}
    for path in path_list:
        meta_data = load_meta_data(path)
        merged_meta_data.update(meta_data)
    json_object = json.dumps(merged_meta_data, indent=4)
    with open(save_path, "w") as outfile:
	    outfile.write(json_object)


def init_chameleon_meta_data(data_path=CHAMELEON_DATA_PATH, save_path="meta_data/chameleon_meta.json"):
    meta_data_dict = {}
    for file_name in os.listdir(f"{data_path}/0_real"):
        id = str(uuid.uuid4())
        meta_data_dict[id] = {
             "dataset": "Chameleon",
             "label": "real",
             "file_name": file_name,
             "dir_path": f"{data_path}/0_real"
        }

    for file_name in os.listdir(f"{data_path}/1_fake"):
        id = str(uuid.uuid4())
        meta_data_dict[id] = {
             "dataset": "Chameleon",
             "label": "fake",
             "file_name": file_name,
             "dir_path": f"{data_path}/1_fake"
        }
    json_object = json.dumps(meta_data_dict, indent=4)
    with open(save_path, "w") as outfile:
	    outfile.write(json_object)


if __name__ == "__main__":
    init_chameleon_meta_data()
