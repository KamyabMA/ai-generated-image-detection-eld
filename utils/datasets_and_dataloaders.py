from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch
from torchvision.transforms.functional import crop
import random
import numpy as np
import multiprocess as mp
import math

from meta_data.meta_data import load_meta_data
from constants import *


def unify_resolutions(image, 
                      target_area,
                      prune=False,
                      prune_min_threshold=None,
                      prune_max_threshold=None):
    if isinstance(image, Image.Image):
        width = image.size[0]
        height = image.size[1]
    elif isinstance(image, torch.Tensor):
        width = image.shape[2]
        height = image.shape[1]
    else:
        raise Exception(f"Image type {type(image)} is not recognized.")
    area = width*height
    if prune:
        if prune_min_threshold != None:
            if area < prune_min_threshold:
                return None
        if prune_max_threshold != None:
            if area > prune_max_threshold:
                return None
    if area > target_area:
        ratio = target_area/area
        sqrt_ratio = math.sqrt(ratio)
        width = int(round(width*sqrt_ratio))
        height = int(round(height*sqrt_ratio))
        resize = transforms.Resize((height, width))
        image = resize(image)
    return image


class CustomDataset(Dataset):
    """
    real: 0
    fake: 1

    __getitem__ returns tuple of size 3: (image, label, id)
    """
    def __init__(self, 
                 ids: list[str], 
                 meta_data_path: str, 
                 unify_res_target_area=None,
                 unify_res_prune_min_threshold=None,
                 transform=None):
        self.ids = ids
        self.meta_data_path = meta_data_path
        self.unify_res_target_area = unify_res_target_area
        self.unify_res_prune_min_threshold = unify_res_prune_min_threshold
        self.transform = transform
        self.item_list = []

        meta_data = load_meta_data(self.meta_data_path)
        for id in ids:
            item = meta_data[id]
            img_path = f"{item["dir_path"]}/{item["file_name"]}"
            if item["label"] == "real":
                label = 0
            else:
                label = 1
            self.item_list.append({
                "img_path": img_path,
                "label": label,
                "id": id
            })
    
    def __len__(self):
        return len(self.item_list)
    
    def __getitem__(self, idx):
        item = self.item_list[idx]
        img_path = item["img_path"]
        label = item["label"]
        id = item["id"]
        image = Image.open(img_path).convert("RGB")
        if self.unify_res_target_area != None:
            if self.unify_res_prune_min_threshold != None:
                image = unify_resolutions(image, self.unify_res_target_area, prune=True, prune_min_threshold=self.unify_res_prune_min_threshold)
            else:
                image = unify_resolutions(image, self.unify_res_target_area)
        if self.transform != None:
            image = self.transform(image)
        return image, label, id


def create_dataloader(ids: list[str],
                      batch_size,
                      meta_data_path,
                      shuffle: bool=True,
                      num_workers: int=0,
                      unify_res_target_area=None,
                      unify_res_prune_min_threshold=None,
                      transform=transforms.Compose([
                          transforms.Resize(224),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])
                          ])):
    dataset = CustomDataset(ids, 
                            meta_data_path,
                            unify_res_target_area=unify_res_target_area,
                            unify_res_prune_min_threshold=unify_res_prune_min_threshold,
                            transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# Dataset and dataloader for ELD approach: 

def patchify(image, patch_size=56, sampling_method="gaussian", n=50, n_upscale=224):
    """
    if sampling_method = "grid", then n is ignored.
    """
    image_patches = []
    if isinstance(image, Image.Image):
        width = image.size[0]
        height = image.size[1]
    elif isinstance(image, torch.Tensor):
        width = image.shape[2]
        height = image.shape[1]
    else:
        raise Exception(f"Image type {type(image)} is not recognized.")
    if sampling_method == "grid":
        ys = []
        h_mod = height%patch_size
        hstart = int(h_mod/2)
        for i in range(int((height-h_mod)/patch_size)):
            ys.append(hstart + (patch_size*i))
        xs = []
        w_mod = width%patch_size
        wstart = int(w_mod/2)
        for i in range(int((width-w_mod)/patch_size)):
            xs.append(wstart + (patch_size*i))
        for y in ys:
            for x in xs:
                image_crop = crop(image, y, x, patch_size, patch_size) # crop(img, y, x, height, width)
                # upscaling image patch (Karim's suggestion)
                upscale = transforms.Resize((n_upscale, n_upscale))
                image_crop = upscale(image_crop)
                image_patches.append(image_crop)
        return image_patches, ys, xs
    for i in range(n):
        if sampling_method == "uniform":
            y = random.randint(0, height - patch_size)
            x = random.randint(0, width - patch_size)
        elif sampling_method == "gaussian":
            y = int(np.random.normal(height/2, height/4))
            x = int(np.random.normal(width/2, width/4))
            if y < 0: y = 0
            if y > (height - patch_size): y = height - patch_size
            if x < 0: x = 0
            if x > (width - patch_size): x = width - patch_size
        else:
            raise Exception(f"sampling method {sampling_method} is not defined.")
        image_crop = crop(image, y, x, patch_size, patch_size) # crop(img, y, x, height, width)
        # upscaling image patch (Karim's suggestion)
        upscale = transforms.Resize((n_upscale, n_upscale))
        image_crop = upscale(image_crop)
        image_patches.append(image_crop)
    return image_patches, None, None


class IndexDataset(Dataset):
    def __init__(self, n: int):
        self.index_list = list(range(0, n)) # [0, 1, ..., n-1]
    
    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self, idx):
        return self.index_list[idx]


def create_index_dataloader(n: int,
                            batch_size: int, 
                            shuffle=True):
    return DataLoader(IndexDataset(n), batch_size=batch_size, shuffle=shuffle)


def index_dataloader_helper(ids,
                            meta_data,
                            index_batch,
                            patch_size,
                            sampling_method,
                            sampling_number,
                            unify_res_target_area=None,
                            unify_res_prune_min_threshold=None):
    """
    THE SEQUENTIAL VERSION OF THE FUNCTION (NOT PARALLELIZED)

    real: 0
    fake: 1

    returns: stacked image patches (Tensor), labels
    """
    all_patches = []
    all_labels = []
    for i in index_batch:
        item = meta_data[ids[i]]
        # label
        if item["label"] == "real":
            label = 0
        else:
            label = 1
        # image
        img_path = f"{item["dir_path"]}/{item["file_name"]}"
        image = Image.open(img_path).convert("RGB")
        if unify_res_target_area != None:
            if unify_res_prune_min_threshold != None:
                image = unify_resolutions(image, unify_res_target_area, prune=True, prune_min_threshold=unify_res_prune_min_threshold)
            else:
                image = unify_resolutions(image, unify_res_target_area)
        if image == None:
            continue
        transform = transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])
        ])
        image = transform(image)
        image_patches, _, _ = patchify(image,
                                       patch_size=patch_size,
                                       sampling_method=sampling_method,
                                       n=sampling_number)
        all_patches += image_patches
        all_labels += [label]*sampling_number
    return torch.stack(all_patches, dim=0), torch.Tensor(all_labels)

