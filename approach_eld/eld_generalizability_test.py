import sys
sys.path.append(".")

import os
import argparse
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from constants import *
from meta_data.meta_data import load_meta_data
from utils.dataset_splits.split_dataset import load_data_split
from utils.datasets_and_dataloaders import create_index_dataloader, CustomDataset
from models import Model_DINOv2L_1_LN
from approach_eld.eld_training import training_loop
from approach_eld.eld_testing import test_eld


def train_on_one_dataset_test_on_rest(train_dataset: str,
                                      model,
                                      test_result_save_dir: str,
                                      wandb_project_name: str,
                                      wandb_per_run_suffix: str):
    train_meta_data_path = None
    train_data_split_path = None
    all_datasets_name = ["chameleon", 
                         "fakeclue", 
                         "mirage_news", 
                         "ru_ai", 
                         "wild_fake"]
    all_datasets_meta_data_paths = [CHAMELEON_META_DATA_PATH, 
                                    FAKECLUE_META_DATA_PATH, 
                                    MIRAGENEWS_META_DATA_PATH, 
                                    RUAI_META_DATA_PATH, 
                                    WILDFAKE_META_DATA_PATH]
    all_datasets_data_split_paths = [CHAMELEON_14K_SPLIT_PATH,
                                     FAKECLUE_14K_SPLIT_PATH,
                                     MIRAGENEWS_14K_SPLIT_PATH,
                                     RUAI_14K_SPLIT_PATH,
                                     WILDFAKE_14K_SPLIT_PATH]
    if train_dataset == "chameleon":
        train_meta_data_path = CHAMELEON_META_DATA_PATH
        train_data_split_path = CHAMELEON_14K_SPLIT_PATH
        unify_res_target_area = 150000
    elif train_dataset == "fakeclue":
        train_meta_data_path = FAKECLUE_META_DATA_PATH
        train_data_split_path = FAKECLUE_14K_SPLIT_PATH
        unify_res_target_area = 40000
    elif train_dataset == "mirage_news":
        train_meta_data_path = MIRAGENEWS_META_DATA_PATH
        train_data_split_path = MIRAGENEWS_14K_SPLIT_PATH
        unify_res_target_area = 150000
    elif train_dataset == "ru_ai":
        train_meta_data_path = RUAI_META_DATA_PATH
        train_data_split_path = RUAI_14K_SPLIT_PATH
        unify_res_target_area = 150000
    elif train_dataset == "wild_fake":
        train_meta_data_path = WILDFAKE_META_DATA_PATH
        train_data_split_path = WILDFAKE_14K_SPLIT_PATH
        unify_res_target_area = 40000
    else:
        raise Exception(f"{train_dataset} not in {all_datasets_name}.")
    
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    if device == 'cuda':
        cudnn.benchmark = True
    
    train_dataset_split = load_data_split(train_data_split_path)
    train_ids = train_dataset_split["train_ids"]
    eval_ids = train_dataset_split["eval_ids"]

    unify_res_prune_min_threshold = unify_res_target_area
    batch_size = 64
    patch_size = 56
    sampling_method = "uniform"
    sampling_number = 50

    train_index_dataloader = create_index_dataloader(len(train_ids), batch_size=batch_size, shuffle=True)
    eval_index_dataloader = create_index_dataloader(len(eval_ids), batch_size=batch_size, shuffle=False)

    model.to(device)

    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    epochs = 20
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # https://medium.com/@g.martino8/one-cycle-lr-scheduler-a-simple-guide-c3aa9c4cbd9f
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                    max_lr=learning_rate,
                                                    epochs=epochs,
                                                    steps_per_epoch=len(train_index_dataloader),
                                                    pct_start=0.1,
                                                    div_factor=1e2,
                                                    final_div_factor=1e1)
    
    training_loop(wandb_project_name=wandb_project_name,
                  train_index_dataloader=train_index_dataloader,
                  eval_index_dataloader=eval_index_dataloader,
                  meta_data_for_index_dataloaders=load_meta_data(train_meta_data_path),
                  train_ids_for_index_dataloaders=train_ids,
                  eval_ids_for_index_dataloaders=eval_ids,
                  patch_size=patch_size,
                  sampling_method=sampling_method,
                  sampling_number=sampling_number,
                  unify_res_target_area=unify_res_target_area,
                  unify_res_prune_min_threshold=unify_res_prune_min_threshold,
                  model=model,
                  optimizer=optimizer,
                  learning_rate=learning_rate,
                  epochs=epochs,
                  scheduler=scheduler,
                  device=device,
                  name_appendix=f"{train_dataset[:2]}-{wandb_per_run_suffix}",
                  save_dir=f"{test_result_save_dir}/model_checkpoints/trained_on_{train_dataset}")
    
    # PATH = f"{test_result_save_dir}/model_checkpoints/trained_on_{train_dataset}/state_dict_epoch={epochs}.pth"
    # model.load_state_dict(torch.load(PATH, weights_only=True))

    model.eval()
    for i in range(len(all_datasets_name)):
        test_dataset_split = load_data_split(all_datasets_data_split_paths[i])
        test_ids = test_dataset_split["eval_ids"]
        test_set = CustomDataset(test_ids,
                                all_datasets_meta_data_paths[i],
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])
                                    ]))
        
        if not os.path.exists(test_result_save_dir):
            os.makedirs(test_result_save_dir)
        test_result_save = f"{test_result_save_dir}/trained_on_{train_dataset}_tested_on_{all_datasets_name[i]}_eld_output.json"
        test_eld(test_set,
                patch_size,
                model,
                save_path=test_result_save,
                device=device,
                unify_res_target_area=unify_res_target_area,
                unify_res_prune_min_threshold=None)


if __name__ == "__main__":
    model = Model_DINOv2L_1_LN()
    test_result_save_dir = "approach_eld/output/Config_DINOv2L_1_LN/exp_generalizability_test"
    wandb_per_run_suffix = "DINOv2L_1_LN"
    wandb_project_name = "GT_ELD"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_on",
                        type=str,
                        required=True)
    args = parser.parse_args()
    
    SEED = 60754
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    train_on_one_dataset_test_on_rest(args.train_on,
                                      model,
                                      test_result_save_dir,
                                      wandb_project_name,
                                      wandb_per_run_suffix)
    
