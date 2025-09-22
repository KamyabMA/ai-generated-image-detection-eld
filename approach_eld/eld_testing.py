import sys
sys.path.append(".")

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json

from utils.datasets_and_dataloaders import index_dataloader_helper, patchify, CustomDataset, unify_resolutions


def evaluate_model_per_patch(index_dataloader,
                             meta_data_for_index_dataloaders,
                             eval_ids_for_index_dataloaders,
                             patch_size,
                             sampling_method,
                             sampling_number,
                             unify_res_target_area,
                             unify_res_prune_min_threshold,
                             model,
                             device="cpu",
                             threshold=0.5):
    
    # Check if the model is in training mode and if so change it to
    # Evaluation mode. This is needed for dropout and batch norm.
    train_mode = model.training
    if train_mode:
        model.eval()

    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Evaluation:
    lables_list = []
    preds = []
    probs = []
    for iteration, (batch_of_ids) in enumerate(iter(index_dataloader)):
        images, labels = index_dataloader_helper(eval_ids_for_index_dataloaders,
                                                meta_data_for_index_dataloaders,
                                                batch_of_ids,
                                                patch_size,
                                                sampling_method,
                                                sampling_number,
                                                unify_res_target_area=unify_res_target_area,
                                                unify_res_prune_min_threshold=unify_res_prune_min_threshold)
        
        lables_list.append(labels)

        images = images.to(device)
        outputs = model.forward(images)

        outputs = outputs.cpu().detach().numpy()
        for i in range(len(outputs)):
            prediction = outputs[i][0]
            probs.append(prediction)
            if prediction < threshold:
                preds.append(0)
            else:
                preds.append(1)
    preds = np.array(preds)
    labels = torch.cat(lables_list).detach().numpy()

    if device == "cuda":
        torch.cuda.empty_cache()

    # Return the model to it's original state
    if train_mode:
        model.train()

    return {
        "accuracy_score": accuracy_score(labels, preds),
        "precision_score": precision_score(labels, preds),
        "recall_score": recall_score(labels, preds),
        "f1_score": f1_score(labels, preds),
        "roc_auc_score": roc_auc_score(labels, probs)
    }


def eld_inference(image_tensor,
                  patch_size,
                  model,
                  device="cpu",
                  unify_res_target_area=None,
                  unify_res_prune_min_threshold=None):
    
    if unify_res_target_area != None:
        if unify_res_prune_min_threshold != None:
            image_tensor = unify_resolutions(image_tensor, unify_res_target_area, prune=True, prune_min_threshold=unify_res_prune_min_threshold)
        else:
            image_tensor = unify_resolutions(image_tensor, unify_res_target_area)
    if image_tensor == None:
        return None, None
    patches, ys, xs = patchify(image_tensor, patch_size=patch_size, sampling_method="grid")
    patch_coordinates = []
    for y in ys:
        for x in xs:
            patch_coordinates.append((y, x))
    
    train_mode = model.training
    if train_mode:
        model.eval()
    stacked_patches = torch.stack(patches, dim=0)
    stacked_patches = stacked_patches.to(device)
    outputs = model.forward(stacked_patches)
    # Return the model to it's original state
    if train_mode:
        model.train()

    outputs_arr = outputs.cpu().detach().numpy().flatten()

    return outputs_arr, patch_coordinates


def classify_image(image_tensor,
                   patch_size,
                   model,
                   device="cpu",
                   unify_res_target_area=None,
                   unify_res_prune_min_threshold=None):
    """
    real: 0
    fake: 1
    """
    outputs_arr, patch_coordinates = eld_inference(image_tensor, 
                                                   patch_size, 
                                                   model, 
                                                   device,
                                                   unify_res_target_area=unify_res_target_area,
                                                   unify_res_prune_min_threshold=unify_res_prune_min_threshold)
    
    try:
        if outputs_arr == None:
            return None, None, None, None
    except ValueError:
        # classify based on average of outputs per patch
        threshold = 0.5
        prob = outputs_arr.mean()
        if prob < threshold:
            final_prediction = 0
        else:
            final_prediction = 1
            
        return final_prediction, prob, outputs_arr, patch_coordinates


def test_eld(dataset: CustomDataset,
             patch_size,
             model,
             save_path,
             device="cpu",
             unify_res_target_area=None,
             unify_res_prune_min_threshold=None):

    output_dict = {}
    true_labels = []
    pred_labels = []
    probs = []
    for i in range(len(dataset)):
        image, label, id = dataset[i]
        try:
            final_prediction, prob, outputs_arr, patch_coordinates = classify_image(image, 
                                                                            patch_size, 
                                                                            model, 
                                                                            device,
                                                                            unify_res_target_area=unify_res_target_area,
                                                                            unify_res_prune_min_threshold=unify_res_prune_min_threshold)
        except TypeError:
            continue
        if final_prediction == None:
            continue
        true_labels.append(label)
        pred_labels.append(final_prediction)
        probs.append(prob)
        if label == final_prediction:
            correctly_classified = True
        else:
            correctly_classified = False
        output_dict[f"{id}"] = {
            "true_label": label,
            "predicted_label": final_prediction,
            "correctly_classified": correctly_classified,
            "output_per_patch": outputs_arr.tolist(),
            "patch_coordinates": patch_coordinates
        }
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    roc_auc = roc_auc_score(true_labels, probs)
    print(f"(test_eld) Accuracy: {accuracy}")
    output_dict["accuracy_score"] = accuracy
    output_dict["precision_score"] = precision
    output_dict["recall_score"] = recall
    output_dict["f1_score"] = f1
    output_dict["roc_auc_score"] = roc_auc
    json_object = json.dumps(output_dict, indent=4)
    with open(save_path, "w") as outfile:
	    outfile.write(json_object)


if __name__ == "__main__":
    pass

