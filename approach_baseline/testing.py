import sys
sys.path.append(".")

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_model(dataloader,
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
    for iteration, (images, labels, _) in enumerate(iter(dataloader)):
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
        "roc_auc_score": roc_auc_score(labels,probs)
    }
