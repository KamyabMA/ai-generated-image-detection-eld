import sys
sys.path.append(".")

import os
from dotenv import load_dotenv
import wandb
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
from fastprogress import master_bar, progress_bar

from constants import *
from approach_baseline.testing import evaluate_model


load_dotenv()
WANDB_API_KEY = os.getenv('WANDB_API_KEY')
wandb.login(key=WANDB_API_KEY)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
if device == 'cuda':
    cudnn.benchmark = True


def training_loop(wandb_project_name,
                  train_dataloader,
                  eval_dataloader,
                  model,
                  optimizer,
                  learning_rate,
                  epochs,
                  scheduler=None,
                  device=device,
                  name_appendix=None,
                  save_dir=None):
    
    print(f"Using {device} device.")
    # Construct a name for the experiment.
    model_name = model.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    run_name = f'{model_name}-{optimizer_name}-lr{learning_rate}'
    if name_appendix:
        # run_name += '_' + name_appendix
        run_name = name_appendix
    
    # Check some constants for easy logging
    scheduler_step_in_loop = scheduler is not None and \
                             isinstance(scheduler, (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.LambdaLR))
    iters_per_epoch = len(train_dataloader)

    with wandb.init(project=wandb_project_name, name=run_name) as run:
        # Log some info
        run.config.learning_rate = learning_rate
        run.config.optimizer = optimizer.__class__.__name__
        run.watch(model)

        criterion = nn.BCELoss() # binary cross entropy

        # Make sure the network is in training mode.
        model.train()

        # For complete logging, let's log the first learning rate
        if scheduler is not None:
            run.log({'learning rate': scheduler.get_last_lr()[0]}, commit=False)
        else:
            run.log({'learning rate': learning_rate}, commit=False)
        
        # Using a fastprogress progress bar we can directly see a bit of info
        # about the training, for live updates we look at weights and biases.
        mb = master_bar(range(epochs))
        for epoch in mb:

            # Go over the complete training set in batches.
            for iteration, (images, labels, _) in enumerate(
                progress_bar(iter(train_dataloader), parent=mb)):

                # Move the data to the gpu.
                images, labels = images.to(device), labels.to(device)

                # Set all parameter gradients to zero.
                optimizer.zero_grad()

                # Compute the forward pass.
                outputs = model.forward(images)

                # Compute the loss and propagate the gradients through the
                # network.
                labels = labels.to(torch.float32) # Because the outputs are in float32
                loss = criterion(outputs.flatten(), labels)
                loss.backward()

                # Update the parameters using the selected optimizer.
                optimizer.step()
                
                # Log some stuff to weights and biases, don't commit in the
                # last iteration of the epoch for proper logging.
                final_iteration = iteration == (iters_per_epoch - 1)
                if scheduler:
                    lr = scheduler.get_last_lr()[0]
                else:
                    lr = learning_rate
                run.log({'loss': loss, 'epoch': epoch, 'learning rate': lr},
                        commit=not final_iteration)

                # Check if we are using a learning rate schedule that needs
                # to be updated after every batch. If so also log the learning 
                # rate.
                if scheduler_step_in_loop:
                    scheduler.step()
                
                if hasattr(model, 'step'):
                    model.step()

            # If we are using a learning rate schedule updated only after each
            # epoch, we now take one step in this schedule and log the learning
            # rate. Otherwise we should get the default learning rate and log it.
            if scheduler is not None and not scheduler_step_in_loop:
                scheduler.step()

            # After every epoch, evaluate the model.
            evaluation = evaluate_model(eval_dataloader,
                                      model,
                                      device=device)
            mb.main_bar.comment = f'val acc:{evaluation["accuracy_score"]}'

            # Log the data.
            run.log({'accuracy': evaluation["accuracy_score"]})
            run.log({'precision': evaluation["precision_score"]})
            run.log({'recall': evaluation["recall_score"]})
            run.log({'roc_auc_score': evaluation["roc_auc_score"]})

        if save_dir != None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            state_dict_path = f"{save_dir}/state_dict_epoch={epochs}.pth"
            torch.save(model.state_dict(), state_dict_path)


if __name__ == "__main__":
    pass