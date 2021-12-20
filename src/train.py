import glob
import os
import sys
import time

import attr
import numpy as np
import pytz
import torch
import yaml

torch.backends.cudnn.benchmark = True
import datetime
import logging

sys.path.extend(["../", "../.."])

from .model import build_model
from .data_augmentation import gpu_da
from .loss import XEDiceLoss
from .metrics import AverageMeter, Metrics


def train(hps, train_loader, val_loader):
    """
    Trains a single or multi class segmentation model given a hps and a train and val_loader.
    """
    logging.shutdown()
    early_patience = hps.patience * 2 + 2 if hps.patience > 1 else 4

    # Set Saving and Logging path
    curr_time = datetime.datetime.now(pytz.timezone("Europe/Amsterdam")).strftime("%Y-%m-%d_%H-%M-%S")
    trained_models_folder = "trained_models"
    log_path = f"{trained_models_folder}/{hps.name}/fold_{hps.fold_nb}/{curr_time}"

    weights_already_trained = glob.glob(f"{trained_models_folder}/{hps.name}/fold_{hps.fold_nb}/*/*pt")
    if len(weights_already_trained) > 0:
        raise ValueError(
            f"For the experiment {hps.name} there already exist weight paths ({weights_already_trained}). "
            f"Choose a different experiment name or delete the existing weight paths"
        )
    os.makedirs(log_path, exist_ok=True)

    # Initialize logging
    log = logging.getLogger()  # root logger - Good to get it only once.
    for hdlr in log.handlers[:]:  # remove the existing file handlers
        log.removeHandler(hdlr)
    formatter = logging.Formatter("%(asctime)s - %(message)s", "%d-%b-%y %H:%M:%S")
    filehandler = logging.FileHandler(os.path.join(log_path, "logs.log"), "w")
    filehandler.setFormatter(formatter)
    log.addHandler(filehandler)  # set the new handler
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    log.addHandler(streamhandler)
    log.setLevel(logging.INFO)

    # Initialize model, loss, optimizer and lr schedule
    model = build_model(hps)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    logging.info(f"Training with {torch.cuda.device_count()} GPUS")

    loss_func = loss_func = XEDiceLoss(
        alpha=hps.alpha,
        num_classes=hps.num_classes,
        ignore_index=255,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hps.lr * torch.cuda.device_count(), weight_decay=hps.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=hps.patience,
        verbose=True,
        threshold=0.0015,
    )
    model.to(device="cuda")

    if hps.use_fp16:
        scaler = torch.cuda.amp.GradScaler()
        logging.info(f"Training is done with mixed precision")

    save_dict = {"model_name": hps.name, "model_time": curr_time, "hps": attr.asdict(hps)}
    best_metric, best_metric_epoch = -1, 0
    early_stop_counter, best_val_early_stopping_metric, early_stop_flag = 0, 0, False
    start_time, yaml_saved, lrs = time.time(), False, []

    for curr_epoch_num in range(1, hps.n_epochs):
        if curr_epoch_num > 1 and lr < lrs[-1]:
            logging.info(f"LR was reduced to {optimizer.param_groups[0]['lr']:.4e}")
        lrs.append(optimizer.param_groups[0]["lr"])
        model.train()
        torch.set_grad_enabled(True)

        num_batches = hps.num_batches if hps.num_batches else len(train_loader)
        losses, batch_time, data_time = AverageMeter(), AverageMeter(), AverageMeter()
        gpu_da_time = AverageMeter()
        end = time.time()
        for iter_num, data in enumerate(train_loader):
            data_time.update(time.time() - end)
            x_data = data["image"].to(device="cuda", non_blocking=True)
            targets = data["mask"].long().to(device="cuda", non_blocking=True)

            start = time.time()
            if hps.gpu_da_params[0] != 0:
                x_data, targets = gpu_da(x_data, targets, hps.gpu_da_params)
            gpu_da_time.update(time.time() - start)

            optimizer.zero_grad()
            if hps.use_fp16:
                with torch.cuda.amp.autocast():
                    preds = model(x_data)
                    loss = loss_func(preds, targets)

                # Scales loss. Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()
                # Unscales the gradients, then optimizer.step() is called if gradients are not inf/nan,
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
            else:
                preds = model(x_data)
                loss = loss_func(preds, targets)
                loss.backward()
                optimizer.step()

            losses.update(loss.detach().item(), x_data.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if iter_num > 0 and iter_num % hps.print_freq == 0:
                logging.info(
                    f"Ep: [{curr_epoch_num}] B: [{iter_num:}/{num_batches}] TotalT: {(time.time() - start_time) / 60:.1f} min, "
                    f"BatchT: {batch_time.avg:.3f}s, DataT: {data_time.avg:.3f}s, GpuDaT: {gpu_da_time.avg:.3f}s, Loss: {losses.avg:.4f}"
                )
            if iter_num == num_batches:
                break

        logging.info(
            f"Ep: [{curr_epoch_num}] TotalT: {(time.time() - start_time) / 60:.1f} min, "
            f"BatchT: {batch_time.avg:.3f}s, DataT: {data_time.avg:.3f}s, GpuDaT: {gpu_da_time.avg:.3f}s, Loss: {losses.avg:.4f}"
        )
        lr = optimizer.param_groups[0]["lr"]

        ## VAL PHASE
        model.eval()
        torch.set_grad_enabled(False)

        batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
        metrics = Metrics()

        end = time.time()
        for iter_num, data in enumerate(val_loader):
            data_time.update(time.time() - end)
            if iter_num > 1 and data_time.val > 0.5:
                logging.info("Waiting, because Validation DataTime is too high")
                time.sleep(np.random.randint(5, 10))  # to unblock validation dataloader
            x_data = data["image"]
            x_data = x_data.to(device="cuda", non_blocking=True)
            targets = data["mask"].long().to(device="cuda", non_blocking=True)

            if hps.use_fp16:
                with torch.cuda.amp.autocast():
                    preds = model(x_data)
                    loss = loss_func(preds, targets)
            else:
                preds = model(x_data)
                loss = loss_func(preds, targets)
            losses.update(loss.detach().item(), x_data.size(0))

            preds = (torch.softmax(preds, dim=1)[:, 1:] > 0.5) * 1

            # for index in range(len(targets)):
            #     # calculate metrics on all water grps
            metrics.update_metrics(preds, targets)

            batch_time.update(time.time() - end)
            end = time.time()

        # Calculating IoUs in all subgroups
        metrics.calc_ious()

        # Log results
        logging.info(
            f"Ep: [{curr_epoch_num}]  ValT: {(batch_time.avg * len(val_loader)) / 60:.2f} min, BatchT: {batch_time.avg:.3f}s, "
            f"DataT: {data_time.avg:.3f}s, Loss: {losses.avg:.4f}, IoU: {metrics.iou:.4f} (val)"
        )

        # Learning Rate Scheduler
        if curr_epoch_num > 6:
            scheduler.step(metrics.early_stopping_metric)

        # Early Stopping and model saving
        if metrics.early_stopping_metric > best_metric:
            best_metric, best_metric_epoch = metrics.early_stopping_metric, curr_epoch_num
            save_dict.update(
                {
                    "Combination_metric": best_metric,
                    "epoch_num": curr_epoch_num,
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                }
            )

            old_model = glob.glob(f"{log_path}/best_metric*")  # delete old best model
            if len(old_model) > 0:
                os.remove(old_model[0])

            save_path = f"{log_path}/best_metric_{curr_epoch_num}_{best_metric:.4f}.pt"
            torch.save(save_dict, save_path)
            if not yaml_saved:  # save hyperparams once as .yaml
                with open("/".join(save_path.split("/")[:-1]) + "/config.yaml", "w") as file:
                    yaml.dump(attr.asdict(hps), file)
                yaml_saved = True

        if metrics.early_stopping_metric < best_val_early_stopping_metric + 0.0001:
            early_stop_counter += 1
            lr = optimizer.param_groups[0]["lr"]
            if early_stop_counter > early_patience and lr < 5e-5:  # only stop training when lr is low
                logging.info("Early Stopping")
                early_stop_flag = True
        else:
            best_val_early_stopping_metric, early_stop_counter = metrics.early_stopping_metric, 0

        if early_stop_flag:
            break
    torch.cuda.empty_cache()

    logging.info(f"Best validation combination metric of {best_metric:.5f} in epoch {best_metric_epoch}.")
    return best_metric, best_metric_epoch
