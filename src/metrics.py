import torch


class Metrics:
    """
    Computes and stores segmentation related metrices for training
    """

    def __init__(self) -> None:
        self.tps, self.fps, self.fns, self.iou = 0, 0, 0, 0

    def update_metrics(self, preds, targets):
        tps, fps, fns = tp_fp_fn_with_ignore(preds, targets)
        self.tps += tps
        self.fps += fps
        self.fns += fns

    def calc_ious(self):
        """
        Calculates IoUs per class and biome, mean biome IoUs, penalty and final metric used for early stopping
        """
        self.iou = self.tps / (self.tps + self.fps + self.fns)
        self.early_stopping_metric = self.iou


def tp_fp_fn_with_ignore(preds, targets):
    """
    Calculates True Positives, False Positives and False Negatives ignoring pixels where the target is 255.

    Args:
        preds (float tensor): Prediction tensor
        targets (long tensor): Target tensor
        c_i (int, optional): Class value of target for the positive class. Defaults to 1.

    Returns:
        tps, fps, fns: True Positives, False Positives and False Negatives 
    """
    preds = preds.flatten()
    targets = targets.flatten()

    # ignore missing label pixels
    no_ignore = targets.ne(255)
    preds = preds.masked_select(no_ignore)
    targets = targets.masked_select(no_ignore)

    # calculate TPs/FPs/FNs on all water
    tps = torch.sum(preds * (targets == 1))
    fps = torch.sum(preds) - tps
    fns = torch.sum(targets == 1) - tps

    return tps, fps, fns 


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
