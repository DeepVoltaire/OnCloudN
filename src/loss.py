import torch
import torch.nn as nn

EPS = 1e-7


class XEDiceLoss(nn.Module):
    """
    Mixture of alpha * CrossEntropy and (1 - alpha) * DiceLoss.
    """

    def __init__(self, alpha=0.5, num_classes=1, debug=False, ignore_index=255):
        super().__init__()
        self.xe = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.alpha = alpha
        self.num_classes = num_classes
        self.debug = debug
        self.ignore_index = ignore_index

    def forward(self, preds, targets):
        xe_loss = self.xe(preds, targets)
        dice_loss = 0
        no_ignore = targets.ne(self.ignore_index)
        targets = targets.masked_select(no_ignore)

        preds = torch.softmax(preds, dim=1)
        for j in range(self.num_classes):
            pred = preds[:, j + 1]
            pred = pred.masked_select(no_ignore)
            y_dat = (targets == j + 1).float() if pred.dtype == torch.float32 else (targets == j + 1).half()
            dice_loss += 1 - (2.0 * torch.sum(pred * y_dat)) / (torch.sum(pred + y_dat) + EPS)
            if self.debug:
                print(
                    f"Dice for class {j}: {1 - (2. * torch.sum(pred * y_dat)) / (torch.sum(pred + y_dat) + EPS):.3f}"
                )
        dice_loss /= self.num_classes

        return self.alpha * xe_loss + (1 - self.alpha) * dice_loss

    def get_name(self):
        return "XEDiceLoss"