import torch
import torch.nn as nn

from ..builder import LOSSES


@LOSSES.register_module()
class InfoNCELoss(nn.Module):

    def forward(self, output, target, target_weight):
        N, K, _, W = output.shape
        heatmaps_reshaped = output.reshape((N, K, -1))
        idx = torch.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
        maxvals = torch.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

        preds = torch.tile(idx, (1, 1, 2))
        preds[:, :, 0] = preds[:, :, 0] % W
        preds[:, :, 1] = preds[:, :, 1] // W

        preds = torch.where(torch.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
        return preds, maxvals
