import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)

from mmpose.core.evaluation import keypoint_pck_accuracy
from mmpose.core.post_processing import flip_back
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from ..builder import HEADS
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead


@HEADS.register_module()
class TopdownHeatmapInfoNCEHead(TopdownHeatmapBaseHead):
    """Top-down heatmap simple head. paper ref: Bin Xiao et al. ``Simple
    Baselines for Human Pose Estimation and Tracking``.

    TopdownHeatmapSimpleHead is consisted of (>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None,
                 in_index=0,
                 input_transform=None,
                 align_corners=False,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.in_channels = in_channels

        self.length_scale = nn.Parameter(torch.ones(out_channels, requires_grad=True))
        self.linear_decoder = nn.Linear(2 * out_channels, 2 * out_channels)
        self.decoder_loss = build_loss(loss_keypoint)

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[
                -1] if num_deconv_layers > 0 else self.in_channels

            layers = []
            if extra is not None:
                num_conv_layers = extra.get('num_conv_layers', 0)
                num_conv_kernels = extra.get('num_conv_kernels',
                                             [1] * num_conv_layers)

                for i in range(num_conv_layers):
                    layers.append(
                        build_conv_layer(
                            dict(type='Conv2d'),
                            in_channels=conv_channels,
                            out_channels=conv_channels,
                            kernel_size=num_conv_kernels[i],
                            stride=1,
                            padding=(num_conv_kernels[i] - 1) // 2))
                    layers.append(
                        build_norm_layer(dict(type='BN'), conv_channels)[1])
                    layers.append(nn.ReLU(inplace=True))

            layers.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=conv_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding))

            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

    def get_loss(self, output, target, target_weight):
        """Gets the loss."""
        losses = dict()

        N, K, H, W = output.shape
        z = self._derive_keypoints(output)

        # separate frames 1 and frames 2
        # transpose batch and keypoint dims
        # so that we compute inter-sample cdist per keypoint
        z_a = z[::2].transpose(0, 1).contiguous()
        z_b = z[1::2].transpose(0, 1).contiguous()
        N = N // 2

        length_scale = (2 * (self.length_scale ** 2))
        length_scale = length_scale.view(-1, 1, 1).expand(-1, N, N)

        total = - ((torch.cdist(z_a, z_b) ** 2) / length_scale)
        keypoint_log_probs = torch.log_softmax(total, 1)
        log_probs = torch.logsumexp(keypoint_log_probs, 0) - math.log(K)

        nce = torch.sum(torch.diag(log_probs))

        losses['infonce_loss'] = - (nce / N)
        losses['decoder_loss'] = self._decoding_loss(z.detach(), target, target_weight)

        correct = torch.sum(torch.eq(torch.argmax(log_probs, dim=0), torch.arange(0, N)))
        losses['acc_infonce'] = 1. * correct.item() / N
        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """
        N, _, H, W = output.shape
        output = self._derive_keypoints(output)
        output = self._linear_decode(output)

        target = self._get_max_preds(target)[0]

        accuracy = dict()

        if self.target_type == 'GaussianHeatmap':
            _, avg_acc, _ = keypoint_pck_accuracy(
                output.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                target_weight.detach().cpu().numpy().squeeze(-1) > 0,
                0.05,
                np.tile(np.array([[H, W]]), (N, 1)))
            accuracy['acc_pose'] = float(avg_acc)

        return accuracy

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        x = torch.sigmoid(x)
        return x

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[NxKxHxW]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)
        _, _, H, W = output.shape
        output = self._derive_keypoints(output)
        output = self._linear_decode(output)
        output = self._keypoints_to_heatmap(output, H, W)

        if flip_pairs is not None:
            output_heatmap = flip_back(
                output.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.target_type)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
        return output_heatmap

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    @staticmethod
    def _derive_keypoints(output):
        N, K, H, W = output.shape

        xs = F.normalize(output.sum(-2), p=1, dim=-1)
        xs = (xs * torch.arange(0, W, device=xs.device)).sum(-1)

        ys = F.normalize(output.sum(-1), p=1, dim=-1)
        ys = (ys * torch.arange(0, H, device=ys.device)).sum(-1)

        return torch.stack((xs, ys), dim=-1)

    @staticmethod
    def _keypoints_to_heatmap(keypoints, H, W):
        """
        Turns keypoint coordinates into one-hot encoded 2D maps

        Args:
            keypoints (torch.Tensor[NxKx2]): keypoint coordinates
            H (int): image height
            W (int): image width

        Returns:
            torch.Tensor[NxKxHxW]: One-hot encoded 2D map of the keypoints
        """
        N, K, _ = keypoints.shape

        rows, cols = keypoints.long().split(1, -1)

        # find out of bounds keypoints
        oob_rows = (rows < 0) | (rows > (W - 1))
        oob_cols = (cols < 0) | (cols > (H - 1))

        # fake coords to successfully one-hot encode
        rows[oob_rows] = 0
        cols[oob_cols] = 0

        rows = F.one_hot(rows, num_classes=W).to(torch.float)
        cols = F.one_hot(cols, num_classes=H).to(torch.float)

        # zero out, out of bounds keypoints
        rows[oob_rows] = 0.
        cols[oob_cols] = 0.

        res = torch.matmul(cols.transpose(-1, -2), rows)
        return res

    @staticmethod
    def _get_max_preds(heatmaps):
        """Get keypoint predictions from score maps.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            heatmaps (tensor[N, K, H, W]): model predicted heatmaps.

        Returns:
            tuple: A tuple containing aggregated results.

            - preds (tensor[N, K, 2]): Predicted keypoint location.
            - maxvals (tensor[N, K, 1]): Scores (confidence) of the keypoints.
        """
        N, K, _, W = heatmaps.shape
        heatmaps_reshaped = heatmaps.reshape((N, K, -1))
        idx = torch.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
        maxvals = torch.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

        preds = torch.tile(idx, (1, 1, 2))
        preds[:, :, 0] = torch.remainder(preds[:, :, 0], W)
        preds[:, :, 1] = torch.div(preds[:, :, 1], W, rounding_mode='floor')

        preds = torch.where(torch.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
        return preds, maxvals

    def _linear_decode(self, keypoints):
        N, K, D = keypoints.shape
        keypoints = keypoints.reshape(N, -1)
        keypoints = self.linear_decoder(keypoints)
        keypoints = keypoints.reshape(N, K, D)
        return keypoints

    def _decoding_loss(self, keypoints, targets, target_weight):
        keypoints = self._linear_decode(keypoints)
        targets = self._get_max_preds(targets)[0]
        return self.decoder_loss(keypoints, targets, target_weight)
