"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import torch
from torch import nn
from torch.nn import functional as F

from second.pytorch.utils import get_paddings_indicator
from torchplus.nn import Empty
from torchplus.tools import change_default_args


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        self.in_channels = in_channels
        # if use_norm:
        #     BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
        #     Linear = change_default_args(bias=False)(nn.Linear)
        # else:
        #     BatchNorm1d = Empty
        #     Linear = change_default_args(bias=True)(nn.Linear)

        self.linear= nn.Linear(self.in_channels, self.units, bias = False)
        self.norm = nn.BatchNorm2d(self.units, eps=1e-3, momentum=0.01)

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.units, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=1, kernel_size=1, stride=1)

        self.t_conv = nn.ConvTranspose2d(100, 1, (1,8), stride=(1,7))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 34), stride=(1, 1), dilation=(1,3))


    def forward(self, input):
        x = self.conv1(input)
        x = self.norm(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x
        # x = self.linear(input)
        # x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        # x = F.relu(x)
        #
        # x_max = torch.max(x, dim=1, keepdim=True)[0]
        #
        # if self.last_vfe:
        #     return x_max
        # else:
        #     x_repeat = x_max.repeat(1, inputs.shape[1], 1)
        #     x_concatenated = torch.cat([x, x_repeat], dim=2)
        #     return x_concatenated


class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, pillar_x, pillar_y, pillar_z, pillar_i, num_voxels, x_sub_shaped, y_sub_shaped, mask):

        # Find distance of x, y, and z from cluster center
        # pillar_xyz =  torch.cat((pillar_x, pillar_y, pillar_z), 3)
        pillar_xyz =  torch.cat((pillar_x, pillar_y, pillar_z), 1)

        # points_mean = pillar_xyz.sum(dim=2, keepdim=True) / num_voxels.view(1,-1, 1, 1)
        points_mean = pillar_xyz.sum(dim=3, keepdim=True) / num_voxels.view(1, 1, -1, 1)
        f_cluster = pillar_xyz - points_mean
        # Find distance of x, y, and z from pillar center
        #f_center = torch.zeros_like(features[:, :, :2])
        #f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
        #f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)

        f_center_offset_0 = pillar_x - x_sub_shaped
        f_center_offset_1 = pillar_y - y_sub_shaped

        f_center_concat = torch.cat((f_center_offset_0, f_center_offset_1), 1)

        pillar_xyzi = torch.cat((pillar_x, pillar_y, pillar_z, pillar_i), 1)
        features_list = [pillar_xyzi, f_cluster, f_center_concat]

        features = torch.cat(features_list, dim=1)
        masked_features = features * mask

        pillar_feature = self.pfn_layers[0](masked_features)
        return pillar_feature

class PointPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=64,
                 batch_size=2):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features
        self.batch_size = batch_size

    # def forward(self, voxel_features, coords, batch_size):
    def forward(self, voxel_features, coords):
        # batch_canvas will be the final output.
        batch_canvas = []

        if self.batch_size == 1:
            canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                 device=voxel_features.device)
            indices = coords[:, 2] * self.nx + coords[:, 3]
            indices = indices.type(torch.float64)
            transposed_voxel_features = voxel_features.t()

            # Now scatter the blob back to the canvas.
            indices_2d = indices.view(1, -1)
            ones = torch.ones([self.nchannels, 1], dtype=torch.float64, device=voxel_features.device)
            indices_num_channel = torch.mm(ones, indices_2d)
            indices_num_channel = indices_num_channel.type(torch.int64)
            scattered_canvas = canvas.scatter_(1, indices_num_channel, transposed_voxel_features)

            # Append to a list for later stacking.
            batch_canvas.append(scattered_canvas)

            # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
            batch_canvas = torch.stack(batch_canvas, 0)

            # Undo the column stacking to final 4-dim tensor
            batch_canvas = batch_canvas.view(1, self.nchannels, self.ny, self.nx)
            return batch_canvas
        elif self.batch_size == 2:
            first_canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                       device=voxel_features.device)
            # Only include non-empty pillars
            first_batch_mask = coords[:, 0] == 0
            first_this_coords = coords[first_batch_mask, :]
            first_indices = first_this_coords[:, 2] * self.nx + first_this_coords[:, 3]
            first_indices = first_indices.type(torch.long)
            first_voxels = voxel_features[first_batch_mask, :]
            first_voxels = first_voxels.t()

            # Now scatter the blob back to the canvas.
            first_canvas[:, first_indices] = first_voxels

            # Append to a list for later stacking.
            batch_canvas.append(first_canvas)

            # Create the canvas for this sample
            second_canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                        device=voxel_features.device)

            second_batch_mask = coords[:, 0] == 1
            second_this_coords = coords[second_batch_mask, :]
            second_indices = second_this_coords[:, 2] * self.nx + second_this_coords[:, 3]
            second_indices = second_indices.type(torch.long)
            second_voxels = voxel_features[second_batch_mask, :]
            second_voxels = second_voxels.t()

            # Now scatter the blob back to the canvas.
            second_canvas[:, second_indices] = second_voxels

            # Append to a list for later stacking.
            batch_canvas.append(second_canvas)

            # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
            batch_canvas = torch.stack(batch_canvas, 0)

            # Undo the column stacking to final 4-dim tensor
            batch_canvas = batch_canvas.view(2, self.nchannels, self.ny, self.nx)
            return batch_canvas
        else:
            print("Expecting batch size less than 2")
            return 0
