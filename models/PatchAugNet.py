import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import List, Tuple, Optional, Any

from libs.pointops.functions import pointops


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size))
        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class _ConvBase(nn.Sequential):

    def __init__(
            self,
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,
            preact=False,
            name=""
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)

        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)


class Conv2d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int] = (1, 1),
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name
        )


# Use kernel size=1x1 Conv2d to implement shared mlp
class SharedMLP(nn.Sequential):

    def __init__(
            self,
            args: List[int],
            *,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = ""
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact
                )
            )


class PointNetDecoder(nn.Module):
    def __init__(self, embedding_size, output_channels=3, num_points=1024):
        super(PointNetDecoder, self).__init__()
        self.num_points = num_points
        self.output_channels = output_channels
        self.fc1 = nn.Linear(embedding_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, num_points * output_channels)

    def forward(self, x):
        """ x: B x C
        """
        batch_size = x.shape[0]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))
        x = x.view(batch_size, self.num_points, self.output_channels)
        x = x.contiguous()
        return x


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)  # B x 256 X 256 x 256 -> B x 256

        if self.add_batch_norm:
            gates = self.bn1(gates)  # B x 256 -> B x 256
        else:
            gates = gates + self.gating_biases  # B x 256 + 256 -> B x 256

        gates = self.sigmoid(gates)  # B x 256 -> B x 256

        activation = x * gates  # B x 256 * B x 256 -> B x 256
        return activation


class NetVLADBase(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True):
        super(NetVLADBase, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size  # K
        self.softmax = nn.Softmax(dim=-1)

        self.cluster_weights = nn.Parameter(
            torch.randn(feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(
            torch.randn(1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(feature_size * cluster_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(
                torch.randn(cluster_size) * 1 / math.sqrt(feature_size))  # attention initialization
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        if gating:
            self.context_gating = GatingContext(output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        x = x.transpose(1, 3).contiguous()  # B x 1024 x N x 1 -> B x 1 x N x 1024
        x = x.view((-1, self.max_samples, self.feature_size))  # B x N x 1024

        activation = torch.matmul(x, self.cluster_weights)  # B x N x 1024 X 1024 x 64 -> B x N x 64
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)  # B x N x 64 -> BN x 64
            activation = self.bn1(activation)  # BN x 64
            activation = activation.view(-1, self.max_samples, self.cluster_size)  # BN x 64 -> B x N x 64
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases  # B x N x 64 + 64 -> B x N x 64

        activation = self.softmax(activation)  # B x N x 64 --(dim=-1)--> B x N x 64

        # activation = activation[:,:,:64]
        activation = activation.view((-1, self.max_samples, self.cluster_size))  # B x N x 64

        a_sum = activation.sum(-2, keepdim=True)  # B x N x K --(dim=-2)--> B x 1 x K
        a = a_sum * self.cluster_weights2  # B x 1 x K X 1 x C x K -> B x C x K
        # element-wise multiply, broadcast mechanism

        activation = torch.transpose(activation, 2, 1)  # B x N x 64 -> B x 64 x N

        x = x.view((-1, self.max_samples, self.feature_size))  # B x N x C -> B x N x C
        vlad = torch.matmul(activation, x)  # B x K x N X B x N x C -> B x K x C
        vlad = torch.transpose(vlad, 2, 1)  # B x K x C -> B x C x K
        vlad = vlad - a  # B x C x K - B x C x K -> B x C x K

        vlad = F.normalize(vlad, dim=1, p=2).contiguous()  # B x C x K -> B x C x K
        return vlad


class MLPAttentionLayer(nn.Module):
    r""" Simple attention layer based on mlp
        Input: B x C x N
        Return: B x C x N
     """
    def __init__(self, channels=None):
        super(MLPAttentionLayer, self).__init__()
        self.mlps = nn.ModuleList()
        for i in range(len(channels)-1):
            mlp_i = nn.Conv1d(channels[i], channels[i+1], 1, bias=False)
            self.mlps.append(mlp_i)
        self.softmax = nn.Softmax(dim=-1)
        self.trans_conv = nn.Conv1d(channels[-1], channels[-1], 1)
        self.after_norm = nn.BatchNorm1d(channels[-1])
        self.act = nn.ReLU()

    def forward(self, x, return_attn=False):
        x_res = x
        for mlp in self.mlps:
            x_res = mlp(x_res)
        x_res = torch.max(x_res, dim=1, keepdim=True)[0]
        x_res = x_res.squeeze(dim=1)  # B x N
        weights = self.softmax(x_res)  # B x N
        weights = weights.unsqueeze(dim=1)  # B x 1 x N
        x_res = x * weights
        x = self.act(x + x_res)  # residual link
        if return_attn:
            return x, weights
        return x


class AdaptiveFeatureAggregator(nn.Module):
    """ B x C_in x K -> B x C_out x 1 """
    def __init__(self, C_in, K, C_out, l2_norm=True):
        """ C_in: channel of input
            K: num of 1xc features
            C_out: channel of output
        """
        super(AdaptiveFeatureAggregator, self).__init__()
        self.mlpa = MLPAttentionLayer(channels=[C_in, C_in])
        self.fc = nn.Linear(C_in * K, C_out)
        self.bn = nn.BatchNorm1d(C_out)
        self.l2_norm = l2_norm

    def forward(self, x):
        x = self.mlpa(x)
        B, C_in, K = x.size()
        x = x.view((B, C_in * K))  # B x C_in x K -> B x C_in*K
        x = self.fc(x)  # B x C_in*K -> B x C_out
        x = self.bn(x)
        if self.l2_norm:
            x = F.normalize(x)
        x = x.unsqueeze(-1)  # B x C_out -> B x C_out x 1
        return x


class SpatialPyramidNetVLAD(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True,
                 aggregation_type=False,
                 add_batch_norm=True):
        super(SpatialPyramidNetVLAD, self).__init__()
        assert len(feature_size) == len(max_samples) == len(cluster_size) == len(output_dim)
        self.vlads = nn.ModuleList()
        for i in range(len(feature_size)):
            vlad_i = NetVLADBase(feature_size[i], max_samples[i], cluster_size[i], output_dim[i], gating, add_batch_norm)
            self.vlads.append(vlad_i)
        # hidden_weights -> MLP(feature_size[0] * sum_cluster_size, output_dim[0])
        sum_cluster_size = 0
        for i in range(len(cluster_size)):
            sum_cluster_size += cluster_size[i]
        self.gating = gating
        if self.gating:
            self.context_gating = GatingContext(output_dim[0], add_batch_norm=add_batch_norm)
        self.aggregation_type = aggregation_type
        if self.aggregation_type == 0:  # not use AdaptiveFeatureAggregator (use FC)
            self.hidden_weights = nn.Parameter(
                torch.randn(feature_size[0] * sum_cluster_size, output_dim[0]) * 1 / math.sqrt(
                    feature_size[0]))  # sum_cluster_size -> 4
            self.bn = nn.BatchNorm1d(output_dim[0])
        elif self.aggregation_type == 1:  # use AdaptiveFeatureAggregator within each scale and between scales
            self.afa_scales = nn.ModuleList()
            for i in range(len(feature_size)):
                afa_scale_i = AdaptiveFeatureAggregator(output_dim[i], cluster_size[i], output_dim[i])
                self.afa_scales.append(afa_scale_i)
            self.afa = AdaptiveFeatureAggregator(output_dim[0], len(feature_size), output_dim[0])
        elif self.aggregation_type == 2:  # use AdaptiveFeatureAggregator cross scales and regions
            self.afa = AdaptiveFeatureAggregator(output_dim[0], sum_cluster_size, output_dim[0])
        elif self.aggregation_type == 4:  # use AdaptiveFeatureAggregator within each scale
            self.afa_scales = nn.ModuleList()
            for i in range(len(feature_size)):
                afa_scale_i = AdaptiveFeatureAggregator(output_dim[i], cluster_size[i], output_dim[i])
                self.afa_scales.append(afa_scale_i)
            self.hidden_weights = nn.Parameter(
                torch.randn(feature_size[0] * len(feature_size), output_dim[0]) * 1 / math.sqrt(
                    feature_size[0]))
            self.bn = nn.BatchNorm1d(output_dim[0])
        elif self.aggregation_type == 5:  # use AdaptiveFeatureAggregator between scales
            self.hidden_weights = nn.ParameterList()
            self.bns = nn.ModuleList()
            for i in range(len(feature_size)):
                hidden_weight_i = nn.Parameter(
                torch.randn(feature_size[i] *  cluster_size[i], output_dim[i]) * 1 / math.sqrt(
                    feature_size[i]))
                self.hidden_weights.append(hidden_weight_i)
                bn_i = nn.BatchNorm1d(output_dim[i])
                self.bns.append(bn_i)
            self.afa = AdaptiveFeatureAggregator(output_dim[0], len(feature_size), output_dim[0])

    def forward(self, features=None):
        if features is None:
            features = []
        # v0: B x C x N0(=128) x 1 -> B x C x K0(=4)
        # v1: B x C x N1(=1024) x 1 -> B x C x K1(=16)
        # v2: B x C x N2(=4096) x 1 -> B x C x K2(=64)
        v_list = []
        for i in range(len(self.vlads)):
            v_i = self.vlads[i](features[i])
            v_list.append(v_i)
        # aggregate vlad features
        if self.aggregation_type == 0:
            v0123 = torch.cat(v_list, dim=-1)  # [B x C x K0, B x C x K1, B x C x K2] -> B x C x (K0+K1+K2)
            B, C, K = v0123.size()
            vlad = v0123.view((B, C * K))  # B x C x (K0+K1+K2) -> B x C*(K0+K1+K2)
            vlad = torch.matmul(vlad, self.hidden_weights)  # B x C*(K0+K1+K2) -> B x C(=256)
            vlad = self.bn(vlad)  # B x C -> B x C
            vlad = F.normalize(vlad)
        elif self.aggregation_type == 1:
            for i in range(len(v_list)):
                v_list[i] = self.afa_scales[i](v_list[i])
            v0123 = torch.cat(v_list, dim=-1)  # [B x C x 1, B x C x 1, B x C x 1] -> B x C x 3
            vlad = self.afa(v0123).squeeze(-1)
        elif self.aggregation_type == 2:
            v0123 = torch.cat(v_list, dim=-1)  # [B x C x K0, B x C x K1, B x C x K2] -> B x C x (K0+K1+K2)
            vlad = self.afa(v0123).squeeze(-1)
        elif self.aggregation_type == 3:  # not use AdaptiveFeatureAggregator (use max pooling)
            v0123 = torch.cat(v_list, dim=-1)  # [B x C x K0, B x C x K1, B x C x K2] -> B x C x (K0+K1+K2)
            vlad = F.max_pool2d(v0123, kernel_size=[1, v0123.size(2)]).squeeze(-1)  # B x C x (K0+K1+K2) -> B x C
            vlad = F.normalize(vlad)
        elif self.aggregation_type == 4:
            for i in range(len(v_list)):
                v_list[i] = self.afa_scales[i](v_list[i])
            v0123 = torch.cat(v_list, dim=-1)  # [B x C x 1, B x C x 1, B x C x 1] -> B x C x 3
            B, C, K = v0123.size()
            vlad = v0123.view((B, C * K))  # B x C x 3 -> B x C*3
            vlad = torch.matmul(vlad, self.hidden_weights)  # B x C*3 -> B x C(=256)
            vlad = self.bn(vlad)  # B x C -> B x C
            vlad = F.normalize(vlad)
        elif self.aggregation_type == 5:
            for i in range(len(v_list)):
                B, C, K = v_list[i].size()
                v_list[i] = v_list[i].view((B, C * K))
                v_list[i] = torch.matmul(v_list[i], self.hidden_weights[i])  # B x C*K -> B x C(=256)
                v_list[i] = self.bns[i](v_list[i])  # B x C -> B x C
                v_list[i] = F.normalize(v_list[i]).unsqueeze(-1)
            v0123 = torch.cat(v_list, dim=-1)  # [B x C x 1, B x C x 1, B x C x 1] -> B x C x 3
            vlad = self.afa(v0123)

        if self.gating:
            vlad = self.context_gating(vlad)  # B x C -> B x C
        return vlad  # B x 256


class PatchAugNet(nn.Module):
    def __init__(self, use_a2a_recon=False, use_l2_norm=False):
        super(PatchAugNet, self).__init__()
        # backbone
        self.backbone = PointNet2()

        # task1: global descriptor
        self.aggregation = SpatialPyramidNetVLAD(
            feature_size=[256,256,256],
            max_samples=[128,1024,4096],
            cluster_size=[4,16,64],
            output_dim=[256,256,256],
            gating=False,
            aggregation_type=2,
            add_batch_norm=True)

        # task2: patch reconstruction
        self.use_l2_norm = use_l2_norm
        self.use_a2a_recon = use_a2a_recon
        if self.use_a2a_recon:
            self.decoder = PointNetDecoder(embedding_size=256, num_points=20)

    def forward(self, batch):
        # task1: global descriptor
        x = batch['cloud']
        # fp_features: [B x 256 x 128, B x 256 x 1024, B x 256 x 4096]
        res = self.backbone(x)
        g_feat = self.aggregation(res['fp_features'])  # Bx256x128, Bx256x1024, Bx256x4096 -> Bx256
        return g_feat


class PointNet2(nn.Module):
    r""" Modified PointNet++ (use EdgeConv and group self-attention module)
    """

    def __init__(self):
        super().__init__()
        c = 3
        sap = [1024,128,16]
        knn = [20,20,20]
        knn_dilation = 2
        gp = 8
        use_xyz = True
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointNet2SAModule(npoint=sap[0], nsample=knn[0], knn_dilation=knn_dilation, gp=gp, mlp=[c, 32, 32, 64],
                              use_xyz=use_xyz))
        self.SA_modules.append(
            PointNet2SAModule(npoint=sap[1], nsample=knn[1], knn_dilation=knn_dilation, gp=gp, mlp=[64, 64, 64, 256],
                              use_xyz=use_xyz))
        self.SA_modules.append(
            PointNet2SAModule(npoint=sap[2], nsample=knn[2], knn_dilation=knn_dilation, gp=gp, mlp=[256, 256, 256, 512],
                              use_xyz=use_xyz))
        fs = [256,256,256]
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointNet2FPModule(mlp=[fs[1] + c, 256, 256, fs[0]]))
        self.FP_modules.append(PointNet2FPModule(mlp=[fs[2] + 64, 256, fs[1]]))
        self.FP_modules.append(PointNet2FPModule(mlp=[512 + 256, 256, fs[2]]))

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        l_xyz = [pointcloud]  # l_xyz[0]: B x N x 3
        l_features = [pointcloud.transpose(1, 2).contiguous()]  # l_features[0]: B x 3 x N
        l_center_idx = []
        l_sample_idx = []
        # set abstraction
        # l0_xyz: B x N(=4096) x 3, l0_feat: B x C(=3) x N(=4096)
        # l1_xyz: B x N(=1024) x 3, l1_feat: B x C(=64) x N(=1024)
        # l2_xyz: B x N(=128) x 3, l2_feat: B x C(=256) x N(=128)
        # l3_xyz: B x N(=16) x 3, l3_feat: B x C(=512) x N(=16)
        for i in range(len(self.SA_modules)):
            li_xyz, li_center_idx, li_sample_idx, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_center_idx.append(li_center_idx)
            l_sample_idx.append(li_sample_idx)
        sa_features = l_features
        # get center idx and sample idx in origin cloud
        l_center_idx_origin = [l_center_idx[0]]
        l_sample_idx_origin = [l_sample_idx[0]]
        for i in range(1, len(l_center_idx)):
            li_center_idx_origin = torch.gather(l_center_idx_origin[i - 1], -1, l_center_idx[i].long())
            temp_l_center_idx_origin = l_center_idx_origin[i - 1].unsqueeze(1)
            temp_l_center_idx_origin = temp_l_center_idx_origin.repeat(1, l_sample_idx[i].shape[1], 1)
            li_sample_idx_origin = torch.gather(temp_l_center_idx_origin, -1, l_sample_idx[i].long())
            l_center_idx_origin.append(li_center_idx_origin)
            l_sample_idx_origin.append(li_sample_idx_origin)
        # feature up sampling and fusion
        # l3: mlp(cat(up_sample(l4_xyz) + l3_xyz)), B x C(=256) x N(=16)
        # l2: mlp(cat(up_sample(l3_xyz) + l2_xyz)), B x C(=256) x N(=128)
        # l1: mlp(cat(up_sample(l2_xyz) + l1_xyz)), B x C(=256) x N(=1024)
        # l0: mlp(cat(up_sample(l1_xyz) + l0_xyz)), B x C(=256) x N(=4096)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        res = {'center_idx_origin': l_center_idx_origin,
               'sample_idx_origin': l_sample_idx_origin,
               'sa_features': [sa_features[1], sa_features[2], sa_features[3]],
               'fp_features': [l_features[2].unsqueeze(-1), l_features[1].unsqueeze(-1), l_features[0].unsqueeze(-1)]}
        return res


class _PointNet2SAModuleBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        #self.sas = None

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[Optional[Any], Any, Tensor, Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        xyz_trans = xyz.transpose(1, 2).contiguous()  # B x 3 x N
        center_idx = pointops.furthestsampling(xyz, self.npoint)  # B x npoint
        # sampled points
        new_xyz = pointops.gathering(
            xyz_trans,
            center_idx
        ).transpose(1, 2).contiguous() if self.npoint is not None else None
        # features of sampled points
        center_features = pointops.gathering(
            features,
            center_idx
        )
        # grouping local features
        sample_idx_list = []  # list of (B , npoint, nsample)
        for i in range(len(self.groupers)):
            new_features, sample_idx = self.groupers[i](xyz, new_xyz, features, center_features)  # B x C x M x K
            new_features = self.mlps[i](new_features)  # B x C' x M x K
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # B x C' x M x 1
            new_features = new_features.squeeze(-1)  # B x C' x M
            # use attention
            #new_features = self.sas[i](new_features)
            new_features_list.append(new_features)
            sample_idx_list.append(sample_idx)
        sample_idx = torch.cat(sample_idx_list, dim=-1)
        return new_xyz, center_idx, sample_idx, torch.cat(new_features_list, dim=1)


class PointNet2SAModuleMSG(_PointNet2SAModuleBase):
    r"""Pointnet set abstraction layer with multiscale grouping
    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query or knn
    mlps : list of int32
        Spec of the pointnet_old before the global max_pool for each scale
    gp: int
        Number of the divided query map in self-attention layer
    bn : bool
        Use batchnorm
    use_xyz: bool
        use xyz only
    """

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], knn_dilation: int,
                 mlps: List[List[int]], gp: int,
                 bn: bool = True, use_xyz: bool = True):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        #self.sas = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointops.QueryAndGroup_Edge(radius, nsample, knn_dilation=knn_dilation, use_xyz=use_xyz,
                                            ret_sample_idx=True)  # EdgeConv
                if npoint is not None else pointops.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(SharedMLP(mlp_spec, bn=bn))


class PointNet2SAModule(PointNet2SAModuleMSG):
    r"""Pointnet set abstraction layer
    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query or knn
    gp: int
        Number of the divided query map in self-attention layer
    mlp : list
        Spec of the pointnet_old before the global max_pool
    bn : bool
        Use batchnorm
    use_xyz: bool
        use xyz only
    """

    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None,
                 knn_dilation: int = 1, gp: int = None,
                 bn: bool = True, use_xyz: bool = True):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], knn_dilation=knn_dilation,
                         gp=gp, bn=bn, use_xyz=use_xyz)


class PointNet2FPModule(nn.Module):
    r"""Propagates the features of one set to another
    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor,
                known_feats: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propagated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propagated
        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointops.nearestneighbor(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = pointops.interpolation(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)
        else:
            new_features = interpolated_feats
        new_features = self.mlp(new_features.unsqueeze(-1)).squeeze(-1)
        return new_features
