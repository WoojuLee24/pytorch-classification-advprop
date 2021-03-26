import torch
import torch.nn as nn
import torch_dct as dct
import torch.nn.functional as F


class ResStemCifar(nn.Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out, norm_layer):
        super(ResStemCifar, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = norm_layer(w_out)
        self.af = nn.ReLU(inplace=True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResStemCifarSM(nn.Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out, norm_layer):
        super(ResStemCifarSM, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.aux_conv = nn.Conv2d(w_in, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.aux_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.af = nn.ReLU()
        self.sm = SM(w_out, w_out, kernel_size=5, stride=2, groups=w_out)
        self.batch_type = 'clean'

    def forward(self, x):
        if self.batch_type == 'adv':
            x = self.aux_conv(x)
            x = self.aux_bn(x)
            x = self.af(x)
        elif self.batch_type == 'clean':
            x = self.conv(x)
            x = self.bn(x)
            x = self.af(x)
        else:
            assert self.batch_type == 'mix'
            batch_size = x.shape[0]
            x0 = self.conv(x[:batch_size // 2])
            x0 = self.bn(x0)
            x0 = self.af(x0)
            x1 = self.aux_conv(x[batch_size // 2:])
            x1 = self.aux_bn(x1)
            x1 = self.af(x1)
            x1 = self.sm(x1)
            x = torch.cat((x0, x1), 0)

        return x


class ResStemCifarDCT(nn.Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out, norm_layer, dct_ratio_low=0.0, dct_ratio_high=0.25):
        super(ResStemCifarDCT, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.aux_conv = nn.Conv2d(w_in, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.aux_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.af = nn.ReLU()
        self.batch_type = 'clean'
        self.dct_ratio_low = dct_ratio_low
        self.dct_ratio_high = dct_ratio_high

    def forward(self, x):
        if self.batch_type == 'adv':
            x = self.aux_conv(x)
            x = self.aux_bn(x)
            x = self.af(x)
        elif self.batch_type == 'clean':
            x = self.conv(x)
            x = self.bn(x)
            x = self.af(x)
        else:
            assert self.batch_type == 'mix'
            batch_size = x.shape[0]
            x0 = self.conv(x[:batch_size // 2])
            x0 = self.bn(x0)
            x0 = self.af(x0)
            x1 = self.aux_conv(x[batch_size // 2:])
            x1 = self.aux_bn(x1)
            x1 = self.af(x1)
            x1 = dct.dct_2d(x1)
            B, C, H, W = x1.size()
            x1[:, :, :int(self.dct_ratio_low * H), :int(self.dct_ratio_low * W)] = 0
            x1[:, :, int(self.dct_ratio_high * H):, int(self.dct_ratio_high * W):] = 0
            x1 = dct.idct_2d(x1)

            x = torch.cat((x0, x1), 0)

        return x


class ResStemCifarDCT2(nn.Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out, norm_layer, dct_ratio_low=0.0, dct_ratio_high=0.25):
        super(ResStemCifarDCT, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.aux_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.af = nn.ReLU()
        self.batch_type = 'clean'
        self.dct_ratio_low = dct_ratio_low
        self.dct_ratio_high = dct_ratio_high

    def forward(self, x):
        if self.batch_type == 'adv':
            x = self.conv(x)
            x = self.aux_bn(x)
            x = self.af(x)
        elif self.batch_type == 'clean':
            x = self.conv(x)
            x = self.bn(x)
            x = self.af(x)
        else:
            assert self.batch_type == 'mix'
            batch_size = x.shape[0]
            x = self.conv(x)
            x0 = self.bn(x[:batch_size // 2])
            x0 = self.af(x0)
            x1 = self.aux_bn(x[batch_size // 2:])
            x1 = self.af(x1)
            x1 = dct.dct_2d(x1)
            B, C, H, W = x1.size()
            x1[:, :, :int(self.dct_ratio_low * H), :int(self.dct_ratio_low * W)] = 0
            x1[:, :, int(self.dct_ratio_high * H):, int(self.dct_ratio_high * W):] = 0
            x1 = dct.idct_2d(x1)

            x = torch.cat((x0, x1), 0)

        return x



class SM(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5), stride=1, padding=2, dilation=1, bias=True, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.replication_pad = nn.ReplicationPad2d(padding)
        self.param = self.get_param(in_channels, out_channels, kernel_size, groups)

    def get_name(self):
        return type(self).__name__

    def get_param(self, in_channels, out_channels, kernel_size, groups):
        kernel = torch.tensor([[-0.27, -0.23, -0.18, -0.23, -0.27],
                               [-0.23, 0.17, 0.49, 0.17, -0.23],
                               [-0.18, 0.49, 1, 0.49, -0.18],
                               [-0.23, 0.17, 0.49, 0.17, -0.23],
                               [-0.27, -0.23, -0.18, -0.23, -0.27]], requires_grad=False).cuda()
        # kernel = torch.tensor([[-1/8, -1/8, -1/8],
        #                       [-1/8, 1, -1/8],
        #                       [-1/8, -1/8, -1/8]], requires_grad=False).cuda()
        kernel = kernel.repeat((out_channels, in_channels//groups, 1, 1))

        return kernel

    def forward(self, x):
        # x = F.conv2d(x, self.param, stride=self.stride, padding=self.padding, groups=self.groups)
        x = self.reflection_pad(x)
        x = F.conv2d(x, self.param, stride=self.stride, groups=self.groups)
        return x



class MixBatchNorm2d(nn.BatchNorm2d):
    '''
    if the dimensions of the tensors from dataloader is [N, 3, 224, 224]
    that of the inputs of the MixBatchNorm2d should be [2*N, 3, 224, 224].

    If you set batch_type as 'mix', this network will using one batchnorm (main bn) to calculate the features corresponding to[:N, 3, 224, 224],
    while using another batch normalization (auxiliary bn) for the features of [N:, 3, 224, 224].
    During training, the batch_type should be set as 'mix'.

    During validation, we only need the results of the features using some specific batchnormalization.
    if you set batch_type as 'clean', the features are calculated using main bn; if you set it as 'adv', the features are calculated using auxiliary bn.

    Usually, we use to_clean_status, to_adv_status, and to_mix_status to set the batch_type recursively. It should be noticed that the batch_type should be set as 'adv' while attacking.
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MixBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.aux_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                     track_running_stats=track_running_stats)
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv':
            input = self.aux_bn(input)
        elif self.batch_type == 'clean':
            input = super(MixBatchNorm2d, self).forward(input)
        else:
            assert self.batch_type == 'mix'
            batch_size = input.shape[0]
            # input0 = self.aux_bn(input[: batch_size // 2])
            # input1 = super(MixBatchNorm2d, self).forward(input[batch_size // 2:])
            input0 = super(MixBatchNorm2d, self).forward(input[:batch_size // 2])
            input1 = self.aux_bn(input[batch_size // 2:])
            input = torch.cat((input0, input1), 0)
        return input


class MixConv2d(nn.Conv2d):
    '''
    if the dimensions of the tensors from dataloader is [N, 3, 224, 224]
    that of the inputs of the MixBatchNorm2d should be [2*N, 3, 224, 224].

    If you set batch_type as 'mix', this network will using one conv (main bn) to calculate the features corresponding to[:N, 3, 224, 224],
    while using another batch normalization (auxiliary bn) for the features of [N:, 3, 224, 224].
    During training, the batch_type should be set as 'mix'.

    During validation, we only need the results of the features using some specific conv.
    if you set batch_type as 'clean', the features are calculated using main bn; if you set it as 'adv', the features are calculated using auxiliary bn.

    Usually, we use to_clean_status, to_adv_status, and to_mix_status to set the batch_type recursively. It should be noticed that the batch_type should be set as 'adv' while attacking.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1):
        super(MixConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=1)
        self.aux_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=1)
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv':
            input = self.aux_conv(input)
        elif self.batch_type == 'clean':
            input = super(MixConv2d, self).forward(input)
        else:
            assert self.batch_type == 'mix'
            batch_size = input.shape[0]
            input0 = super(MixConv2d, self).forward(input[:batch_size // 2])
            input1 = self.aux_conv(input[batch_size // 2:])
            input = torch.cat((input0, input1), 0)
        return input
