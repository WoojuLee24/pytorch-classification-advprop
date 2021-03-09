import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math


class SMNorm(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5), stride=1, padding=(2, 2), dilation=1, bias=False, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.gamma = self.get_param(in_channels, constant=1.0)
        self.beta = self.get_param(in_channels, constant=0.0)
        self.kernel = self.get_kernel(in_channels, out_channels, kernel_size, groups)

    def get_name(self):
        return type(self).__name__

    def get_param(self, channels, constant):
        param = torch.zeros([1, channels, 1, 1], dtype=torch.float, requires_grad=True)
        param = param.cuda()
        # fan_out = kernel_size * kernel_size * out_channels
        # param.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
        nn.init.constant_(param, constant)
        return nn.Parameter(param)

    def get_kernel(self, in_channels, out_channels, kernel_size, groups):
        kernel = torch.tensor([[-0.27, -0.23, -0.18, -0.23, -0.27],
                               [-0.23, 0.17, 0.49, 0.17, -0.23],
                               [-0.18, 0.49, 1, 0.49, -0.18],
                               [-0.23, 0.17, 0.49, 0.17, -0.23],
                               [-0.27, -0.23, -0.18, -0.23, -0.27]], requires_grad=False).cuda()

        kernel = kernel.repeat((out_channels, in_channels // groups, 1, 1))

        return kernel

    def forward(self, x):
        x = F.conv2d(x, self.kernel, stride=self.stride, padding=(2, 2), groups=self.groups)
        x = self.gamma * x + self.beta
        return x


class FixedSM(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5), stride=1, padding=(2, 2), dilation=1, bias=True, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.padding = padding
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
        x = F.conv2d(x, self.param, stride=self.stride, padding=(2, 2), groups=self.groups)
        return x


class FixedHP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1), dilation=1, bias=False, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.param = self.get_param(in_channels, out_channels, kernel_size, groups)

    def get_name(self):
        return type(self).__name__

    def get_param(self, in_channels, out_channels, kernel_size, groups):
        # kernel = torch.tensor([[-0.27, -0.23, -0.18, -0.23, -0.27],
        #                        [-0.23, 0.17, 0.49, 0.17, -0.23],
        #                        [-0.18, 0.49, 1, 0.49, -0.18],
        #                        [-0.23, 0.17, 0.49, 0.17, -0.23],
        #                        [-0.27, -0.23, -0.18, -0.23, -0.27]], requires_grad=False).cuda()
        kernel = torch.tensor([[-1/8, -1/8, -1/8],
                              [-1/8, 1, -1/8],
                              [-1/8, -1/8, -1/8]], requires_grad=False).cuda()
        # kernel = kernel/torch.sum(kernel)
        kernel = kernel.repeat((out_channels, in_channels//groups, 1, 1))

        return kernel

    def forward(self, x):
        # x = F.conv2d(x, self.param, stride=self.stride, padding=self.padding, groups=self.groups)
        x = F.conv2d(x, self.param, stride=self.stride, padding=(1, 1), groups=self.groups)
        return x


class CustomBlurPool(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2, dilation=1,
                 groups=1, bias=False, padding_mode='reflect', sigma=0.8, kernel_norm=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.kernel_norm = kernel_norm
        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.param = self.get_param(self.in_channels, self.out_channels, self.groups, sigma=sigma)
        # self.kernel = self.get_weight(self.param)

    def get_param(self, in_channels, out_channels, groups, sigma=.8):
        param = torch.ones([out_channels, in_channels // groups, 1], dtype=torch.float,
                           requires_grad=False)
        param = param.cuda()
        param *= sigma
        return nn.Parameter(param)

    def get_weight(self, sigma):

        x = self.get_gaussian(sigma, loc=0)
        y = self.get_gaussian(sigma, loc=1)
        if self.kernel_size == 3:
            param = torch.cat([y, x, y], dim=2)
        elif self.kernel_size == 5:
            z = self.get_gaussian(sigma, loc=4)
            param = torch.cat([z, y, x, y, z], dim=2)
        kernel = torch.einsum('bci,bcj->bcij', param, param)
        if self.kernel_norm == True:
            kernel_sum = kernel.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
            kernel = kernel / kernel_sum

        return kernel

    def get_gaussian(self, a, loc):
        return 1 / math.sqrt(2 * math.pi) / a * torch.exp(-loc / 2 / a / a)

    def forward(self, x):
        kernel = self.get_weight(self.param)
        x = self.reflection_pad(x)
        x = F.conv2d(x, kernel, stride=self.stride, groups=self.groups)
        return x


class MixConv2d(nn.Conv2d):
    '''
    if the dimensions of the tensors from dataloader is [N, 3, 224, 224]
    that of the inputs of the MixConv2d should be [2*N, 3, 224, 224].

    If you set batch_type as 'mix', this network will using one batchnorm (main bn) to calculate the features corresponding to[:N, 3, 224, 224],
    while using another batch normalization (auxiliary bn) for the features of [N:, 3, 224, 224].
    During training, the batch_type should be set as 'mix'.

    During validation, we only need the results of the features using some specific batchnormalization.
    if you set batch_type as 'clean', the features are calculated using main bn; if you set it as 'adv', the features are calculated using auxiliary bn.

    Usually, we use to_clean_status, to_adv_status, and to_mix_status to set the batch_type recursively. It should be noticed that the batch_type should be set as 'adv' while attacking.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=False, groups=1):
        super(MixConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         bias=bias)
        self.conv = FixedSM(in_channels, out_channels, kernel_size, stride, padding, dilation, bias, groups)
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'clean':
            pass
        else:
            assert self.batch_type == 'mix'
            batch_size = input.shape[0]
            # input0 = self.aux_bn(input[: batch_size // 2])
            # input1 = super(MixBatchNorm2d, self).forward(input[batch_size // 2:])
            input0 = input[:batch_size // 2]
            input1 = self.conv(input[batch_size // 2:])
            input = torch.cat((input0, input1), 0)
        return input

