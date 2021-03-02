import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch_dct as dct
IMAGE_SCALE = 2.0/255


def get_kernel(size, nsig, mode='gaussian', device='cuda:0'):
    if mode == 'gaussian':
        # since we have to normlize all the numbers 
        # there is no need to calculate the const number like \pi and \sigma.
        vec = torch.linspace(-nsig, nsig, steps=size).to(device)
        vec = torch.exp(-vec*vec/2)
        res = vec.view(-1, 1) @ vec.view(1, -1) 
        res = res / torch.sum(res)
    elif mode == 'linear':
        # originally, res[i][j] = (1-|i|/(k+1)) * (1-|j|/(k+1))
        # since we have to normalize it
        # calculate res[i][j] = (k+1-|i|)*(k+1-|j|)
        vec = (size+1)/2 - torch.abs(torch.arange(-(size+1)/2, (size+1)/2+1, step=1)).to(device)
        res = vec.view(-1, 1) @ vec.view(1, -1) 
        res = res / torch.sum(res)
    else:
        raise ValueError("no such mode in get_kernel.")
    return res


class NoOpAttacker():
    
    def attack(self, image, label, model):
        return image, -torch.ones_like(label)


class PGDAttacker():
    def __init__(self, num_iter, epsilon, step_size, kernel_size=15, prob_start_from_clean=0.0, translation=False, num_classes=1000, device='cuda:0'):
        step_size = max(step_size, epsilon / num_iter)
        self.num_iter = num_iter
        self.epsilon = epsilon * IMAGE_SCALE
        self.step_size = step_size*IMAGE_SCALE
        self.prob_start_from_clean = prob_start_from_clean
        self.device = device
        self.translation = translation
        self.num_classes = num_classes
        if translation:
            # this is equivalent to deepth wise convolution
            # details can be found in the docs of Conv2d.
            # "When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, this operation is also termed in literature as depthwise convolution."
            self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, stride=(kernel_size-1)//2, bias=False, groups=3).to(self.device)
            self.gkernel = get_kernel(kernel_size, nsig=3, device=self.device).to(self.device)
            self.conv.weight = self.gkernel

    def _create_random_target(self, label):
        label_offset = torch.randint_like(label, low=0, high=self.num_classes)
        return (label + label_offset) % self.num_classes

    def attack(self, image_clean, label, model, original=False, mode='pgd'):
        if mode == 'pgd':
            return self.pgd_attack(image_clean, label, model, original=False)
        elif mode == "dct":
            return self.dct_attack(image_clean, label, model, original=False)
        elif mode == "common":
            return self.common_attack(image_clean, label, model, original=False)
        elif mode == "advbn":
            return self.advbn_attack(image_clean, label, model, original=False)

    def pgd_attack(self, image_clean, label, model, original=False):
        """
        aux_images, _ = self.attacker.attack(x, labels, self._forward_impl)
        """
        if original:
            target_label = label    # untargeted
        else:
            target_label = self._create_random_target(label)    # targeted
        lower_bound = torch.clamp(image_clean - self.epsilon, min=-1., max=1.)
        upper_bound = torch.clamp(image_clean + self.epsilon, min=-1., max=1.)

        ori_images = image_clean.clone().detach()

        init_start = torch.empty_like(image_clean).uniform_(-self.epsilon, self.epsilon)
        
        start_from_noise_index = (torch.randn([])>self.prob_start_from_clean).float() 
        start_adv = image_clean + start_from_noise_index * init_start

        adv = start_adv
        for i in range(self.num_iter):
            adv.requires_grad = True
            logits = model(adv)
            losses = F.cross_entropy(logits, target_label)
            g = torch.autograd.grad(losses, adv, 
                                    retain_graph=False, create_graph=False)[0]
            if self.translation:
                g = self.conv(g)
            # Linf step
            if original:
                adv = adv + torch.sign(g) * self.step_size  # untargeted
            else:
                adv = adv - torch.sign(g) * self.step_size  # targeted
            # Linf project
            adv = torch.where(adv > lower_bound, adv, lower_bound).detach()
            adv = torch.where(adv < upper_bound, adv, upper_bound).detach()
        
        return adv, target_label

    def common_attack(self, image_clean, label, model, original=False):
        """
        aux_images, _ = self.attacker.attack(x, labels, self._forward_impl)
        """
        if original:
            target_label = label  # untargeted
        else:
            target_label = self._create_random_target(label)  # targeted

        ori_images = image_clean.clone().detach()
        B, C, H, W = ori_images.size()
        ones = torch.ones([B, C, 1, 1]).cuda()

        lower_bound = torch.clamp(image_clean - self.epsilon, min=-1., max=1.)
        upper_bound = torch.clamp(image_clean + self.epsilon, min=-1., max=1.)
        sigma_lower_bound = torch.clamp(ones - self.epsilon, min=0., max=2.)
        sigma_upper_bound = torch.clamp(ones + self.epsilon, min=0., max=2.)

        init_start = torch.empty_like(image_clean).uniform_(-self.epsilon, self.epsilon)

        start_from_noise_index = (torch.randn([]) > self.prob_start_from_clean).float()
        start_adv = image_clean + start_from_noise_index * init_start

        adv = start_adv
        adv_sigma = torch.ones([B, C, 1, 1]).cuda()

        for i in range(self.num_iter):
            adv.requires_grad = True
            adv_sigma.requires_grad = True
            logits = model(adv_sigma*image_clean+adv)
            losses = F.cross_entropy(logits, target_label)
            g = torch.autograd.grad(losses, adv,
                                    retain_graph=True, create_graph=True)[0]
            g_sigma = torch.autograd.grad(losses, adv_sigma,
                                    retain_graph=False, create_graph=False)[0]
            if self.translation:
                g = self.conv(g)
            # Linf step
            if original:
                adv = adv + torch.sign(g) * self.step_size  # untargeted
                adv_sigma = adv_sigma + torch.sign(g_sigma) * self.step_size
            else:
                adv = adv - torch.sign(g) * self.step_size  # targeted
                adv_sigma = adv_sigma - torch.sign(g_sigma)
            # Linf project
            adv = torch.where(adv > lower_bound, adv, lower_bound).detach()
            adv = torch.where(adv < upper_bound, adv, upper_bound).detach()
            adv_sigma = torch.where(adv_sigma > sigma_lower_bound, adv_sigma, sigma_lower_bound).detach()
            adv_sigma = torch.where(adv_sigma < sigma_upper_bound, adv_sigma, sigma_upper_bound).detach()

        return adv_sigma*image_clean+adv, target_label

    def advbn_attack(self, image_clean, label, model, original=False):
        """
        aux_images, _ = self.attacker.attack(x, labels, self._forward_impl)
        """
        if original:
            target_label = label  # untargeted
        else:
            target_label = self._create_random_target(label)  # targeted

        ori_images = image_clean.clone().detach()
        B, C, H, W = ori_images.size()
        ones = torch.ones([B, C, 1, 1]).cuda()

        mean_lower_bound = torch.clamp(ones - self.epsilon, min=0., max=2.)
        mean_upper_bound = torch.clamp(ones + self.epsilon, min=0., max=2.)
        sigma_lower_bound = torch.clamp(ones - self.epsilon, min=0., max=2.)
        sigma_upper_bound = torch.clamp(ones + self.epsilon, min=0., max=2.)

        image_mean = image_clean.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        adv_mean = ones.clone().detach()
        adv_sigma = ones.clone().detach()

        for i in range(self.num_iter):
            adv_mean.requires_grad = True
            adv_sigma.requires_grad = True
            logits = model(adv_sigma * image_clean + adv_mean * image_mean)
            losses = F.cross_entropy(logits, target_label)
            g_mean = torch.autograd.grad(losses, adv_mean,
                                    retain_graph=True, create_graph=True)[0]
            g_sigma = torch.autograd.grad(losses, adv_sigma,
                                          retain_graph=False, create_graph=False)[0]
            # if self.translation:
            #     g = self.conv(g)
            # Linf step
            if original:
                adv_mean = adv_mean + torch.sign(g_mean) * self.step_size  # untargeted
                adv_sigma = adv_sigma + torch.sign(g_sigma) * self.step_size
            else:
                adv_mean = adv_mean - torch.sign(g_mean) * self.step_size  # targeted
                adv_sigma = adv_sigma - torch.sign(g_sigma)
            # Linf project
            adv_mean = torch.where(adv_mean > mean_lower_bound, adv_mean, mean_lower_bound).detach()
            adv_mean = torch.where(adv_mean < mean_upper_bound, adv_mean, mean_upper_bound).detach()
            adv_sigma = torch.where(adv_sigma > sigma_lower_bound, adv_sigma, sigma_lower_bound).detach()
            adv_sigma = torch.where(adv_sigma < sigma_upper_bound, adv_sigma, sigma_upper_bound).detach()

        return adv_sigma * image_clean + adv_mean * image_mean, target_label


    def dct_attack(self, image_clean, label, model, original=False, dct_ratio_low=0.0, dct_ratio_high=1.0):
        """
        dct attack gradient
        # aux_images, _ = self.attacker.dct_attack(x, labels, self._forward_impl,
                                                  dct_ratio_low=self.dct_ratio_low,
                                                  dct_ratio_high=self.dct_ratio_high)
        """
        if original:
            target_label = label  # untargeted
        else:
            target_label = self._create_random_target(label)  # targeted
        lower_bound = torch.clamp(image_clean - self.epsilon, min=-1., max=1.)
        upper_bound = torch.clamp(image_clean + self.epsilon, min=-1., max=1.)

        ori_images = image_clean.clone().detach()

        # frequency domain
        init_start = torch.empty_like(image_clean).uniform_(-self.epsilon, self.epsilon)
        B, C, H, W = image_clean.size()
        # dct_ratio_low bound
        init_start[:, :, :int(dct_ratio_low * H), :int(dct_ratio_low * W)] = 0
        # dct_ratio_high bound
        init_start[:, :, int(dct_ratio_high * H):, int(dct_ratio_high * W):] = 0
        # idct 2d
        init_start = dct.idct_2d(init_start)

        start_from_noise_index = (torch.randn([]) > self.prob_start_from_clean).float()
        start_adv = image_clean + start_from_noise_index * init_start

        adv = start_adv
        for i in range(self.num_iter):
            adv.requires_grad = True
            logits = model(adv)
            losses = F.cross_entropy(logits, target_label)
            g = torch.autograd.grad(losses, adv,
                                    retain_graph=False, create_graph=False)[0]
            # Linf freq project
            dct_g = dct.dct_2d(g)
            dct_g[:, :, :int(dct_ratio_low * H), :int(dct_ratio_low * W):] = 0
            dct_g[:, :, int(dct_ratio_high * H):, int(dct_ratio_high * W):] = 0
            g = dct.idct_2d(dct_g)
            if self.translation:
                g = self.conv(g)
            # Linf step
            if original:
                adv = adv + torch.sign(g) * self.step_size  # untargeted
            else:
                adv = adv - torch.sign(g) * self.step_size  # targeted

            # Linf project
            adv = torch.where(adv > lower_bound, adv, lower_bound)
            adv = torch.where(adv < upper_bound, adv, upper_bound).detach()

        return adv, target_label

