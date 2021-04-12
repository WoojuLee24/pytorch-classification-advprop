from PIL import Image
import numpy as np
import torch
import json
import torch.nn.functional as F

from visualization.misc_functions import get_example, save_class_activation_images


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer, attack_iter=1, attack_epsilon=1, attack_step_size=1):
        self.model = model
        self.target_layer = target_layer
        self.attack_iter = attack_iter
        self.attack_epsilon = attack_epsilon
        self.attack_step_size = attack_step_size
        self.model.eval()
        self.hook_layer()

    def hook_layer(self):
        def forward_hook(module, input, output):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = output

        def backward_hook(module, grad_in, grad_output):
            self.gradient = torch.squeeze(grad_output[0])

        # Hook the selected layer
        for n, m in self.model.named_modules():
            if n == str(self.target_layer):
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)


    def generate_adv_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        # input image [1, 3, 224, 224]
        # cont_output [1, 256, 13, 13]

        adv = input_image
        B, C, H, W = input_image.size()
        lower_bound = torch.clamp(input_image - self.attack_epsilon, min=-1., max=1.)
        upper_bound = torch.clamp(input_image + self.attack_epsilon, min=-1., max=1.)
        #         lower_bound = input_image - self.attack_epsilon
        #         upper_bound = input_image + self.attack_epsilon

        for i in range(self.attack_iter):

            model_output, target_class = self.model(adv, target_class)
            if target_class is None:
                target_class = np.argmax(model_output.data.numpy())
            # Target for backprop
            one_hot_output = torch.zeros_like(model_output).cuda()
            one_hot_output = torch.nn.functional.one_hot(targets, num_classes=10)

            # Zero grads
            self.model.zero_grad()
            # Backward pass with specified target
            model_output.backward(gradient=one_hot_output, retain_graph=True)
            # Get hooked gradients
            guided_gradients = self.gradient
            # Get convolution outputs
            target = self.conv_output
            # Get weights from gradients
            weights = guided_gradients.mean(dim=2, keepdim=True).mean(dim=3,
                                                                      keepdim=True)
            # Create empty numpy array for cam
            cam = (target * weights).sum(dim=1, keepdim=True)
            resized_cam = torch.nn.functional.interpolate(cam, (H, W), mode='bicubic')
            resized_cam_mean = resized_cam.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            resized_cam_std = resized_cam.std(dim=2, keepdim=True).std(dim=3, keepdim=True) * attack_epsilon

            resized_cam2 = (resized_cam - resized_cam_mean) / (resized_cam_std + 1e-5)
            # attention_map = F.sigmoid(resized_cam2)
            adv_attention_map = F.sigmoid(-resized_cam2)

            adv = adv * adv_attention_map.repeat(1, 3, 1, 1)

            # Linf project
            adv = torch.where(adv > lower_bound, adv, lower_bound).detach()
            adv = torch.where(adv < upper_bound, adv, upper_bound).detach()

        return adv

if __name__ == '__main__':
    # target layer and label
    target_class = "30"  # black swan 100, bullfrog 30, centipede 79, thunder snake 52
    label_path = "/ws/data/imagenet/imagenet_class_index.json"
    target_layer = "stem.e"  # "stem.conv" "s4.b3.f.b"
    data_path = "/ws/data/imagenet-c/noise/gaussian_noise/3"    # "/ws/data/imagenet/val"

    # load the model
    config.load_cfg_fom_args("Test a trained classification model.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    pretrained_model = setup_model()
    cp.load_checkpoint(cfg.TEST.WEIGHTS, pretrained_model)

    (original_image, prep_img, class_name, jpg) =\
        get_example(target_class, label_path, data_path, target_layer)

    output_path = "/ws/external/visualization_results/grad_cam/" + class_name + "_" + target_layer + "_" + jpg

    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=target_layer)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)
    # Save mask
    save_class_activation_images(original_image, cam, output_path)
    print('Grad cam completed')