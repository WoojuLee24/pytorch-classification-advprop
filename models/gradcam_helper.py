from PIL import Image
import numpy as np
import torch
import json

from visualization.misc_functions import get_example, save_class_activation_images


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        # self.model.eval()
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


    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        # input image [1, 3, 224, 224]
        # cont_output [1, 256, 13, 13]

        model_output, target_class = self.model(input_image, target_class)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).cuda().zero_()
        one_hot_output[0][int(target_class)] = 1
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.gradient.data.cpu().numpy()
        # Get convolution outputs
        target = self.conv_output.cpu().data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[3],
                       input_image.shape[2]), Image.ANTIALIAS))/255
        return cam

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