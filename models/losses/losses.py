import torch
from torch import nn
from torch.nn import functional as F
from basicsr.models.losses.vgg_arch import VGGFeatureExtractor
import numpy as np
from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

class ContrastPreservationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = np.array([[1, -1]])
        self.kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).cuda()

    def forward(self, original_image, generated_image):
        contrast_loss = 0
        for i in range(original_image.shape[1]):
            original_channel = original_image[:, i:i+1, :, :]
            generated_channel = generated_image[:, i:i+1, :, :]
            original_dx = F.conv2d(original_channel, self.kernel, padding=1)
            generated_dx = F.conv2d(generated_channel, self.kernel, padding=1)
            contrast_loss += torch.abs(torch.std(generated_dx) - torch.std(original_dx))
        return contrast_loss

class DLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.contrast_loss = ContrastPreservationLoss()

    def forward(self, L1, R1, im2):
        max_rgb1, _ = torch.max(im2, 1, keepdim=True)
        loss1 = self.mse_loss(L1 * R1, im2) + self.mse_loss(R1, im2 / L1.detach())
        loss3 = self.contrast_loss(L1 * R1, im2) + self.contrast_loss(R1, im2 / L1.detach())
        loss2 = self.mse_loss(L1, max_rgb1)
        return self.loss_weight * (loss1 + loss2 + 0.5 * loss3)

class RLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.contrast_loss = ContrastPreservationLoss()

    def tv_loss(self, illumination):
        grad_h = torch.mean(torch.abs(illumination[:, :, :, :-1] - illumination[:, :, :, 1:]))
        grad_w = torch.mean(torch.abs(illumination[:, :, :-1, :] - illumination[:, :, 1:, :]))
        return grad_h + grad_w

    def forward(self, L1, R1, im2):
        max_rgb1, _ = torch.max(im2, 1, keepdim=True)
        loss1 = self.mse_loss(L1 * R1, im2)
        loss3 = self.contrast_loss(L1 * R1, im2)
        loss2 = self.mse_loss(L1, max_rgb1) + self.tv_loss(L1)
        return self.loss_weight * (loss1 + loss2 + 0.5 * loss3)

class CLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, R1, R2):
        return self.mse_loss(R1, R2)

class BrightnessPreservationLoss(nn.Module):
    def forward(self, original_image, generated_image):
        return torch.abs(torch.mean(generated_image) - torch.mean(original_image))

class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None):
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)

class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1).cuda()
        self.weight = loss_weight

    def conv_gauss(self, img):
        n_channels = img.shape[1]
        pad = self.kernel.shape[-1] // 2
        img = F.pad(img, (pad, pad, pad, pad), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        return current - filtered

    def forward(self, x, y):
        return mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y)) * self.weight

class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights, vgg_type='vgg19', use_input_norm=True, range_norm=True,
                 perceptual_weight=1.0, style_weight=0., criterion='mse'):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'mse':
            self.criterion = nn.MSELoss(reduction='mean')
        else:
            raise NotImplementedError(f'{criterion} not supported.')

    def forward(self, x, gt):
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        percep_loss = sum(
            self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            for k in x_features
        ) * self.perceptual_weight

        return percep_loss, None

class CombinedLoss(nn.Module):
    def __init__(self,
                 r_loss_weight=1.0,
                 c_loss_weight=1.0,
                 brightness_loss_weight=1.0,
                 l1_loss_weight=1.0,
                 p_weight=1.0,
                 edge_loss_weight=50.0,
                 reduction='mean'):
        super().__init__()
        self.r_lossr = RLoss(loss_weight=r_loss_weight, reduction=reduction)
        self.r_lossd = DLoss(loss_weight=r_loss_weight, reduction=reduction)
        self.c_loss = CLoss(reduction=reduction)
        self.brightness_loss = BrightnessPreservationLoss()
        self.l1_loss = L1Loss(loss_weight=l1_loss_weight, reduction=reduction)
        self.edge_loss = EdgeLoss(loss_weight=edge_loss_weight, reduction=reduction)
        self.p_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1},
                                     perceptual_weight=p_weight, criterion='mse').cuda()

        self.r_loss_weight = r_loss_weight
        self.c_loss_weight = c_loss_weight
        self.brightness_loss_weight = brightness_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.edge_loss_weight = edge_loss_weight
        self.p_weight = p_weight

    def forward(self, L1, R1, pred, target, R2, L2, L3, L4):
        r_loss_val = self.r_lossr(L1, R1, target)
        r_loss_val1 = self.r_lossd(L2, R2, target)
        c_loss_val = self.c_loss(R2, R1)
        edge_loss_val = self.edge_loss(R1, R2)
        brightness_val = self.brightness_loss(L1, L2)
        l1_loss_val = self.l1_loss(pred, target)
        p_loss_val = self.p_loss(pred, target)[0]

        total_loss = (
            self.r_loss_weight * r_loss_val +
            self.r_loss_weight * 0.5 * r_loss_val1 +
            self.c_loss_weight * c_loss_val +
            self.edge_loss_weight * edge_loss_val +
            self.p_weight * p_loss_val +
            self.l1_loss_weight * l1_loss_val
        )

        return total_loss
