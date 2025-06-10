from ast import arg
import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils

from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte
from pdb import set_trace as stx
from skimage import metrics

from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse

# Argument parser for command-line options
parser = argparse.ArgumentParser(description='Image Enhancement using MIRNet-v2')

parser.add_argument('--input_dir', default='/media/Backup/MacNet/Enhancement/Datasets',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/media/Backup/MacNet/result',
                    type=str, help='Directory for results')
parser.add_argument('--opt', type=str, default='/media/Backup/MacNet/Options/MacNet_LOL_v1.yml',
                    help='Path to option YAML file.')
parser.add_argument('--weights', default='/media/Backup/MacNet/pretrained_weights/LOL_v1.pth',
                    type=str, help='Path to weights')
parser.add_argument('--dataset', default='LOL_v1', type=str,
                    help='Test Dataset')
parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')
parser.add_argument('--GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')

args = parser.parse_args()

# Set GPU devices
gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

# Load YAML configuration file
yaml_file = args.opt
weights = args.weights
print(f"dataset {args.dataset}")

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

# Parse config
opt = parse(args.opt, is_train=False)
opt['dist'] = False

x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')  # remove 'type' key as it's not needed later

# Create model
model_restoration = create_model(opt).net_g

# Load model weights
checkpoint = torch.load(weights)
try:
    model_restoration.load_state_dict(checkpoint['params'])
except:
    # If state_dict keys mismatch, adapt the keys
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===>Testing using weights: ", weights)

# Prepare model for evaluation
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# Prepare directories for results
factor = 4
dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
os.makedirs(result_dir, exist_ok=True)

psnr = []
ssim = []

# Get input and target image paths
input_dir = opt['datasets']['val']['dataroot_lq']
target_dir = opt['datasets']['val']['dataroot_gt']
print(input_dir)
print(target_dir)

# Sort image file paths
input_paths = natsorted(glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')))
target_paths = natsorted(glob(os.path.join(target_dir, '*.png')) + glob(os.path.join(target_dir, '*.jpg')))

# Inference loop
with torch.inference_mode():
    for inp_path, tar_path in tqdm(zip(input_paths, target_paths), total=len(target_paths)):

        # Free up GPU memory
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        # Load and normalize images
        img = np.float32(utils.load_img(inp_path)) / 255.
        target = np.float32(utils.load_img(tar_path)) / 255.

        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).cuda()
        target1 = torch.from_numpy(target).permute(2, 0, 1)
        input1 = target1.unsqueeze(0).cuda()

        # Pad image to be multiple of 4
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        # Run model
        restored, a, b, c, d, f, e = model_restoration(input_, input_)

        # Crop to original size
        restored = restored[:, :, :h, :w]

        # Convert to numpy and clamp values
        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        # Optional: Rectify brightness using mean gray-level
        if args.GT_mean:
            mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
            mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
            restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

        # Calculate metrics
        psnr.append(utils.PSNR(target, restored))
        ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored)))

        # Save restored image
        utils.save_img(os.path.join(result_dir, os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png'),
                       img_as_ubyte(restored))

# Report average PSNR and SSIM
psnr = np.mean(np.array(psnr))
ssim = np.mean(np.array(ssim))
print("PSNR: %f " % (psnr))
print("SSIM: %f " % (ssim))
