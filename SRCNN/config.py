# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import numpy as np
import torch
from torch.backends import cudnn


# =========================
# Reproducibility
# =========================
# Set random seed to make the experiment more reproducible
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# =========================
# Basic configuration
# =========================
# Use CPU by default
device = torch.device("cpu")

# Enable cuDNN benchmark mode.
# This can speed up training when input image size is fixed.
cudnn.benchmark = True

# Image magnification factor
upscale_factor = 3

# Running mode: "train" or "test"
mode = "test"

# Experiment name, used for saving weights and log files
exp_name = "SRCNN_x3"


# =========================
# Training configuration
# =========================
if mode == "train":
    # Dataset paths
    train_image_dir = "./data/T91/SRCNN/train"
    test_lr_image_dir = "./data/Set5/GTmod12"
    test_hr_image_dir = "./data/Set5/GTmod12"

    # Data loading settings
    image_size = 32
    batch_size = 16
    num_workers = 4

    # Resume path for incremental or transfer training
    resume = ""

    # Total number of training epochs
    epochs = 58000

    # SGD optimizer parameters
    model_lr = 1e-4
    model_momentum = 0.9
    model_weight_decay = 1e-4
    model_nesterov = False

    # Frequency for printing training logs
    print_frequency = 200


# =========================
# Testing configuration
# =========================
if mode == "test":
    # Test data paths
    lr_dir = f"./data/Set5/GTmod12"
    sr_dir = f"./results/test/{exp_name}"
    hr_dir = f"./data/Set5/GTmod12"

    # Pretrained model path
    model_path = f"results/pretrained_models/srcnn_x{upscale_factor}-T91.pth.tar"
