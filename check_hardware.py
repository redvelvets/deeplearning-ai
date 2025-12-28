#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""简单的GPU检测脚本"""

import torch

print("=" * 50)
print("硬件检测")
print("=" * 50)

# 检测GPU
if torch.cuda.is_available():
    print("✓ GPU 可用")
    print(f"设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    device = torch.device("cuda")
    print(f"\n使用设备: GPU")
else:
    print("✗ GPU 不可用")
    device = torch.device("cpu")
    print(f"\n使用设备: CPU")

print("=" * 50)
