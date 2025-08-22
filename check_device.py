#!/usr/bin/env python3
# 检查设备可用性

import torch
from utils.device import check_mps_availability, get_device

if __name__ == "__main__":
    print("=" * 50)
    print("PyTorch设备检查")
    print("=" * 50)
    
    # 检查MPS可用性
    mps_available = check_mps_availability()
    
    print("\n" + "=" * 50)
    print("获取最佳可用设备:")
    print("=" * 50)
    
    # 获取设备（会自动选择最佳设备）
    device = get_device(verbose=True)
    
    print("\n" + "=" * 50)
    print("系统信息:")
    print("=" * 50)
    import platform
    import sys
    print(f"Python版本: {platform.python_version()}")
    print(f"操作系统: {platform.platform()}")
    print(f"处理器: {platform.processor()}")
    
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        
    print("\n" + "=" * 50)
    
    # 创建一个简单的张量并检查计算
    print("测试张量计算...")
    a = torch.ones(5, 5, device=device)
    b = torch.ones(5, 5, device=device) * 2
    c = a + b
    
    print(f"计算结果 (应为全3张量):")
    print(c)
    print(f"张量设备: {c.device}")
    print("=" * 50)