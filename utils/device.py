import torch

def get_device(use_mps=True, use_cuda=True, verbose=True):
    """
    获取可用的最佳设备(CUDA, MPS, 或 CPU)
    
    Args:
        use_mps: 是否尝试使用MPS (macOS Metal Performance Shaders)
        use_cuda: 是否尝试使用CUDA (NVIDIA GPU)
        verbose: 是否打印设备信息
    
    Returns:
        torch.device: 可用的最佳设备
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"使用CUDA设备: {torch.cuda.get_device_name(0)}")
    elif use_mps and torch.backends.mps.is_available():
        # MPS仅在macOS 12.3+上可用，且需要PyTorch 1.12+
        device = torch.device("mps")
        if verbose:
            print(f"使用MPS设备 (Apple Silicon加速)")
    else:
        device = torch.device("cpu")
        if verbose:
            print("使用CPU")
    
    return device

def check_mps_availability():
    """
    检查MPS是否可用，并提供详细信息
    
    Returns:
        bool: MPS是否可用
    """
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查MPS可用性
    mps_available = False
    
    try:
        if torch.backends.mps.is_available():
            mps_available = True
            print("✓ MPS可用 - PyTorch可以使用Apple Silicon (M1/M2/M3) GPU加速")
            print("✓ 设备检测成功: MPS设备可用")
        else:
            if torch.backends.mps.is_built():
                print("✗ MPS后端已编译，但设备不可用")
                print("  可能原因: 当前设备不是Apple Silicon (M1/M2/M3) Mac，或macOS版本低于12.3")
            else:
                print("✗ PyTorch未编译MPS支持")
                print("  解决方案: 安装支持MPS的PyTorch版本 (>= 1.12)")
    except AttributeError:
        print("✗ 当前PyTorch版本不支持MPS (需要PyTorch >= 1.12)")
        print("  解决方案: 使用conda或pip升级PyTorch")
    
    print("\n建议设置:")
    if mps_available:
        print("device = torch.device('mps')  # 使用MPS加速")
    else:
        print("device = torch.device('cpu')  # 使用CPU")
    
    return mps_available