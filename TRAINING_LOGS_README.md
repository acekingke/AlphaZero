# AlphaZero 训练日志与可视化功能

## 新增功能

本次更新增加了以下功能：

1. **详细的训练日志记录系统**
   - 记录每轮训练的策略损失、价值损失、总损失
   - 记录训练时间戳和每轮训练持续时间
   - 将日志保存到CSV文件中，便于后续分析

2. **增强的可视化工具**
   - 生成四合一训练图表，包括策略损失、价值损失、总损失和损失随时间变化的曲线
   - 支持查看训练摘要和统计信息
   - 可合并多个训练日志进行分析

3. **辅助脚本和工具**
   - `utils/training_logger.py`: 训练日志记录和可视化的核心模块
   - `utils/visualize_logs.py`: 用于可视化和分析已保存的日志文件

## 使用方法

### 安装依赖

在使用新功能前，请确保安装了必要的依赖项：

```bash
# 方法1：使用提供的安装脚本
./install_dependencies.sh

# 方法2：手动安装
pip install -r requirements.txt
```

### 训练并记录日志

训练过程中会自动记录日志到 `logs` 目录：

```bash
python train.py
```

### 可视化训练日志

训练完成后，可以使用以下命令查看训练结果：

```bash
# 可视化所有训练日志
python utils/visualize_logs.py --log_dir ./logs

# 只查看最新的训练日志
python utils/visualize_logs.py --log_dir ./logs --latest

# 保存图表到指定位置
python utils/visualize_logs.py --log_dir ./logs --save ./my_training_chart.png

# 显示图表窗口
python utils/visualize_logs.py --log_dir ./logs --show
```

## 日志文件格式

日志文件为CSV格式，包含以下字段：
- `iteration`: 迭代次数
- `timestamp`: 时间戳
- `policy_loss`: 策略损失
- `value_loss`: 价值损失
- `total_loss`: 总损失
- `examples`: 本轮使用的训练样本数
- `elapsed_time`: 本轮训练耗时（秒）