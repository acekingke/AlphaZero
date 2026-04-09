# 🐛 AlphaZero Othello 项目 Bug 汇总报告

*生成日期: 2026年4月9日*  
*项目类型: AlphaZero 黑白棋(Othello)训练代码*

---

## 📋 概述

通过静态代码分析和已有Bug报告，汇总了项目中已发现和潜在的问题。

---

## 🚨 新发现的严重Bug (Critical)

### 14. torch.load() 使用了不存在的 `weights_only` 参数 (Critical - ✅ 已修复)
- **影响文件**:
  - `play.py` 第54行
  - `train.py` 第492行
  - `test_fixed_mcts.py` 第20行
  - `test_network.py` 第18, 73行
  - `analyze_checkpoints.py` 第12行
  - `comprehensive_debug.py` 第36行
- **问题代码**:
```python
checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
```
- **原因**: PyTorch的 `torch.load()` 不支持 `weights_only` 参数（这是 `torch.load_state_dict` 的参数）
- **影响**: 这些文件加载模型时会直接崩溃，报 TypeError
- **修复方法**: 移除 `weights_only=True` 参数
- **修复状态**: ✅ 已在所有6个文件中修复

---

## ✅ 已修复的关键Bug

### 1. MCTS价值计算错误 (Critical - 已修复)
- **文件**: `mcts/mcts.py`
- **问题**: 终端状态价值计算使用了错误的玩家引用 `env_copy.board.current_player` 而非搜索开始时的玩家
- **影响**: 训练信号完全颠倒，模型越训练越差
- **修复**: 新增 `self.search_start_player` 属性，在搜索开始时记录玩家，终端价值计算时使用正确引用

### 2. 训练价值标签计算Bug (Critical - 已修复)
- **文件**: `train.py`
- **问题**: 使用了 `winner * player_at_move` 而非正确的胜负标记
- **修复**: 获胜者=1.0，失败者=-1.0，平局=0.0

### 3. 环境Reward计算Bug (Critical - 已修复)
- **文件**: `env/othello.py` - `OthelloEnv.step()`
- **问题**: 在step()中使用移动后的 `current_player` 计算reward
- **修复**: 在移动执行前保存 `player_who_moved`

### 4. Pass动作Reward计算Bug (Critical - 已修复)
- **文件**: `env/othello.py` - `OthelloEnv.step()` pass逻辑
- **问题**: `pass_turn()` 会切换 `current_player`，导致 `player_who_moved` 追踪错误
- **修复**: 在调用 `pass_turn()` 前保存实际的pass玩家

### 5. 状态表示不一致 (Major - 已修复)
- **文件**: `play.py`, `gui_play.py`
- **问题**: 评估时使用 `get_observation()` 而训练时使用canonical states
- **修复**: 统一使用canonical states + `_canonical_to_observation()` 转换

---

## 🔍 潜在Bug和代码问题

### 6. MCTS动作概率数组越界风险 (Medium)
- **文件**: `mcts/mcts.py` 第236-238行
```python
for action, child in root.children.items():
    if 0 <= action < action_space_size:  # 仅检查上限不够
        action_probs[action] = child.visit_count
```
- **问题**: action可能是负数，当前检查不够充分
- **建议**: 已有基础检查，可进一步增强

### 7. OthelloBoard边界检查 (Low)
- **文件**: `env/othello.py`
- **问题**: 部分函数边界检查可能遗漏边缘情况
- **状态**: `_is_valid_move()` 有完整边界检查，但其他函数可能不完整

### 8. 全局变量竞争条件 (Medium)
- **文件**: `train.py` 第28-38行
```python
global WORKER_MODEL
global WORKER_MCTS_CLASS
# ...其他全局变量
```
- **问题**: 多进程环境下全局变量可能导致状态不一致
- **建议**: 使用参数传递替代全局变量

### 9. 随机种子潜在冲突 (Low)
- **文件**: `train.py` 第62-63行, 第70-71行
```python
random.seed(s)
np.random.seed(s + 1)
```
- **问题**: worker ID未加入种子，可能产生相同游戏
- **建议**: 添加worker ID到种子

### 10. MCTS深拷贝内存消耗 (Medium)
- **文件**: `mcts/mcts.py` 第186行
```python
env_copy = copy.deepcopy(env)
```
- **问题**: 每次模拟都深拷贝整个环境，内存消耗大
- **建议**: 使用轻量级状态保存/恢复机制

### 11. 经验缓冲区内存控制 (Medium)
- **文件**: `train.py` 第186行
```python
self.buffer = deque(maxlen=10000000)  # 10M samples
```
- **问题**: 10M样本可能消耗数GB内存
- **建议**: 动态调整缓冲区大小或实现内存监控

### 12. 策略归一化数值稳定性 (Low)
- **文件**: `mcts/mcts.py` 第133-143行
```python
policy_sum = np.sum(masked_policy)
if policy_sum > 0:
    masked_policy /= policy_sum
else:
    # fallback
```
- **问题**: policy_sum接近0但不为0时可能不稳定
- **建议**: 使用 `np.isclose()` 或设置最小阈值

### 13. 浮点运算精度问题 (Low)
- **文件**: `mcts/mcts.py` 第21-23行
```python
if self.visit_count == 0:
    return 0
return self.value_sum / self.visit_count
```
- **问题**: 极小visit_count可能导致数值不稳定
- **建议**: 添加最小值阈值保护

---

## 📊 问题优先级

| 优先级 | Bug编号 | 描述 | 状态 |
|--------|---------|------|------|
| Critical | 1-5, 14 | 已修复的关键bug (含weights_only) | ✅ 已修复 |
| Medium | 6, 8, 10, 11 | 内存/稳定性问题 | ⚠️ 潜在问题 |
| Low | 7, 9, 12, 13 | 边缘情况 | 📝 观察中 |

---

## 🎯 性能影响

### 修复前的表现
- Checkpoint 10 vs random: 5% 胜率
- Checkpoint 20 vs random: 0% 胜率
- 后续checkpoint比早期更差（由于累积的错误训练数据）

### 修复后的预期
- 模型应展现持续性能提升
- 后续checkpoint应优于早期checkpoint

---

## 📝 备注

1. **项目实际游戏**: 本项目是黑白棋(Othello)，而非五子棋(Gomoku)
2. **推荐训练命令**:
```bash
python main.py train --num_iterations 50 --self_play_games 200 --mcts_simulations 256 --use_multiprocessing --mp_num_workers 6 --mp_games_per_worker 5 --use_mps
```
3. **建议**: 由于所有旧checkpoint都是用损坏的reward信号训练的，建议从头开始新鲜训练

---

*报告生成时间: 2026年4月9日*
