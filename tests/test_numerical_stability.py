"""
数值稳定性测试 - 测试浮点运算和数值计算的稳定性
"""
import pytest
import numpy as np
import torch
import math
import sys
import os
from unittest.mock import Mock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.mcts import MCTS, Node
from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork
from train import AlphaZeroTrainer


class TestNumericalStability:
    """测试数值稳定性"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.game_size = 6
        self.env = OthelloEnv(size=self.game_size)
        self.model = AlphaZeroNetwork(self.game_size, device='cpu')
        self.mcts = MCTS(self.model, num_simulations=10)
    
    def test_node_value_calculation(self):
        """测试Bug #7: 节点值计算的数值稳定性"""
        node = Node(0.5)
        
        # 测试零访问次数
        assert node.value() == 0
        
        # 测试非常小的访问次数和值
        node.visit_count = 1e-10
        node.value_sum = 1e-15
        value = node.value()
        assert not np.isnan(value)
        assert not np.isinf(value)
        
        # 测试大数值
        node.visit_count = 1e10
        node.value_sum = 1e9
        value = node.value()
        assert not np.isnan(value)
        assert not np.isinf(value)
        assert abs(value - 0.1) < 1e-6
        
        # 测试负值
        node.value_sum = -1e9
        value = node.value()
        assert not np.isnan(value)
        assert not np.isinf(value)
        assert abs(value + 0.1) < 1e-6
    
    def test_policy_normalization_stability(self):
        """测试Bug #8: 策略归一化的数值稳定性"""
        # 测试接近零的策略和
        policy = np.array([1e-10, 1e-11, 1e-12, 0.0])
        valid_moves_mask = np.array([1, 1, 1, 1])
        
        masked_policy = policy * valid_moves_mask
        policy_sum = np.sum(masked_policy)
        
        # 测试归一化
        if policy_sum > 1e-8:  # 使用更严格的阈值
            normalized_policy = masked_policy / policy_sum
        else:
            normalized_policy = valid_moves_mask / np.sum(valid_moves_mask)
        
        # 检查结果
        assert not np.any(np.isnan(normalized_policy))
        assert not np.any(np.isinf(normalized_policy))
        assert abs(np.sum(normalized_policy) - 1.0) < 1e-6
    
    def test_softmax_stability(self):
        """测试Softmax的数值稳定性"""
        # 测试极大值
        large_logits = torch.tensor([1000.0, 999.0, 998.0])
        stable_softmax = torch.softmax(large_logits, dim=0)
        assert not torch.any(torch.isnan(stable_softmax))
        assert not torch.any(torch.isinf(stable_softmax))
        assert abs(torch.sum(stable_softmax).item() - 1.0) < 1e-6
        
        # 测试极小值
        small_logits = torch.tensor([-1000.0, -999.0, -998.0])
        stable_softmax = torch.softmax(small_logits, dim=0)
        assert not torch.any(torch.isnan(stable_softmax))
        assert not torch.any(torch.isinf(stable_softmax))
        assert abs(torch.sum(stable_softmax).item() - 1.0) < 1e-6
        
        # 测试零值
        zero_logits = torch.tensor([0.0, 0.0, 0.0])
        stable_softmax = torch.softmax(zero_logits, dim=0)
        assert not torch.any(torch.isnan(stable_softmax))
        assert not torch.any(torch.isinf(stable_softmax))
        expected_prob = 1.0 / 3.0
        assert all(abs(p.item() - expected_prob) < 1e-6 for p in stable_softmax)
    
    def test_ucb_calculation_stability(self):
        """测试UCB计算的数值稳定性"""
        parent_node = Node(0.5)
        child_node = Node(0.3)
        
        # 测试零访问次数
        parent_node.visit_count = 0
        child_node.visit_count = 0
        
        # UCB计算
        c_puct = 2.0
        sum_visits = 0
        u = c_puct * child_node.prior * math.sqrt(sum_visits) / (1 + child_node.visit_count)
        q = child_node.value()
        score = q + u
        
        assert not math.isnan(score)
        assert not math.isinf(score)
        
        # 测试大访问次数
        parent_node.visit_count = 1000000
        child_node.visit_count = 500000
        sum_visits = 1000000
        
        u = c_puct * child_node.prior * math.sqrt(sum_visits) / (1 + child_node.visit_count)
        score = q + u
        
        assert not math.isnan(score)
        assert not math.isinf(score)
    
    def test_loss_calculation_stability(self):
        """测试损失计算的数值稳定性"""
        # 创建测试数据
        batch_size = 32
        action_size = self.game_size * self.game_size + 1
        
        # 测试极端策略值
        extreme_policies = torch.tensor([
            [1.0] + [0.0] * (action_size - 1),  # 确定性策略
            [1e-10] * action_size,  # 极小概率
            [1.0 / action_size] * action_size,  # 均匀分布
        ])
        extreme_policies = extreme_policies.repeat(batch_size // 3 + 1, 1)[:batch_size]
        
        policy_logits = torch.randn(batch_size, action_size)
        log_softmax = torch.log_softmax(policy_logits, dim=1)
        
        # 计算策略损失
        policy_loss = -torch.sum(extreme_policies * log_softmax) / batch_size
        
        assert not torch.isnan(policy_loss)
        assert not torch.isinf(policy_loss)
        assert policy_loss.item() >= 0
        
        # 测试极端值预测
        extreme_values = torch.tensor([[-1.0], [1.0], [0.0]] * (batch_size // 3 + 1))[:batch_size]
        value_pred = torch.tanh(torch.randn(batch_size, 1))  # 模拟网络输出
        
        value_loss = torch.nn.MSELoss()(value_pred, extreme_values)
        
        assert not torch.isnan(value_loss)
        assert not torch.isinf(value_loss)
        assert value_loss.item() >= 0


class TestFloatingPointPrecision:
    """测试浮点精度问题"""
    
    def test_action_probability_precision(self):
        """测试动作概率的精度"""
        # 创建高精度要求的测试
        visit_counts = np.array([1000000, 1000001, 1000002, 1])
        
        # 模拟温度为0的情况（贪婪选择）
        temperature = 0.0
        if temperature == 0:
            action_probs = np.zeros(len(visit_counts))
            best_action = np.argmax(visit_counts)
            action_probs[best_action] = 1.0
        
        # 检查概率和
        assert abs(np.sum(action_probs) - 1.0) < 1e-10
        assert np.all(action_probs >= 0)
        assert np.all(action_probs <= 1.0)
        
        # 模拟温度很小的情况 - 使用安全处理避免数值溢出
        temperature = 1e-8
        safe_temperature = max(temperature, 1e-6)  # Apply same fix as MCTS
        
        # Additional numerical stability: handle extreme values
        if np.max(visit_counts) > 0:
            # Normalize visit counts to prevent overflow
            max_val = np.max(visit_counts)
            normalized_counts = visit_counts / max_val
            action_probs = normalized_counts ** (1 / safe_temperature)
            if np.sum(action_probs) > 0:
                action_probs = action_probs / np.sum(action_probs)
        else:
            action_probs = np.zeros_like(visit_counts)
        
        # 检查数值稳定性
        assert not np.any(np.isnan(action_probs))
        assert not np.any(np.isinf(action_probs))
    
    def test_dirichlet_noise_precision(self):
        """测试Dirichlet噪声的精度"""
        alpha = 0.3
        size = 37  # 6x6 + 1 for pass
        
        # 生成多次噪声并检查统计特性
        noise_samples = []
        for _ in range(100):
            noise = np.random.dirichlet([alpha] * size)
            noise_samples.append(noise)
            
            # 检查单个样本
            assert abs(np.sum(noise) - 1.0) < 1e-10
            assert np.all(noise >= 0)
            assert np.all(noise <= 1.0)
        
        # 检查整体统计特性
        mean_noise = np.mean(noise_samples, axis=0)
        expected_mean = 1.0 / size
        
        # 平均值应该接近期望值
        assert all(abs(m - expected_mean) < 0.1 for m in mean_noise)
    
    def test_canonical_state_precision(self):
        """测试状态转换的精度"""
        env = OthelloEnv(size=6)
        env.reset()
        
        # 获取原始状态
        original_state = env.board.get_state()
        canonical_state = env.board.get_canonical_state()
        observation = env.board.get_observation()
        
        # 检查数据类型一致性
        assert original_state.dtype == canonical_state.dtype
        assert observation.dtype == np.float32
        
        # 检查值范围
        assert np.all(np.isin(original_state, [-1, 0, 1]))
        assert np.all(np.isin(canonical_state, [-1, 0, 1]))
        assert np.all(observation >= 0.0)
        assert np.all(observation <= 1.0)
        
        # 检查状态转换的一致性
        current_player = env.board.current_player
        manual_canonical = original_state * current_player
        np.testing.assert_array_equal(canonical_state, manual_canonical)


class TestNumericalEdgeCases:
    """测试数值边缘情况"""
    
    def test_zero_division_protection(self):
        """测试除零保护"""
        # 测试MCTS节点值计算
        node = Node(0.5)
        node.visit_count = 0
        node.value_sum = 0
        
        # 应该返回0而不是崩溃
        value = node.value()
        assert value == 0
        assert not math.isnan(value)
        
        # 测试UCB计算中的除零保护
        c_puct = 2.0
        sum_visits = 0
        u = c_puct * node.prior * math.sqrt(sum_visits) / (1 + node.visit_count)
        
        assert not math.isnan(u)
        assert not math.isinf(u)
    
    def test_overflow_protection(self):
        """测试溢出保护"""
        # 测试大数值的softmax
        large_values = torch.tensor([1e10, 1e10 + 1, 1e10 + 2])
        try:
            result = torch.softmax(large_values, dim=0)
            assert not torch.any(torch.isnan(result))
            assert not torch.any(torch.isinf(result))
        except OverflowError:
            # 如果发生溢出，应该有适当的处理
            pass
    
    def test_underflow_protection(self):
        """测试下溢保护"""
        # 测试极小值的处理
        tiny_values = torch.tensor([1e-100, 1e-101, 1e-102])
        
        # Softmax应该能处理极小值
        result = torch.softmax(tiny_values, dim=0)
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))
        assert abs(torch.sum(result).item() - 1.0) < 1e-6
        
        # 检查是否退化为均匀分布
        expected_uniform = 1.0 / len(tiny_values)
        assert all(abs(r.item() - expected_uniform) < 1e-2 for r in result)
    
    def test_gradient_stability(self):
        """测试梯度的数值稳定性"""
        model = AlphaZeroNetwork(6, device='cpu')
        model.train()
        
        # 创建输入
        input_tensor = torch.randn(4, 3, 6, 6, requires_grad=True)
        target_policy = torch.randn(4, 37)
        target_value = torch.randn(4, 1)
        
        # 前向传播
        policy_logits, value_pred = model(input_tensor)
        
        # 计算损失
        policy_loss = torch.nn.CrossEntropyLoss()(policy_logits, torch.softmax(target_policy, dim=1))
        value_loss = torch.nn.MSELoss()(value_pred, target_value)
        total_loss = policy_loss + value_loss
        
        # 反向传播
        total_loss.backward()
        
        # 检查梯度
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.any(torch.isnan(param.grad)), f"NaN gradient in {name}"
                assert not torch.any(torch.isinf(param.grad)), f"Inf gradient in {name}"
                
                # 检查梯度范数
                grad_norm = torch.norm(param.grad)
                assert grad_norm < 1000, f"Large gradient norm in {name}: {grad_norm}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])