"""
内存管理测试 - 测试内存泄漏和内存使用优化
"""
import pytest
import numpy as np
import torch
import gc
import sys
import os
import copy
from unittest.mock import Mock, patch

# Try to import psutil, skip memory tests if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import AlphaZeroTrainer
from mcts.mcts import MCTS
from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork


class TestMemoryLeaks:
    """测试内存泄漏"""
    
    def setup_method(self):
        """每个测试前的设置"""
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available, skipping memory tests")
        gc.collect()  # 清理垃圾回收
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss
    
    def get_memory_usage(self):
        """获取当前内存使用量"""
        if not PSUTIL_AVAILABLE:
            return 0
        gc.collect()
        return self.process.memory_info().rss
    
    def test_mcts_deep_copy_memory(self):
        """测试Bug #5: MCTS深拷贝内存消耗"""
        game_size = 6
        env = OthelloEnv(size=game_size)
        model = AlphaZeroNetwork(game_size, device='cpu')
        mcts = MCTS(model, num_simulations=10)
        
        memory_before = self.get_memory_usage()
        
        # 执行多次MCTS搜索
        for _ in range(5):
            state = env.reset()
            action_probs = mcts.search(state, env, temperature=1.0)
            assert len(action_probs) == game_size * game_size + 1
        
        memory_after = self.get_memory_usage()
        memory_increase = memory_after - memory_before
        
        # 内存增长应该在合理范围内 (小于50MB)
        max_allowed_increase = 50 * 1024 * 1024
        assert memory_increase < max_allowed_increase, \
            f"Memory increase too large: {memory_increase / 1024 / 1024:.2f}MB"
    
    def test_training_buffer_memory(self):
        """测试Bug #6: 经验缓冲区内存控制"""
        trainer = AlphaZeroTrainer(
            game_size=6,
            num_iterations=1,
            num_self_play_games=5,
            use_multiprocessing=False
        )
        
        memory_before = self.get_memory_usage()
        
        # 模拟添加大量样本到缓冲区
        sample_size = 1000
        sample_examples = []
        
        for i in range(sample_size):
            state = np.random.random((3, 6, 6)).astype(np.float32)
            policy = np.random.random(37).astype(np.float32)
            policy /= np.sum(policy)  # 归一化
            value = np.random.uniform(-1, 1)
            sample_examples.append((state, policy, value))
        
        # 添加到缓冲区
        trainer.buffer.extend(sample_examples)
        
        memory_after = self.get_memory_usage()
        memory_increase = memory_after - memory_before
        
        # 计算每个样本的预期内存使用
        state_memory = 3 * 6 * 6 * 4  # float32
        policy_memory = 37 * 4
        value_memory = 4
        expected_per_sample = state_memory + policy_memory + value_memory
        expected_total = expected_per_sample * sample_size
        
        # 实际内存增长应该在预期的2倍以内（考虑Python对象开销）
        assert memory_increase < expected_total * 2, \
            f"Memory usage too high: {memory_increase} vs expected {expected_total}"
    
    def test_environment_copy_efficiency(self):
        """测试环境拷贝的效率"""
        env = OthelloEnv(size=8)  # 使用较大的棋盘
        
        memory_before = self.get_memory_usage()
        
        # 创建多个环境拷贝
        copies = []
        for _ in range(100):
            env_copy = copy.deepcopy(env)
            copies.append(env_copy)
        
        memory_after = self.get_memory_usage()
        memory_increase = memory_after - memory_before
        
        # 清理
        del copies
        gc.collect()
        
        # 内存增长应该合理（每个拷贝不超过1KB）
        max_allowed = 100 * 1024  # 100KB total
        assert memory_increase < max_allowed, \
            f"Environment copy memory usage too high: {memory_increase} bytes"
    
    def test_model_memory_consistency(self):
        """测试模型内存使用的一致性"""
        game_size = 6
        
        memory_before = self.get_memory_usage()
        
        # 创建和销毁多个模型
        for _ in range(10):
            model = AlphaZeroNetwork(game_size, device='cpu')
            # 使用模型
            test_input = torch.randn(1, 3, game_size, game_size)
            with torch.no_grad():
                _ = model(test_input)
            del model
        
        gc.collect()
        memory_after = self.get_memory_usage()
        memory_increase = memory_after - memory_before
        
        # 创建和销毁模型不应该导致显著的内存增长
        max_allowed_increase = 10 * 1024 * 1024  # 10MB
        assert memory_increase < max_allowed_increase, \
            f"Model creation/destruction memory leak: {memory_increase / 1024 / 1024:.2f}MB"


class TestMemoryOptimization:
    """测试内存优化"""
    
    def test_buffer_size_limits(self):
        """测试缓冲区大小限制"""
        trainer = AlphaZeroTrainer(
            game_size=6,
            use_multiprocessing=False
        )
        
        # 检查缓冲区最大大小设置
        assert hasattr(trainer.buffer, 'maxlen')
        assert trainer.buffer.maxlen is not None
        assert trainer.buffer.maxlen > 0
        
        # 测试缓冲区自动限制
        initial_maxlen = trainer.buffer.maxlen
        
        # 添加超过限制的样本
        sample_count = initial_maxlen + 1000
        for i in range(sample_count):
            state = np.random.random((3, 6, 6)).astype(np.float32)
            policy = np.random.random(37).astype(np.float32)
            value = 0.0
            trainer.buffer.append((state, policy, value))
        
        # 缓冲区大小不应该超过限制
        assert len(trainer.buffer) <= initial_maxlen
    
    def test_tensor_device_optimization(self):
        """测试张量设备优化"""
        game_size = 6
        
        # 测试CPU模型
        model_cpu = AlphaZeroNetwork(game_size, device='cpu')
        input_tensor = torch.randn(1, 3, game_size, game_size)
        
        # 输入应该自动移动到正确设备
        with torch.no_grad():
            output = model_cpu(input_tensor)
        
        assert output[0].device.type == 'cpu'
        assert output[1].device.type == 'cpu'
        
        # 测试设备一致性
        for param in model_cpu.parameters():
            assert param.device.type == 'cpu'
    
    def test_batch_processing_efficiency(self):
        """测试批处理效率"""
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available, skipping memory test")
            
        game_size = 6
        model = AlphaZeroNetwork(game_size, device='cpu')
        
        # 测试不同批大小的内存使用
        batch_sizes = [1, 4, 16, 32]
        memory_usage = []
        
        for batch_size in batch_sizes:
            memory_before = self.get_memory_usage()
            
            # 创建批输入
            batch_input = torch.randn(batch_size, 3, game_size, game_size)
            
            with torch.no_grad():
                _ = model(batch_input)
            
            memory_after = self.get_memory_usage()
            memory_usage.append(memory_after - memory_before)
            
            del batch_input
            gc.collect()
        
        # 内存使用应该随批大小线性增长
        # 检查前两个批大小的比例
        if len(memory_usage) >= 2 and memory_usage[0] > 0:
            ratio = memory_usage[1] / memory_usage[0]
            # 批大小4比批大小1的内存使用应该在2-6倍之间（允许一些开销）
            assert 2 <= ratio <= 6, f"Memory scaling ratio unexpected: {ratio}"
    
    def get_memory_usage(self):
        """获取当前内存使用量"""
        if not PSUTIL_AVAILABLE:
            return 0
        gc.collect()
        process = psutil.Process()
        return process.memory_info().rss


class TestMemoryStress:
    """内存压力测试"""
    
    @pytest.mark.slow
    def test_large_batch_training(self):
        """测试大批量训练的内存稳定性"""
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available, skipping memory test")
            
        trainer = AlphaZeroTrainer(
            game_size=6,
            batch_size=64,  # 较大的批大小
            num_epochs=1,
            use_multiprocessing=False
        )
        
        # 创建大量训练样本
        large_examples = []
        for _ in range(1000):
            state = np.random.random((3, 6, 6)).astype(np.float32)
            policy = np.random.random(37).astype(np.float32)
            policy /= np.sum(policy)
            value = np.random.uniform(-1, 1)
            large_examples.append((state, policy, value))
        
        memory_before = self.get_memory_usage()
        
        # 训练
        try:
            policy_loss, value_loss, total_loss = trainer.train_network(large_examples)
            assert policy_loss >= 0
            assert value_loss >= 0
            assert total_loss >= 0
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip("Out of memory - expected for stress test")
            else:
                raise
        
        memory_after = self.get_memory_usage()
        memory_increase = memory_after - memory_before
        
        # 内存增长应该在合理范围内
        max_allowed = 200 * 1024 * 1024  # 200MB
        assert memory_increase < max_allowed, \
            f"Large batch training memory usage too high: {memory_increase / 1024 / 1024:.2f}MB"
    
    @pytest.mark.slow
    def test_repeated_mcts_searches(self):
        """测试重复MCTS搜索的内存稳定性"""
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available, skipping memory test")
            
        game_size = 6
        env = OthelloEnv(size=game_size)
        model = AlphaZeroNetwork(game_size, device='cpu')
        mcts = MCTS(model, num_simulations=50)
        
        memory_measurements = []
        
        # 执行多次搜索并监控内存
        for i in range(20):
            state = env.reset()
            _ = mcts.search(state, env, temperature=1.0)
            
            if i % 5 == 0:  # 每5次记录一次内存
                memory_measurements.append(self.get_memory_usage())
        
        # 内存使用应该保持相对稳定
        if len(memory_measurements) >= 3:
            memory_growth = memory_measurements[-1] - memory_measurements[0]
            max_allowed_growth = 50 * 1024 * 1024  # 50MB
            assert memory_growth < max_allowed_growth, \
                f"Memory growth in repeated MCTS: {memory_growth / 1024 / 1024:.2f}MB"
    
    def get_memory_usage(self):
        """获取当前内存使用量"""
        if not PSUTIL_AVAILABLE:
            return 0
        gc.collect()
        process = psutil.Process()
        return process.memory_info().rss


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])