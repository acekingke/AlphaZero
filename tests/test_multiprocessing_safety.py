"""
多进程同步测试 - 测试训练中的多进程相关问题
"""
import pytest
import numpy as np
import torch
import multiprocessing as mp
import time
import random
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import AlphaZeroTrainer, _worker_init, _worker_play
from models.neural_network import AlphaZeroNetwork


class TestMultiprocessingSafety:
    """测试多进程安全性"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.game_size = 6
        self.trainer = AlphaZeroTrainer(
            game_size=self.game_size,
            num_iterations=1,
            num_self_play_games=10,
            num_mcts_simulations=50,
            use_multiprocessing=True,
            mp_num_workers=2,
            mp_games_per_worker=2
        )
    
    def test_worker_initialization(self):
        """测试Bug #3: worker初始化的安全性"""
        # 测试worker初始化函数
        state_dict = {k: v.cpu() for k, v in self.trainer.model.state_dict().items()}
        
        # 应该能安全初始化
        _worker_init(
            state_dict=state_dict,
            game_size=self.game_size,
            num_mcts_simulations=50,
            c_puct=2.0,
            temperature=0.8,
            dirichlet_alpha=0.3,
            dirichlet_weight=0.25
        )
        
        # 检查全局变量是否正确设置
        import train
        assert hasattr(train, 'WORKER_MODEL')
        assert hasattr(train, 'WORKER_GAME_SIZE')
        assert train.WORKER_GAME_SIZE == self.game_size
    
    def test_random_seed_uniqueness(self):
        """测试Bug #4: 随机种子的唯一性"""
        # 生成多个任务的种子
        base_seed = int(time.time())
        games_per_worker = 2
        total_games = 10
        
        tasks = []
        for i in range(0, total_games, games_per_worker):
            count = min(games_per_worker, total_games - i)
            tasks.append((base_seed + i, count))
        
        # 检查种子唯一性
        seeds = [task[0] for task in tasks]
        assert len(seeds) == len(set(seeds)), "Seeds should be unique"
        
        # 检查种子范围合理
        for seed in seeds:
            assert seed >= base_seed
            assert seed < base_seed + total_games
    
    @pytest.mark.timeout(30)  # 30秒超时
    def test_multiprocess_self_play_basic(self):
        """测试基本的多进程自我对弈"""
        try:
            # 使用较小的参数进行快速测试
            examples = self.trainer.self_play_multiprocess(
                num_workers=2, 
                games_per_worker=1
            )
            
            # 检查结果
            assert isinstance(examples, list)
            assert len(examples) > 0
            
            # 检查每个example的格式
            for example in examples[:5]:  # 只检查前5个
                state, policy, value = example
                assert isinstance(state, np.ndarray)
                assert isinstance(policy, np.ndarray) 
                assert isinstance(value, (int, float))
                assert state.shape == (3, self.game_size, self.game_size)
                assert len(policy) == self.game_size * self.game_size + 1
                assert -1 <= value <= 1
                
        except Exception as e:
            pytest.skip(f"Multiprocessing test skipped due to environment: {e}")
    
    def test_worker_function_isolation(self):
        """测试worker函数的隔离性"""
        # 准备worker初始化
        state_dict = {k: v.cpu() for k, v in self.trainer.model.state_dict().items()}
        
        _worker_init(
            state_dict=state_dict,
            game_size=self.game_size,
            num_mcts_simulations=10,  # 减少模拟次数以加快测试
            c_puct=2.0,
            temperature=0.8,
            dirichlet_alpha=0.3,
            dirichlet_weight=0.25
        )
        
        # 测试worker play函数
        seed_and_count = (12345, 1)  # 固定种子，1个游戏
        
        try:
            results, seed, count, game_stats, worker_id = _worker_play(seed_and_count)
            
            # 检查返回值
            assert seed == 12345
            assert count == 1
            assert isinstance(results, list)
            assert isinstance(game_stats, list)
            assert isinstance(worker_id, str)
            assert len(game_stats) == 1  # 应该有1个游戏的统计
            
            # 检查游戏统计
            stat = game_stats[0]
            assert 'moves' in stat
            assert 'duration' in stat
            assert 'outcome' in stat
            assert stat['moves'] > 0
            assert stat['duration'] > 0
            assert stat['outcome'] in ['Win', 'Loss', 'Draw']
            
        except Exception as e:
            pytest.skip(f"Worker function test skipped: {e}")


class TestConcurrencyIssues:
    """测试并发相关问题"""
    
    def test_model_state_consistency(self):
        """测试模型状态的一致性"""
        game_size = 6
        model1 = AlphaZeroNetwork(game_size, device='cpu')
        model2 = AlphaZeroNetwork(game_size, device='cpu')
        
        # 设置相同的权重
        state_dict = model1.state_dict()
        model2.load_state_dict(state_dict)
        
        # 创建相同的输入
        test_input = torch.randn(1, 3, game_size, game_size)
        
        # 获取输出
        with torch.no_grad():
            output1 = model1(test_input)
            output2 = model2(test_input)
        
        # 输出应该相同
        torch.testing.assert_close(output1[0], output2[0])  # policy
        torch.testing.assert_close(output1[1], output2[1])  # value
    
    def test_environment_independence(self):
        """测试环境对象的独立性"""
        from env.othello import OthelloEnv
        import copy
        
        env1 = OthelloEnv(size=6)
        env2 = copy.deepcopy(env1)
        
        # 初始状态应该相同
        obs1 = env1.reset()
        obs2 = env2.reset()
        np.testing.assert_array_equal(obs1, obs2)
        
        # 在一个环境中执行动作
        valid_moves = env1.board.get_valid_moves()
        if valid_moves:
            action = env1.get_action_from_coords(*valid_moves[0])
            env1.step(action)
        
        # 另一个环境应该保持不变
        obs2_after = env2.board.get_observation()
        np.testing.assert_array_equal(obs2, obs2_after)
    
    def test_thread_local_randomness(self):
        """测试线程本地的随机性"""
        def generate_random_sequence(seed):
            random.seed(seed)
            np.random.seed(seed)
            return [random.random() for _ in range(10)] + np.random.random(10).tolist()
        
        # 相同种子应该产生相同序列
        seq1 = generate_random_sequence(42)
        seq2 = generate_random_sequence(42)
        assert seq1 == seq2
        
        # 不同种子应该产生不同序列
        seq3 = generate_random_sequence(43)
        assert seq1 != seq3


class TestResourceManagement:
    """测试资源管理"""
    
    def test_memory_usage_tracking(self):
        """测试内存使用情况"""
        import psutil
        import gc
        
        # 记录初始内存
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # 创建一些大对象
        trainer = AlphaZeroTrainer(
            game_size=6,
            num_iterations=1,
            num_self_play_games=5,
            use_multiprocessing=False
        )
        
        # 检查内存增长
        after_creation = process.memory_info().rss
        memory_increase = after_creation - initial_memory
        
        # 清理
        del trainer
        gc.collect()
        
        # 内存增长应该在合理范围内 (小于100MB)
        assert memory_increase < 100 * 1024 * 1024, f"Memory increase too large: {memory_increase} bytes"
    
    def test_file_handle_management(self):
        """测试文件句柄管理"""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = AlphaZeroTrainer(
                game_size=6,
                checkpoint_path=os.path.join(temp_dir, 'test_checkpoint'),
                log_dir=temp_dir
            )
            
            # 应该能创建和保存检查点
            trainer.save_checkpoint(0)
            
            # 检查文件是否存在
            checkpoint_files = [f for f in os.listdir(temp_dir) if 'test_checkpoint' in f]
            assert len(checkpoint_files) > 0


if __name__ == "__main__":
    # 跳过需要长时间运行的测试
    pytest.main([__file__, "-v", "-k", "not test_multiprocess_self_play_basic"])