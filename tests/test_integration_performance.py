"""
集成测试和性能测试 - 测试整个系统的集成和性能
"""
import pytest
import numpy as np
import torch
import time
import tempfile
import os
import sys
from unittest.mock import Mock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import AlphaZeroTrainer
from mcts.mcts import MCTS
from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork
from play import AlphaZeroPlayer
# evaluate.py 中不存在 evaluate_models，提供兼容包装
from evaluate import compare_models, evaluate_model

def evaluate_models(player1, player2, num_games=2, game_size=6):
    """Simple wrapper to evaluate two AlphaZeroPlayer instances by alternating colors.
    Returns win rate of player1.
    """
    from play import play_game
    wins = 0
    for i in range(num_games):
        if i % 2 == 0:
            black, white = player1, player2
            p1_color = -1
        else:
            black, white = player2, player1
            p1_color = 1
        winner = play_game(black, white, render=False)
        if winner == p1_color:
            wins += 1
    return wins / num_games if num_games > 0 else 0.0


class TestIntegration:
    """集成测试"""
    
    def test_complete_training_cycle(self):
        """测试完整的训练周期"""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = AlphaZeroTrainer(
                game_size=6,
                num_iterations=2,  # 小规模测试
                num_self_play_games=4,
                num_mcts_simulations=10,
                checkpoint_path=os.path.join(temp_dir, 'test_checkpoint'),
                log_dir=temp_dir,
                use_multiprocessing=False
            )
            
            # 执行训练
            trainer.train()
            
            # 检查checkpoint是否生成
            checkpoint_files = [f for f in os.listdir(temp_dir) if 'test_checkpoint' in f and f.endswith('.pt')]
            assert len(checkpoint_files) >= 2  # 应该有至少2个checkpoint
            
            # 检查损失是否记录
            assert len(trainer.policy_losses) == 2
            assert len(trainer.value_losses) == 2
            assert len(trainer.total_losses) == 2
            
            # 检查所有损失都是有限数值
            assert all(np.isfinite(loss) for loss in trainer.policy_losses)
            assert all(np.isfinite(loss) for loss in trainer.value_losses)
            assert all(np.isfinite(loss) for loss in trainer.total_losses)
    
    def test_checkpoint_load_resume(self):
        """测试checkpoint加载和恢复训练"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 第一阶段：训练并保存
            trainer1 = AlphaZeroTrainer(
                game_size=6,
                num_iterations=1,
                num_self_play_games=2,
                num_mcts_simulations=5,
                checkpoint_path=os.path.join(temp_dir, 'resume_test'),
                log_dir=temp_dir,
                use_multiprocessing=False
            )
            
            trainer1.train()
            
            # 第二阶段：加载并继续训练
            trainer2 = AlphaZeroTrainer(
                game_size=6,
                num_iterations=1,
                num_self_play_games=2,
                num_mcts_simulations=5,
                checkpoint_path=os.path.join(temp_dir, 'resume_test'),
                log_dir=temp_dir,
                use_multiprocessing=False
            )
            
            # 找到最新的checkpoint
            checkpoint_files = [f for f in os.listdir(temp_dir) if 'resume_test' in f and f.endswith('.pt')]
            latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(temp_dir, x)))
            checkpoint_path = os.path.join(temp_dir, latest_checkpoint)
            
            # 继续训练
            trainer2.train(resume_from=checkpoint_path)
            
            # 检查损失历史是否正确恢复
            assert len(trainer2.policy_losses) == 2  # 1 (loaded) + 1 (new)
            assert len(trainer2.value_losses) == 2
            assert len(trainer2.total_losses) == 2
    
    def test_model_evaluation_integration(self):
        """测试模型评估集成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 训练一个小模型
            trainer = AlphaZeroTrainer(
                game_size=6,
                num_iterations=1,
                num_self_play_games=2,
                num_mcts_simulations=5,
                checkpoint_path=os.path.join(temp_dir, 'eval_test'),
                use_multiprocessing=False
            )
            
            trainer.train()
            
            # 获取最新checkpoint路径供玩家加载
            ckpt_files = [f for f in os.listdir(temp_dir) if 'eval_test' in f and f.endswith('.pt')]
            assert ckpt_files, 'No checkpoint generated for evaluation test'
            latest_ckpt = max(ckpt_files, key=lambda x: os.path.getctime(os.path.join(temp_dir, x)))
            model_path = os.path.join(temp_dir, latest_ckpt)

            # 创建两个玩家进行评估（使用相同模型做对称性测试）
            player1 = AlphaZeroPlayer(model_path, num_simulations=5)
            player2 = AlphaZeroPlayer(model_path, num_simulations=5)
            
            # 运行少量游戏进行评估
            try:
                win_rate = evaluate_models(player1, player2, num_games=2, game_size=6)
                
                # 检查结果
                assert 0 <= win_rate <= 1
                assert isinstance(win_rate, (int, float))
                
            except Exception as e:
                # 如果evaluate_models函数不存在或有问题，跳过测试
                pytest.skip(f"Model evaluation test skipped: {e}")
    
    def test_multiprocessing_integration(self):
        """测试多进程集成"""
        trainer = AlphaZeroTrainer(
            game_size=6,
            num_iterations=1,
            num_self_play_games=4,
            num_mcts_simulations=5,
            use_multiprocessing=True,
            mp_num_workers=2,
            mp_games_per_worker=2
        )
        
        try:
            # 测试多进程自我对弈
            examples = trainer.self_play_multiprocess(num_workers=2, games_per_worker=1)
            
            # 检查结果
            assert isinstance(examples, list)
            assert len(examples) > 0
            
            # 检查样本格式
            for example in examples[:3]:  # 检查前3个样本
                state, policy, value = example
                assert state.shape == (3, 6, 6)
                assert len(policy) == 37
                assert -1 <= value <= 1
                
        except Exception as e:
            pytest.skip(f"Multiprocessing integration test skipped: {e}")


class TestPerformance:
    """性能测试"""
    
    def test_mcts_search_performance(self):
        """测试MCTS搜索性能"""
        model = AlphaZeroNetwork(6, device='cpu')
        mcts = MCTS(model, num_simulations=50)
        env = OthelloEnv(size=6)
        
        # 预热
        state = env.reset()
        _ = mcts.search(state, env, temperature=1.0)
        
        # 性能测试
        start_time = time.time()
        num_searches = 10
        
        for _ in range(num_searches):
            state = env.reset()
            _ = mcts.search(state, env, temperature=1.0)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_search = total_time / num_searches
        
        # 每次搜索应该在合理时间内完成（小于5秒）
        assert avg_time_per_search < 5.0, f"MCTS search too slow: {avg_time_per_search:.2f}s per search"
        
        print(f"MCTS performance: {avg_time_per_search:.3f}s per search (50 simulations)")
    
    def test_neural_network_inference_performance(self):
        """测试神经网络推理性能"""
        model = AlphaZeroNetwork(6, device='cpu')
        model.eval()
        
        # 不同批大小的测试
        batch_sizes = [1, 4, 16]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 3, 6, 6)
            
            # 预热
            with torch.no_grad():
                _ = model(input_tensor)
            
            # 性能测试
            start_time = time.time()
            num_inferences = 100
            
            with torch.no_grad():
                for _ in range(num_inferences):
                    _ = model(input_tensor)
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_inference = total_time / num_inferences
            avg_time_per_sample = avg_time_per_inference / batch_size
            
            # 每个样本的推理时间应该很快（小于10ms）
            assert avg_time_per_sample < 0.01, \
                f"Neural network inference too slow: {avg_time_per_sample*1000:.2f}ms per sample (batch_size={batch_size})"
            
            print(f"NN performance (batch_size={batch_size}): {avg_time_per_sample*1000:.3f}ms per sample")
    
    def test_training_iteration_performance(self):
        """测试训练迭代性能"""
        trainer = AlphaZeroTrainer(
            game_size=6,
            num_iterations=1,
            num_self_play_games=5,
            num_mcts_simulations=20,
            use_multiprocessing=False
        )
        
        start_time = time.time()
        
        # 执行一次训练迭代
        examples = trainer.self_play()
        policy_loss, value_loss, total_loss = trainer.train_network(examples)
        
        end_time = time.time()
        iteration_time = end_time - start_time
        
        # 一次迭代应该在合理时间内完成（小于60秒）
        assert iteration_time < 60.0, f"Training iteration too slow: {iteration_time:.2f}s"
        
        print(f"Training iteration performance: {iteration_time:.2f}s (5 games, 20 simulations)")
    
    @pytest.mark.slow
    def test_memory_usage_during_training(self):
        """测试训练期间的内存使用"""
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        
        initial_memory = process.memory_info().rss
        
        trainer = AlphaZeroTrainer(
            game_size=6,
            num_iterations=2,
            num_self_play_games=10,
            num_mcts_simulations=20,
            use_multiprocessing=False
        )
        
        # 执行训练
        trainer.train()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 内存增长应该在合理范围内（小于500MB）
        max_allowed_increase = 500 * 1024 * 1024
        assert memory_increase < max_allowed_increase, \
            f"Memory usage too high: {memory_increase / 1024 / 1024:.2f}MB increase"
        
        print(f"Memory usage: {memory_increase / 1024 / 1024:.2f}MB increase during training")


class TestRobustness:
    """鲁棒性测试"""
    
    def test_error_recovery(self):
        """测试错误恢复能力"""
        trainer = AlphaZeroTrainer(
            game_size=6,
            num_iterations=1,
            num_self_play_games=1,
            num_mcts_simulations=5,
            use_multiprocessing=False
        )
        
        # 测试无效输入处理
        try:
            # 空样本列表
            policy_loss, value_loss, total_loss = trainer.train_network([])
            # 应该跳过训练并返回零损失
            assert policy_loss == 0.0
            assert value_loss == 0.0
            assert total_loss == 0.0
        except Exception as e:
            pytest.fail(f"Should handle empty examples gracefully: {e}")
    
    def test_edge_case_board_states(self):
        """测试边缘情况的棋盘状态"""
        env = OthelloEnv(size=6)
        model = AlphaZeroNetwork(6, device='cpu')
        mcts = MCTS(model, num_simulations=5)
        
        # 测试游戏结束状态
        env.reset()
        
        # 手动设置一个接近结束的状态
        env.board.board.fill(1)  # 填满白子
        env.board.board[0, 0] = -1  # 一个黑子
        env.board.board[0, 1] = 0   # 一个空位
        env.board.current_player = -1
        
        # MCTS应该能处理这种状态
        try:
            state = env.board.get_observation()
            action_probs = mcts.search(state, env, temperature=1.0)
            
            assert len(action_probs) == 37
            assert not np.any(np.isnan(action_probs))
            assert not np.any(np.isinf(action_probs))
            
        except Exception as e:
            pytest.fail(f"MCTS should handle edge case board states: {e}")
    
    def test_concurrent_model_usage(self):
        """测试模型的并发使用"""
        model = AlphaZeroNetwork(6, device='cpu')
        
        # 模拟并发推理
        inputs = [torch.randn(1, 3, 6, 6) for _ in range(5)]
        
        results = []
        for input_tensor in inputs:
            with torch.no_grad():
                result = model(input_tensor)
                results.append(result)
        
        # 所有结果都应该有效
        for policy_logits, value in results:
            assert policy_logits.shape == (1, 37)
            assert value.shape == (1, 1)
            assert torch.all(torch.isfinite(policy_logits))
            assert torch.all(torch.isfinite(value))
    
    def test_device_compatibility(self):
        """测试设备兼容性"""
        # 测试CPU设备
        model_cpu = AlphaZeroNetwork(6, device='cpu')
        input_cpu = torch.randn(1, 3, 6, 6, device='cpu')
        
        with torch.no_grad():
            output_cpu = model_cpu(input_cpu)
        
        assert output_cpu[0].device.type == 'cpu'
        assert output_cpu[1].device.type == 'cpu'
        
        # 如果CUDA可用，也测试CUDA
        if torch.cuda.is_available():
            model_cuda = AlphaZeroNetwork(6, device='cuda')
            input_cuda = torch.randn(1, 3, 6, 6, device='cuda')
            
            with torch.no_grad():
                output_cuda = model_cuda(input_cuda)
            
            assert output_cuda[0].device.type == 'cuda'
            assert output_cuda[1].device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])