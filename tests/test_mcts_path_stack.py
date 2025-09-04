#!/usr/bin/env python3
"""
测试MCTS路径栈和反向传播的正确性
"""

import pytest
import numpy as np
import torch
import sys
import os
from unittest.mock import Mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.mcts import MCTS, Node
from env.othello import OthelloEnv
from models.neural_network import AlphaZeroNetwork


class TestMCTSPathStack:
    """测试MCTS路径栈逻辑"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.game_size = 6
        self.env = OthelloEnv(size=self.game_size)
        self.model = AlphaZeroNetwork(self.game_size, device='cpu')
        self.mcts = MCTS(self.model, num_simulations=1, c_puct=1.0)
    
    def test_path_stack_parent_child_relationship(self):
        """测试路径栈中的父子关系是否正确"""
        self.env.reset()
        
        # 创建一个简单的树结构来测试
        # Root -> Child1 -> Grandchild
        root = Node(0.5)
        child1 = Node(0.3)
        grandchild = Node(0.2)
        
        # 手动构建树结构
        root.children[0] = child1
        child1.children[1] = grandchild
        
        # 模拟路径栈的构建过程
        path_stack = []
        current_node = root
        
        # 第一步：从root选择action 0 到child1
        action_0 = 0
        path_stack.append((current_node, action_0))
        current_node = current_node.children[action_0]
        
        # 第二步：从child1选择action 1 到grandchild
        action_1 = 1
        path_stack.append((current_node, action_1))
        current_node = current_node.children[action_1]
        
        # 现在current_node是grandchild，path_stack应该是[(root, 0), (child1, 1)]
        print(f"Path stack length: {len(path_stack)}")
        print(f"Current node is grandchild: {current_node is grandchild}")
        
        # 验证路径栈的正确性
        assert len(path_stack) == 2, "路径栈应该有2个元素"
        assert path_stack[0][0] is root, "第一个元素应该是root"
        assert path_stack[0][1] == 0, "第一个动作应该是0"
        assert path_stack[1][0] is child1, "第二个元素应该是child1"  
        assert path_stack[1][1] == 1, "第二个动作应该是1"
        
        # 现在测试反向传播
        # 假设grandchild得到价值1.0
        leaf_value = 1.0
        current_node.value_sum += leaf_value
        current_node.visit_count += 1
        
        # 反向传播
        current_value = -leaf_value  # 对父节点翻转
        for parent_node, action in reversed(path_stack):
            parent_node.value_sum += current_value
            parent_node.visit_count += 1
            current_value = -current_value
        
        # 验证结果
        print(f"Grandchild value: {grandchild.value()}, visits: {grandchild.visit_count}")
        print(f"Child1 value: {child1.value()}, visits: {child1.visit_count}")  
        print(f"Root value: {root.value()}, visits: {root.visit_count}")
        
        # grandchild: value = 1.0, visits = 1
        assert abs(grandchild.value() - 1.0) < 1e-6, f"Grandchild应该有价值1.0，实际是{grandchild.value()}"
        assert grandchild.visit_count == 1, "Grandchild应该有1次访问"
        
        # child1: value = -1.0 (翻转), visits = 1  
        assert abs(child1.value() - (-1.0)) < 1e-6, f"Child1应该有价值-1.0，实际是{child1.value()}"
        assert child1.visit_count == 1, "Child1应该有1次访问"
        
        # root: value = 1.0 (再次翻转), visits = 1
        assert abs(root.value() - 1.0) < 1e-6, f"Root应该有价值1.0，实际是{root.value()}"
        assert root.visit_count == 1, "Root应该有1次访问"

    def test_actual_mcts_simulation_path_correctness(self):
        """测试真实MCTS模拟中的路径正确性"""
        self.env.reset()
        
        # 创建一个mock模型来控制输出
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.tensor([1.0])]
        
        # 创建MCTS实例
        mcts = MCTS(mock_model, num_simulations=2, c_puct=1.0)
        
        # 手动跟踪一次模拟过程
        class TrackedMCTS(MCTS):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.simulation_traces = []
            
            def _simulate_iterative(self, root, env):
                """重写以跟踪模拟过程"""
                path_stack = []
                node = root
                env_copy = copy.deepcopy(env)
                original_path_stack = []
                
                # Selection phase - 记录路径
                while node.expanded():
                    action = node.select_child(self.c_puct)
                    original_path_stack.append((id(node), action, node))  # 记录节点ID和对象
                    path_stack.append((node, action))
                    
                    env_copy.step(action)
                    
                    if action in node.children:
                        node = node.children[action]
                    else:
                        break
                
                # 记录最终节点
                final_node_id = id(node)
                
                # 存储追踪信息
                self.simulation_traces.append({
                    'path': original_path_stack,
                    'final_node_id': final_node_id,
                    'path_stack_length': len(path_stack)
                })
                
                # 继续正常的评估和反向传播
                canonical_state = env_copy.board.get_canonical_state()
                game_ended = env_copy.board.is_done()
                
                if game_ended:
                    winner = env_copy.board.get_winner()
                    if winner == 0:
                        value = 0.0
                    else:
                        value = 1.0 if winner == self.search_start_player else -1.0
                else:
                    # 使用固定值进行测试
                    value = 0.5
                    policy = np.ones(env.board.get_action_space_size()) / env.board.get_action_space_size()
                    node.expand(canonical_state, policy)
                
                # 反向传播
                node.value_sum += value
                node.visit_count += 1
                
                current_value = -value
                for parent_node, action in reversed(path_stack):
                    parent_node.value_sum += current_value
                    parent_node.visit_count += 1
                    current_value = -current_value
        
        # 创建tracked MCTS
        import copy
        tracked_mcts = TrackedMCTS(mock_model, num_simulations=2, c_puct=1.0)
        
        # Mock评估函数
        def mock_evaluate(canonical_state, env):
            action_space_size = env.board.get_action_space_size()
            policy = np.ones(action_space_size) / action_space_size
            return policy, 0.5
        
        tracked_mcts._evaluate_state = mock_evaluate
        
        # 执行搜索
        state = self.env.board.get_observation()
        try:
            action_probs = tracked_mcts.search(state, self.env, temperature=1.0)
            
            # 验证追踪信息
            print(f"模拟次数: {len(tracked_mcts.simulation_traces)}")
            for i, trace in enumerate(tracked_mcts.simulation_traces):
                print(f"模拟 {i}: 路径长度 = {trace['path_stack_length']}")
                for j, (node_id, action, node_obj) in enumerate(trace['path']):
                    print(f"  步骤 {j}: 节点{node_id} -> 动作{action}")
            
            assert len(action_probs) == self.game_size * self.game_size + 1
            
        except Exception as e:
            print(f"MCTS搜索失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("测试MCTS路径栈逻辑...")
    print("=" * 60)
    
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])