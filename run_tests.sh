#!/bin/bash

# 测试运行脚本
# 使用方法: ./run_tests.sh [test_type]

set -e

echo "🧪 AlphaZero测试套件"
echo "===================="

# 检查是否安装了pytest
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest未安装，请运行: pip install pytest pytest-timeout"
    exit 1
fi

# 获取测试类型参数
TEST_TYPE=${1:-"all"}

case $TEST_TYPE in
    "unit")
        echo "🔍 运行单元测试..."
        pytest tests/test_boundary_conditions.py tests/test_numerical_stability.py tests/test_state_consistency.py -v
        ;;
    "integration")
        echo "🔗 运行集成测试..."
        pytest tests/test_integration_performance.py::TestIntegration -v
        ;;
    "performance")
        echo "⚡ 运行性能测试..."
        pytest tests/test_integration_performance.py::TestPerformance -v
        ;;
    "memory")
        echo "💾 运行内存测试..."
        pytest tests/test_memory_management.py -v
        ;;
    "multiprocessing")
        echo "🔄 运行多进程测试..."
        pytest tests/test_multiprocessing_safety.py -v
        ;;
    "boundary")
        echo "🚧 运行边界条件测试..."
        pytest tests/test_boundary_conditions.py -v
        ;;
    "stability")
        echo "🎯 运行数值稳定性测试..."
        pytest tests/test_numerical_stability.py -v
        ;;
    "consistency")
        echo "📏 运行状态一致性测试..."
        pytest tests/test_state_consistency.py -v
        ;;
    "fast")
        echo "🚀 运行快速测试（跳过慢速测试）..."
        pytest tests/ -v -m "not slow"
        ;;
    "slow")
        echo "🐌 运行完整测试套件（包括慢速测试）..."
        pytest tests/ -v
        ;;
    "critical")
        echo "🚨 运行关键bug测试..."
        pytest tests/test_boundary_conditions.py::TestMCTSBoundaryConditions::test_action_probability_array_bounds -v
        pytest tests/test_boundary_conditions.py::TestOthelloBoundaryConditions::test_board_boundary_access -v
        pytest tests/test_multiprocessing_safety.py::TestMultiprocessingSafety::test_worker_initialization -v
        pytest tests/test_numerical_stability.py::TestNumericalStability::test_policy_normalization_stability -v
        pytest tests/test_state_consistency.py::TestStateConsistency::test_state_representation_consistency -v
        ;;
    "coverage")
        echo "📊 运行测试覆盖率分析..."
        if command -v pytest-cov &> /dev/null; then
            pytest tests/ --cov=. --cov-report=html --cov-report=term-missing -v
            echo "📈 覆盖率报告生成在 htmlcov/index.html"
        else
            echo "❌ pytest-cov未安装，请运行: pip install pytest-cov"
            exit 1
        fi
        ;;
    "all"|*)
        echo "🎯 运行所有测试..."
        echo ""
        echo "1️⃣ 边界条件测试..."
        pytest tests/test_boundary_conditions.py -v
        echo ""
        echo "2️⃣ 多进程安全测试..."
        pytest tests/test_multiprocessing_safety.py -v -k "not test_multiprocess_self_play_basic"
        echo ""
        echo "3️⃣ 内存管理测试..."
        pytest tests/test_memory_management.py -v -m "not slow"
        echo ""
        echo "4️⃣ 数值稳定性测试..."
        pytest tests/test_numerical_stability.py -v
        echo ""
        echo "5️⃣ 状态一致性测试..."
        pytest tests/test_state_consistency.py -v
        echo ""
        echo "6️⃣ 集成测试..."
        pytest tests/test_integration_performance.py::TestIntegration -v
        echo ""
        echo "7️⃣ 性能测试..."
        pytest tests/test_integration_performance.py::TestPerformance -v
        ;;
esac

echo ""
echo "✅ 测试完成！"
echo ""
echo "📋 可用的测试类型:"
echo "   unit          - 单元测试"
echo "   integration   - 集成测试"
echo "   performance   - 性能测试"
echo "   memory        - 内存测试"
echo "   multiprocessing - 多进程测试"
echo "   boundary      - 边界条件测试"
echo "   stability     - 数值稳定性测试"
echo "   consistency   - 状态一致性测试"
echo "   fast          - 快速测试（跳过慢速）"
echo "   slow          - 完整测试套件"
echo "   critical      - 关键bug测试"
echo "   coverage      - 测试覆盖率"
echo "   all           - 所有测试（默认）"