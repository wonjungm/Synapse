#!/bin/bash
# Mathematical Model Quick Test Script

echo "🧪 Testing Mathematical Failover Model..."

cd /acpl-ssd10/Synapse-private/benchmarks/soft_target/planner

echo "1️⃣ Running restart cost benchmark..."
python -c "
import sys
sys.path.append('/acpl-ssd10/Synapse-private')
from restart_cost_benchmark import RestartCostBenchmark
benchmark = RestartCostBenchmark()
costs = benchmark.run_full_benchmark()
print(f'✅ Measured costs: Save={costs.C_load:.2f}s, Load={costs.C_load:.2f}s')
"

echo "2️⃣ Testing mathematical optimizer..."
python -c "
import sys
sys.path.append('/acpl-ssd10/Synapse-private')
from mathematical_optimizer import monitor_and_replan_with_mathematical_model
print('🧠 Running mathematical model simulation...')
monitor_and_replan_with_mathematical_model(total_epochs=1, steps_per_epoch=100)
"

echo "✅ Mathematical model test completed!"