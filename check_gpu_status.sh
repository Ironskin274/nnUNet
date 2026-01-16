#!/bin/bash
# GPU 状态检查脚本

echo "=========================================="
echo "GPU 状态诊断"
echo "=========================================="
echo "时间: $(date)"
echo ""

echo "=== 所有 GPU 状态 ==="
nvidia-smi

echo ""
echo "=== GPU 详细信息 ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv

echo ""
echo "=== GPU 1 上的进程 ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv --id=1 2>/dev/null || echo "GPU 1 上无进程或 GPU 1 不可用"

echo ""
echo "=== 检查 GPU 1 设备文件 ==="
if [ -e /dev/nvidia1 ]; then
    echo "✓ /dev/nvidia1 存在"
    echo "使用该设备的进程:"
    fuser -v /dev/nvidia1 2>/dev/null || echo "  无进程使用"
else
    echo "❌ /dev/nvidia1 不存在"
fi

echo ""
echo "=== CUDA 可用性测试 ==="
python3 << 'PYEOF'
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"\n当前 CUDA_VISIBLE_DEVICES: {torch.cuda.device_count()}")
else:
    print("❌ CUDA 不可用")
PYEOF

echo ""
echo "=== 环境变量 ==="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-未设置}"

echo ""
echo "=========================================="




