# MultiCBR + CAGCN* 实验部署指南

## 1. 环境准备
在租用的 GPU 服务器（Linux/Windows）上，请确保已安装 CUDA 环境。

### 推荐环境
- Python 3.8+
- PyTorch >= 1.10 (推荐 2.0+)
- CUDA 11.x / 12.x

### 安装依赖
```bash
# 1. 安装基础依赖
pip install -r requirements.txt

# 2. (可选但推荐) 安装 torch-scatter 以加速图聚合
# 请根据你的 PyTorch 和 CUDA 版本选择对应的 whl 文件
# 参考：https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
# 例如 (Torch 2.0.0 + CUDA 11.8):
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## 2. 运行实验
本项目已配置自动化实验脚本，会自动遍历以下组合：
- **数据集**: NetEase, iFashion, Youshu
- **CAGCN 指标**: jc (Jaccard), sc (Cosine), lhn, co (Common Neighbors)

### 启动命令
```bash
# 直接运行脚本
python run_experiments.py
```

### 运行说明
- **首次运行**：会自动计算并缓存 CIR 权重（.pt 文件），这可能需要几分钟（视数据集大小而定）。
- **显存占用**：Utility 模块已针对 24G 显存（RTX 3090/4090）优化，默认 batch_size=2000。如果遇到 OOM，请修改 `utility.py` 中的 `batch_size`。
- **日志监控**：
  - 调度日志（看进度）：`experiment_logs/` 目录下的 `.log` 文件。
  - 实验结果（看指标）：`log/<Dataset>/MultiCBR/` 目录下对应文件夹。
