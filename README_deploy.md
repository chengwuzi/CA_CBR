# MultiCBR + CAGCN* 实验部署指南

## 1. 环境准备 (Conda 推荐)

强烈建议使用 Conda 创建干净的隔离环境，避免依赖冲突。

### 推荐版本
- **Python**: 3.10 (稳定且兼容性最好)
- **PyTorch**: 2.0.1 + CUDA 11.8 (或根据服务器显卡驱动选择)

### 极速安装步骤

1. **创建环境**
```bash
conda create -n MultiCBR python=3.10 -y
conda activate MultiCBR
```

2. **安装 PyTorch (带 CUDA 支持)**
   *请根据服务器实际情况选择命令，以下是通用推荐 (CUDA 11.8)*：
```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

3. **安装项目依赖**
```bash
pip install -r requirements.txt
```

4. **(可选) 安装 torch-scatter 加速**
   *只有在安装完 PyTorch 后执行此步。如果服务器网不好，可以跳过，代码会自动兼容。*
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## 2. 运行实验
本项目已配置自动化实验脚本，会自动遍历以下组合：
- **数据集**: NetEase, iFashion, Youshu
- **CAGCN 指标**: jc (Jaccard), sc (Cosine), lhn, co (Common Neighbors)

### 启动命令
```bash
# 1. 确保代码是最新的（如果是在本地修改后上传）
# 请确保 utility.py, models/MultiCBR.py, train.py, run_experiments.py 都是最新版

# 2. 运行诊断（可选）
python diagnose.py

# 3. 启动实验
python run_experiments.py
```

### 运行说明
- **首次运行**：会自动计算并缓存 CIR 权重（.pt 文件），这可能需要几分钟。
- **显存占用**：Utility 模块默认 batch_size=2000。如果遇到 OOM，请修改 `utility.py` 中的 `batch_size`。
- **日志监控**：
  - 屏幕会实时显示进度条。
  - 实验结果保存在 `log/<Dataset>/MultiCBR/` 目录下。

## 附录：Conda 环境打包与迁移

如果您想把配置好的环境保存下来（防止下次租服务器要重装），或者在多台服务器间同步，推荐使用 **Conda Pack**。

### 1. 安装打包工具
```bash
# 在当前环境中安装
conda install -c conda-forge conda-pack -y
```

### 2. 打包环境 (生成 .tar.gz)
```bash
# 将当前激活的 MultiCBR 环境打包名为 my_env_packed.tar.gz
conda pack -n MultiCBR -o my_env_packed.tar.gz
```
*提示：这个压缩包通常有几百MB到1GB，包含了 Python 解析器和所有依赖包。*

### 3. 恢复环境 (在另一台机器上)
**无需安装 Conda**，直接解压即可使用：
```bash
# 1. 创建目录并解压
mkdir -p my_env
tar -xzf my_env_packed.tar.gz -C my_env

# 2. 激活环境 (使用 source)
source my_env/bin/activate

# 3. 验证
python --version
```

### ⚠️ 关键注意事项
- **系统必须一致**：**Linux 打包的只能在 Linux 用**，Windows 打包的只能在 Windows 用。
- **不能跨平台**：您**不能**在本地 Windows 电脑上打包好，然后上传到 Linux 服务器上用（二进制文件不兼容）。
- **用途**：最适合在租用的服务器上配好一次环境后打包下载保存，下次租新机器时上传解压即用，省去 `conda install` 的漫长等待。
