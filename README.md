# RG-SCS

**Relation-Guided Sea Clutter Suppression with Joint Target and Background Constraints**

RG-SCS 从复数雷达回波中学习目标相位变化，用两侧参考距离单元估计海杂波，再通过预条件共轭梯度法求解目标回波。

| 实测输入 | RG-SCS 输出 |
| :---: | :---: |
| ![处理前的距离–多普勒图](assets/input.png) | ![处理后的距离–多普勒图](assets/output.png) |

两图使用相同的幅度色标。处理后，近零多普勒处的目标回波仍清晰可见，周围杂波减弱。

### 快速运行

在仓库根目录执行。最后一条命令使用随机权重和模拟输入，在 CPU 上验证模型前向计算。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python method/check_forward.py
```

### 训练与推理

完整实验需要 CUDA 和原始 NPZ 数据。按 [样本清单](reports/manifest.csv) 中的相对路径放置输入和标签。仓库提供源码与实测效果图，数据和预训练权重需另行准备。

```bash
python method/train.py              # 训练并评估 best 和 last
python method/train.py --resume     # 中断后续训
python method/infer.py --tag best   # 用已训练的最佳权重单独推理
```

[模型](method/model.py) · [求解器](method/base_model.py) · [损失函数](method/losses.py) · [训练配置](method/config.json)

权重保存在 `runs/rg-scs/`，推理图与指标保存在 `inference/best/` 或 `inference/last/`。默认从头训练使用 250 轮余弦学习率调度；图示实验在第 85 轮后以固定学习率续训。
