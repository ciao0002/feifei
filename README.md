# Enhanced CoLight: Dynamic Adjacency and 2-Hop Neighbor Extension

基于 [Advanced_XLight](https://github.com/LiangZhang1996/Advanced_XLight) (ICML 2022) 的增强版 CoLight 实现。

## 改进内容

本项目在原有 CoLight 模型基础上提出两项增强：

### 1. 动态邻接权重 (Dynamic Adjacency Weights)
- 将静态的 0/1 邻接矩阵替换为基于实时车流量的动态权重
- 权重计算：`weight[j] = 邻居j方向入流车辆数 / 总入流车辆数`

### 2. 二阶邻居扩展 (2-Hop Neighbor Extension)
- 将协调范围从一阶邻居扩展到二阶邻居（邻居的邻居）
- 使用衰减因子（默认 0.5）模拟距离对影响力的削弱

## 实验结果

在 Jinan 3×4 数据集上的消融实验结果：

| 方法 | 平均行程时间 | 改进率 |
|-----|------------:|------:|
| Baseline CoLight | 84.79s | - |
| +Dynamic Adj | 83.94s | +1.00% |
| +2-Hop | 83.98s | +0.96% |
| Full (Both) | 83.95s | +0.99% |

## 使用方法

```bash
# 安装依赖
pip install tensorflow==2.4 pandas numpy cityflow

# 运行增强版 CoLight
python run_colight_enhanced.py -jinan --dynamic_adj --two_hop --rounds 80

# 仅启用动态邻接权重
python run_colight_enhanced.py -jinan --dynamic_adj --rounds 80

# 仅启用二阶邻居
python run_colight_enhanced.py -jinan --two_hop --rounds 80
```

## 引用

本项目基于以下工作：

```bibtex
@inproceedings{advanced_xlight,
  title={Expression might be enough: representing pressure and demand for reinforcement learning based traffic signal control},
  author={Zhang, Liang and Wu, Qiang and Shen, Jun and L{\"u}, Linyuan and Du, Bo and Wu, Jianqing},
  booktitle={International Conference on Machine Learning},
  pages={26645--26654},
  year={2022},
  organization={PMLR}
}
```

## License

本项目遵循 GNU GPLv3 协议。
