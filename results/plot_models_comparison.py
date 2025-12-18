import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import numpy as np

# 1. 准备数据 (Data Setup)
models = ["AutoDock-Vina", "RLaffinity", "RSAPred", "GatorAffinity-RNA\nw/o Pre-training", "GatorAffinity-RNA"]

data = {
    "Model": models,
    "RMSE": [3.454, 1.704, 1.428, 1.617, 1.290],
    "MAE": [2.608, 1.321, 1.178, 1.283, 1.060],
    "Pearson": [0.279, 0.048, 0.471, 0.459, 0.611],
    "Spearman": [0.356, 0.047, 0.377, 0.367, 0.500]
}

df = pd.DataFrame(data)

# 2. 全局样式设置 (Global Style Settings)
plt.rcParams['font.size'] = 28
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']  # 添加中文字体支持
plt.rcParams['axes.unicode_minus'] = False

# 3. 创建画布 (Create Figure)
fig, axes = plt.subplots(1, 4, figsize=(36, 8))

# 子图配置
metrics_config = [
    ("RMSE", "RMSE (Lower is Better)", False),  # 降序 (最右侧为最小值/最优)
    ("MAE", "MAE (Lower is Better)", False),    # 降序 (最右侧为最小值/最优)
    ("Pearson", "Pearson's R (Higher is Better)", True),  # 升序 (最右侧为最大值/最优)
    ("Spearman", "Spearman's Rho (Higher is Better)", True)
]

# 辅助函数：调整Y轴范围，避免柱子顶天立地
def adjust_yaxis(ax, data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    margin = (max_val - min_val) * 0.15
    lower = max(0, min_val - margin) if min_val > 0 else min_val - margin
    ax.set_ylim(lower, max_val + margin)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# 辅助函数：添加数值标签
def add_labels(ax):
    for p in ax.patches:
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ymin, ymax = ax.get_ylim()
        offset = (ymax - ymin) * 0.01
        ax.text(x, y + offset, f'{y:.3f}', ha='center', va='bottom', fontsize=22, fontweight='bold')

# 4. 绘图循环 (Plotting Loop)
for i, (metric, title, ascending) in enumerate(metrics_config):
    ax = axes[i]

    # 不排序，保持原始顺序
    df_sorted = df.copy()

    # 生成渐变色板 (Gradient Palette)
    palette = list(sns.color_palette("Blues_d", len(df)))

    # 将GatorAffinity-RNA对应的颜色修改为红色高亮
    for idx, model in enumerate(df_sorted['Model']):
        if model == 'GatorAffinity-RNA':
            palette[idx] = '#E35B54'

    # 绘图核心代码
    # edgecolor='black': 设置边框颜色为黑色
    # linewidth=2: 设置边框粗细
    sns.barplot(x='Model', y=metric, data=df_sorted, ax=ax, palette=palette,
                edgecolor='black', linewidth=2)

    # 样式美化
    ax.set_title(title, fontsize=32, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # 设置X轴标签，GatorAffinity-RNA加粗
    labels = df_sorted['Model'].tolist()
    for idx, label in enumerate(ax.get_xticklabels()):
        if labels[idx] == 'GatorAffinity-RNA':
            label.set_fontweight('bold')
            label.set_fontsize(26)
        else:
            label.set_fontsize(26)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    adjust_yaxis(ax, df_sorted, metric)
    add_labels(ax)

    # 去除上方和右侧的边框 (Spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)  # 添加虚线网格
    ax.set_axisbelow(True)  # 网格置于图层下方

plt.tight_layout()
plt.savefig('models_comparison.png', dpi=300, bbox_inches='tight')
print("图表已保存为 models_comparison.png")
plt.show()
