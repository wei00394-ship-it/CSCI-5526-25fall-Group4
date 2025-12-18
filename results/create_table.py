import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 准备数据（均值 ± 标准差）
data = {
    "Model": [
        "AutoDock-Vina",
        "RLaffinity",
        "RSAPred",
        "GatorAffinity-RNA\nw/o Pre-training",
        "GatorAffinity-RNA"
    ],
    "RMSE": ["3.454±0.814", "1.704±0.246", "1.428±0.144", "1.617±0.430", "1.290±0.175"],
    "MAE": ["2.608±0.393", "1.321±0.213", "1.178±0.128", "1.283±0.332", "1.060±0.167"],
    "Pearson": ["0.279±0.203", "0.048±0.234", "0.471±0.081", "0.459±0.150", "0.611±0.046"],
    "Spearman": ["0.356±0.166", "0.047±0.249", "0.377±0.122", "0.367±0.136", "0.500±0.047"]
}

df = pd.DataFrame(data)

# 创建图形
fig, ax = plt.subplots(figsize=(18, 8))
ax.axis('off')

# 创建表格，留出上下边距
table = ax.table(cellText=df.values,
                colLabels=df.columns,
                cellLoc='center',
                loc='center',
                bbox=[0, 0.08, 1, 0.84])

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(15)
table.scale(1, 3)

# 设置所有单元格无边框
for key, cell in table.get_celld().items():
    cell.set_edgecolor('white')
    cell.set_linewidth(0)

# 设置表头样式
for i in range(len(df.columns)):
    cell = table[(0, i)]
    cell.set_facecolor('white')
    cell.set_text_props(weight='bold', size=18)

# 设置第一列（模型名称）左对齐
for i in range(1, len(df) + 1):
    cell = table[(i, 0)]
    cell.set_text_props(ha='left', weight='bold', size=16)

    # GatorAffinity-RNA 行加粗
    if i == len(df):
        for j in range(len(df.columns)):
            table[(i, j)].set_text_props(weight='bold', size=16)

# 设置其他单元格样式
for i in range(1, len(df) + 1):
    for j in range(1, len(df.columns)):
        cell = table[(i, j)]
        cell.set_text_props(size=16)

# 手动绘制三条横线
# 顶线（表格顶部）
ax.plot([0.0, 1.0], [0.93, 0.93], 'k-', linewidth=2, transform=ax.transAxes)

# 中线（表头下方）
header_y = 0.93 - (0.84 / (len(df) + 1))
ax.plot([0.0, 1.0], [header_y, header_y], 'k-', linewidth=1.5, transform=ax.transAxes)

# 底线（表格底部）
ax.plot([0.0, 1.0], [0.08, 0.08], 'k-', linewidth=2, transform=ax.transAxes)

plt.savefig('models_table.png', dpi=300, bbox_inches='tight', facecolor='white')
print("三线表已保存为 models_table.png")
plt.show()
