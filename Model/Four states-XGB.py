import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import shap
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import plotly.graph_objects as go


# 读取训练集和测试集
train_df = pd.read_excel('训练集.xlsx')
test_df = pd.read_excel('测试集.xlsx')

# 特征和标签
X_train = train_df[['pH', 'K+', 'Ca2+']]  # 假设你的特征是 pH, K+, Ca2+
y_train = train_df['标签']
X_test = test_df[['pH', 'K+', 'Ca2+']]  # 同样的特征
y_test = test_df['标签']

# 标签编码
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 定义XGBoost分类器
model = xgb.XGBClassifier(
    objective='multi:softmax',  # 多分类任务
    num_class=4,  # 类别数
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100
)

# 训练模型
model.fit(X_train, y_train_encoded)

# 对测试集进行预测
y_pred = model.predict(X_test)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test_encoded, y_pred)

# 归一化混淆矩阵（显示为百分比）
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# 输出分类报告
print("分类报告:")
target_names = [str(name) for name in label_encoder.classes_]
print(classification_report(y_test_encoded, y_pred, target_names=target_names))

# 使用Seaborn的heatmap方法绘制混淆矩阵
conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
print("Normalized Confusion Matrix:")
print(conf_matrix)

# 绘制热图
sns.heatmap(conf_matrix, annot=True, fmt=".2%", cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()
# ========== SHAP分析部分 ==========
print("Running SHAP analysis...")
# 使用SHAP进行解释
feature_names = ["pH", "K+", "Ca2+"]

# 创建SHAP解释器
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)



# 保存每个类别的平均SHAP值（可选）
class_shap_data = []
for i, sv in enumerate(shap_values):
    class_data = pd.DataFrame({
        'feature': feature_names,
        'mean_shap': np.mean(sv, axis=0),
        'mean_abs_shap': np.mean(np.abs(sv), axis=0),
        'class': i
    })
    class_shap_data.append(class_data)

# 合并所有类别的数据
all_class_data = pd.concat(class_shap_data)
all_class_data.to_csv("shap_by_class_data.excel", index=False)


# 假设 model、shap_values 和 feature_names 已经定义
for i in range(model.n_classes_):
    # 创建一个图形对象，修改比例
    fig, ax1 = plt.subplots(figsize=(8, 8))  # 修改图形的宽高比例，宽度12，高度8

    # 设置全局字体
    plt.rc('font', size=20, family='Arial')  # 设置字体大小和类型

    # 绘制SHAP蜂巢图
    shap.summary_plot(
        shap_values[i],
        X_train,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        color=plt.cm.coolwarm
    )
    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置

    # 获取当前的SHAP蜂巢图的轴
    ax1 = plt.gca()

    # 创建共享y轴的另一个图，绘制特征贡献图
    ax2 = ax1.twiny()

    # 重新绘制SHAP特征贡献图（柱状图）
    shap.summary_plot(
        shap_values[i],
        X_train,
        plot_type="bar",
        show=False
    )
    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置，与蜂巢图对齐

    # 获取柱状图对象并调整透明度
    bars = ax2.patches
    for bar in bars:
        bar.set_alpha(0.2)  # 设置透明度

    # 设置顶部X轴的位置和刻度
    ax2.xaxis.set_label_position('top')  # 将标签移动到顶部
    ax2.xaxis.tick_top()  # 将刻度也移动到顶部
    ax2.set_xlabel('Mean Shapley Value', fontsize=10)  # 增大字体

    # 确保顶部X轴的线可见
    ax2.spines['top'].set_visible(True)

    # 设置底部X轴的标签
    ax1.set_xlabel('Shapley Value Contribution', fontsize=10)  # 增大字体

    # 调整子图布局，确保柱状图和蜂巢图的位置和大小适当
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    # 设置顶部X轴的刻度
    ax2.tick_params(axis='x', labelsize=12)

    # 显示最终结果
    plt.tight_layout()

    # 保存图像并设置分辨率
    plt.savefig(f'shap_plot_class_{i}.jpg', transparent=True, dpi=3000)
    plt.show()

# -------------------- SHAP 决策图（所有类别合并） --------------------
expected_value = explainer.expected_value
shap_values_ = np.array(shap_values).transpose(1, 0, 2)
stacked_shap = np.vstack(shap_values_)
stacked_expected_value = np.mean(expected_value)

# 创建正方形图形并强制设置宽高比
plt.figure(figsize=(12, 12), dpi=600)
ax = plt.gca()  # 获取当前坐标轴

# 绘制决策图
shap.decision_plot(
    stacked_expected_value,
    stacked_shap,
    feature_names=feature_names,
    show=False,
    ignore_warnings=True
)

# 关键：强制设置正方形比例
ax.set_aspect('auto')  # 先重置比例
ax.set_position([0.1, 0.1, 0.8, 0.8])  # 调整绘图区域位置和大小
ax.set_box_aspect(1)  # 1:1 宽高比 (核心解决方案)

plt.show()

# -------------------- Precision-Recall 曲线 --------------------
plt.figure(figsize=(8, 8))  # 也改为正方形
y_pred_prob = model.predict_proba(X_test)

for i in range(model.n_classes_):
    precision, recall, _ = precision_recall_curve(y_test_encoded == i, y_pred_prob[:, i])
    ap_score = average_precision_score(y_test_encoded == i, y_pred_prob[:, i])
    plt.plot(recall, precision, label=f'Class {i} (AP={ap_score:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.gca().set_aspect('equal')  # 强制PR曲线也是正方形
plt.tight_layout()
plt.show()
