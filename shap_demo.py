import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

# 加载示例数据集
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 创建SHAP解释器
explainer = shap.TreeExplainer(model)

# 计算SHAP值
shap_values = explainer.shap_values(X_test)

# 1. 特征重要性总结图（条形图）
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('feature_importance_bar.png')
plt.close()

# 2. SHAP值分布图
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.close()

# 3. 单个预测的局部解释
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], show=False, matplotlib=True)
plt.tight_layout()
plt.savefig('force_plot.png')
plt.close()

# 4. 依赖图
plt.figure(figsize=(10, 6))
shap.dependence_plot("MedInc", shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('dependence_plot.png')
plt.close()

# 打印特征重要性
feature_importance = pd.DataFrame({
    'feature': X_test.columns,
    'importance': np.abs(shap_values).mean(0)
})
print("\n特征重要性排序:")
print(feature_importance.sort_values('importance', ascending=False))
