import xgboost
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 加载加州房价数据集
X, y = shap.datasets.california()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练XGBoost模型
model = xgboost.XGBRegressor().fit(X_train, y_train)

# 评估模型性能
y_pred = model.predict(X_test)
print(f"模型准确率: {model.score(X_test, y_test)}")

# 计算SHAP值
explainer = shap.Explainer(model)
shap_values = explainer(X) # Explainer对象

# 瀑布图
def plot1():
    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title('SHAP Waterfall Plot for First Prediction')
    plt.tight_layout()
    plt.savefig('waterfall.png')
    plt.show()
    plt.close()

# 柱状图
def plot2():
    shap.plots.bar(shap_values, show=False)
    plt.title('SHAP Bar Plot')
    plt.tight_layout()
    plt.savefig('bar.png')
    plt.show()
    plt.close()

# 蜂群图
def plot3():
    shap.plots.beeswarm(shap_values, show=False)
    plt.title('SHAP Beeswarm Plot')
    plt.tight_layout()
    plt.savefig('beeswarm.png')
    plt.show()
    plt.close()

# 散点图
def plot4():
    shap.plots.scatter(shap_values[:, "Latitude"], color=shap_values, show=False)
    plt.title('SHAP Scatter Plot for Latitude')
    plt.tight_layout()
    plt.savefig('scatter.png')
    plt.show()
    plt.close()

# # 引力图：有问题
# def plot5():
#     shap.plots.force(shap_values[100], show=False)
#     plt.title('SHAP Force Plot for First Prediction')
#     plt.tight_layout()
#     plt.savefig('force.png')
#     plt.show()
#     plt.close()

# 核密度估计图
def plot6():
    shap.plots.force(shap_values[:500], show=False)
    plt.title('SHAP Force Plot for First 500 Predictions')
    plt.tight_layout()
    plt.savefig('force.png')