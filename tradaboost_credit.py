import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

class TrAdaBoost:
    def __init__(self, base_estimator=None, n_estimators=50, source_weight=1.0, target_weight=10.0):
        """
        TrAdaBoost算法实现，用于迁移学习场景
        
        参数:
            base_estimator: 基分类器，默认为决策树桩(单层决策树)
            n_estimators: 迭代训练的基分类器数量
            source_weight: 源域样本的初始权重系数
            target_weight: 目标域样本的初始权重系数
        """
        self.base_estimator = base_estimator or DecisionTreeClassifier(max_depth=1)
        self.n_estimators = n_estimators
        self.source_weight = source_weight  # 源域样本初始权重系数
        self.target_weight = target_weight  # 目标域样本初始权重系数
        self.estimators_ = []  # 存储训练好的基分类器
        self.estimator_weights_ = []  # 存储每个基分类器的权重
        
    def fit(self, X_source, y_source, X_target, y_target):
        """
        训练TrAdaBoost模型
        
        参数:
            X_source: 源域特征矩阵
            y_source: 源域标签
            X_target: 目标域特征矩阵
            y_target: 目标域标签
        """
        # 合并源域和目标域数据
        X = np.vstack((X_source, X_target))
        y = np.hstack((y_source, y_target))
        
        # 样本数和类别数
        n_samples = len(y)
        n_source = len(y_source)
        n_target = len(y_target)
        
        # 初始化样本权重
        # 源域样本赋予较低初始权重，目标域样本赋予较高初始权重
        weights = np.ones(n_samples)
        weights[:n_source] *= self.source_weight  # 源域样本权重
        weights[n_source:] *= self.target_weight  # 目标域样本权重
        weights /= np.sum(weights)  # 归一化权重，确保总和为1
        
        for i in range(self.n_estimators):
            # 训练弱分类器，使用当前样本权重
            estimator = self.base_estimator.__class__()
            estimator.fit(X, y, sample_weight=weights)
            self.estimators_.append(estimator)
            
            # 预测所有样本
            y_pred = estimator.predict(X)
            
            # 计算误差率（仅考虑目标域样本）
            # 关键差异：传统AdaBoost考虑所有样本，而TrAdaBoost仅关注目标域
            error_mask = (y_pred[n_source:] != y_target)
            error_rate = np.sum(weights[n_source:] * error_mask) / np.sum(weights[n_source:])
            
            # 计算分类器权重（误差率越小，权重越大）
            alpha = 0.5 * np.log((1 - error_rate) / max(error_rate, 1e-10))
            self.estimator_weights_.append(alpha)
            
            # 更新样本权重（仅调整源域样本）
            # 关键机制：若源域样本被误分类，降低其权重；若正确分类，提高权重
            # 这使得后续分类器更关注目标域而非源域
            exp_factor = np.ones(n_samples)
            exp_factor[:n_source] = np.exp(-alpha * y_pred[:n_source] * y_source)
            weights *= exp_factor
            weights /= np.sum(weights)  # 归一化权重
            
        return self
    
    def predict(self, X):
        """对输入数据进行预测"""
        # 加权投票：综合所有基分类器的预测结果
        predictions = np.zeros(len(X))
        for i, estimator in enumerate(self.estimators_):
            predictions += self.estimator_weights_[i] * estimator.predict(X)
        return np.sign(predictions)  # 符号函数，将结果转为±1
    
    def predict_proba(self, X):
        """预测样本属于正类的概率"""
        # 计算概率（需要基分类器支持predict_proba方法）
        proba = np.zeros((len(X), 2))
        for i, estimator in enumerate(self.estimators_):
            if hasattr(estimator, 'predict_proba'):
                proba += self.estimator_weights_[i] * estimator.predict_proba(X)
        return proba / np.sum(self.estimator_weights_)  # 归一化概率

# 模拟信贷数据（源域和目标域）
def generate_credit_data(n_source=1000, n_target=200, n_features=20, random_state=42):
    """
    生成模拟信贷数据，用于测试TrAdaBoost
    
    参数:
        n_source: 源域样本量（已有信贷用户）
        n_target: 目标域样本量（新用户）
        n_features: 特征数量
        random_state: 随机种子，保证结果可复现
    """
    # 源域数据（已有信贷用户）
    # 生成分类数据，包含n_informative个信息特征和n_redundant个冗余特征
    X_source, y_source = make_classification(
        n_samples=n_source,
        n_features=n_features,
        n_informative=10,  # 信息特征数
        n_redundant=5,    # 冗余特征数
        random_state=random_state
    )
    
    # 目标域数据（新用户，分布略有偏移）
    # 与源域类似但有分布偏移，模拟实际场景中新用户与老用户的差异
    X_target, y_target = make_classification(
        n_samples=n_target,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        shift=0.2,  # 特征分布偏移，模拟新用户与老用户的差异
        random_state=random_state+1
    )
    
    return X_source, y_source, X_target, y_target

# 实验比较TrAdaBoost与传统AdaBoost
def compare_models():
    """比较TrAdaBoost与传统AdaBoost在信贷数据上的性能"""
    # 生成数据
    X_source, y_source, X_target, y_target = generate_credit_data()
    
    # 划分目标域数据为训练集和测试集
    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        X_target, y_target, test_size=0.5, random_state=42
    )
    
    # 传统AdaBoost（仅使用目标域训练数据）
    from sklearn.ensemble import AdaBoostClassifier
    ada = AdaBoostClassifier(n_estimators=50)
    ada.fit(X_target_train, y_target_train)
    y_pred_ada = ada.predict(X_target_test)
    y_prob_ada = ada.predict_proba(X_target_test)[:, 1]
    
    # TrAdaBoost（使用源域和目标域训练数据）
    tradaboost = TrAdaBoost(n_estimators=50, source_weight=1.0, target_weight=5.0)
    tradaboost.fit(
        X_source, y_source,
        X_target_train, y_target_train
    )
    y_pred_tradaboost = tradaboost.predict(X_target_test)
    y_prob_tradaboost = tradaboost.predict_proba(X_target_test)[:, 1]
    
    # 评估模型
    results = {
        'AdaBoost': {
            'accuracy': accuracy_score(y_target_test, y_pred_ada),
            'auc': roc_auc_score(y_target_test, y_prob_ada)
        },
        'TrAdaBoost': {
            'accuracy': accuracy_score(y_target_test, y_pred_tradaboost),
            'auc': roc_auc_score(y_target_test, y_prob_tradaboost)
        }
    }
    
    # 可视化结果
    plt.figure(figsize=(10, 5))
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, [results['AdaBoost']['accuracy'], results['AdaBoost']['auc']],
            width, label='AdaBoost')
    plt.bar(x + width/2, [results['TrAdaBoost']['accuracy'], results['TrAdaBoost']['auc']],
            width, label='TrAdaBoost')
    
    plt.ylabel('Score')
    plt.title('模型性能比较')
    plt.xticks(x, ['Accuracy', 'AUC'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    return results

if __name__ == "__main__":
    results = compare_models()
    print("模型性能比较结果:")
    for model, metrics in results.items():
        print(f"{model}:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  AUC值: {metrics['auc']:.4f}")
        print()