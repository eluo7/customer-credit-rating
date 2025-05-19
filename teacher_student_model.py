import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子确保结果可复现
np.random.seed(42)

# ======================================
# 1. 生成模拟数据
# ======================================
def generate_sample_data(n_samples=1000):
    """生成模拟信用风险数据"""
    data = pd.DataFrame({
        'income': np.random.normal(50, 30, n_samples).clip(10, 200),  # 收入（千元）
        'age': np.random.normal(35, 10, n_samples).clip(18, 70),      # 年龄
        'years_employed': np.random.gamma(5, 1, n_samples).clip(0, 40),  # 工作年限
        'debt_ratio': np.random.beta(2, 5, n_samples),  # 负债率 (0-1)
    })
    
    # 添加一些非线性关系来模拟真实世界的复杂性
    default_prob = 1 / (1 + np.exp(
        -(-3 
          + 0.03 * (data['income'] - 50) 
          - 0.05 * (data['age'] - 35)
          - 0.5 * (data['years_employed'] - 5)
          + 5 * (data['debt_ratio'] - 0.3)
          - 0.001 * data['income'] * data['debt_ratio']
          + 0.005 * np.maximum(data['age'] - 60, 0)
         )
    ))
    
    # 根据违约概率生成二值违约标签
    data['default'] = np.random.binomial(1, default_prob)
    
    return data

# 生成数据并划分训练测试集
data = generate_sample_data(5000)
X = data.drop('default', axis=1)
y = data['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ======================================
# 2. 训练教师模型（黑盒模型）
# ======================================
def train_teacher_model(X_train, y_train):
    """训练随机森林作为教师模型"""
    teacher = RandomForestClassifier(
        n_estimators=100, 
        max_depth=6,
        min_samples_leaf=50,
        random_state=42
    )
    teacher.fit(X_train, y_train)
    return teacher

teacher_model = train_teacher_model(X_train, y_train)

# 计算训练集上的预测概率
X_train_with_probs = X_train.copy()
X_train_with_probs['teacher_prob'] = teacher_model.predict_proba(X_train)[:, 1]

# 评估教师模型性能
y_pred_probs = teacher_model.predict_proba(X_test)[:, 1]
print(f"教师模型 AUC: {roc_auc_score(y_test, y_pred_probs):.4f}")

# ======================================
# 3. 基于教师模型预测概率的分箱函数
# ======================================
def create_probability_driven_bins(feature, teacher_probs, n_bins=3, strategy='quantile'):
    """
    基于教师模型的预测概率创建特征分箱
    
    参数:
    - feature: 待分箱的特征值
    - teacher_probs: 教师模型的预测概率
    - n_bins: 分箱数量
    - strategy: 分箱策略 ('quantile' 或 'uniform')
    
    返回:
    - bins: 分箱边界
    - bin_labels: 每个样本的分箱标签
    - bin_stats: 每个分箱的统计信息
    """
    # 创建一个包含特征值和教师概率的DataFrame
    df = pd.DataFrame({
        'feature': feature,
        'teacher_prob': teacher_probs
    })
    
    # 根据策略进行分箱
    if strategy == 'quantile':
        # 等频分箱（每个分箱包含相同数量的样本）
        df['bin'] = pd.qcut(df['teacher_prob'], q=n_bins, labels=False)
    else:
        # 等距分箱（每个分箱的宽度相同）
        df['bin'] = pd.cut(df['teacher_prob'], bins=n_bins, labels=False)
    
    # 计算每个分箱的边界和统计信息
    bins = []
    bin_stats = {}
    
    for bin_id in sorted(df['bin'].unique()):
        bin_data = df[df['bin'] == bin_id]
        feature_min = bin_data['feature'].min()
        feature_max = bin_data['feature'].max()
        avg_prob = bin_data['teacher_prob'].mean()
        count = len(bin_data)
        
        bins.append((feature_min, feature_max))
        bin_stats[bin_id] = {
            'feature_range': (feature_min, feature_max),
            'avg_teacher_prob': avg_prob,
            'sample_count': count
        }
    
    # 为每个样本分配分箱标签（基于特征值范围）
    bin_labels = []
    for value in feature:
        for bin_id, (min_val, max_val) in enumerate(bins):
            if min_val <= value <= max_val:
                bin_labels.append(bin_id)
                break
    
    return bins, bin_labels, bin_stats

# ======================================
# 4. 对收入特征进行分箱演示
# ======================================
feature_name = 'income'
n_bins = 3  # 分为3个风险等级

bins, bin_labels, bin_stats = create_probability_driven_bins(
    X_train_with_probs[feature_name],
    X_train_with_probs['teacher_prob'],
    n_bins=n_bins
)

# 打印分箱结果
print(f"\n{feature_name} 基于教师模型预测概率的分箱结果:")
for bin_id, stats in bin_stats.items():
    print(f"分箱 {bin_id}: {stats['feature_range'][0]:.1f} - {stats['feature_range'][1]:.1f} 千元")
    print(f"  样本数: {stats['sample_count']}")
    print(f"  平均教师模型预测概率: {stats['avg_teacher_prob']:.4f}")
    print("-" * 40)

# ======================================
# 5. 可视化分箱结果
# ======================================
plt.figure(figsize=(12, 6))

# 绘制原始数据分布
plt.subplot(1, 2, 1)
sns.histplot(X_train_with_probs[feature_name], bins=30, kde=True)
plt.title(f'{feature_name} 原始分布')
plt.xlabel(f'{feature_name} (千元)')
plt.ylabel('样本数')

# 绘制基于预测概率的分箱
plt.subplot(1, 2, 2)
for bin_id, (min_val, max_val) in enumerate(bins):
    bin_data = X_train_with_probs[
        (X_train_with_probs[feature_name] >= min_val) & 
        (X_train_with_probs[feature_name] <= max_val)
    ]
    sns.histplot(bin_data[feature_name], bins=10, kde=True, 
                 label=f'分箱 {bin_id} (概率: {bin_stats[bin_id]["avg_teacher_prob"]:.4f})')

plt.title(f'{feature_name} 基于教师模型预测概率的分箱')
plt.xlabel(f'{feature_name} (千元)')
plt.ylabel('样本数')
plt.legend()

plt.tight_layout()
plt.savefig('probability_driven_binning.png', dpi=300)
plt.show()

# ======================================
# 6. 构建学生模型（评分卡）
# ======================================
def build_scorecard(X_train, y_train, teacher_model, feature_bins):
    """
    构建基于分箱的评分卡模型
    
    参数:
    - X_train: 训练特征
    - y_train: 训练标签
    - teacher_model: 教师模型
    - feature_bins: 特征分箱规则
    
    返回:
    - scorecard_model: 评分卡模型
    - scorecard_rules: 评分卡规则
    """
    # 创建一个包含分箱特征的新DataFrame
    X_train_binned = pd.DataFrame(index=X_train.index)
    
    # 对每个特征进行分箱
    for feature in X_train.columns:
        if feature in feature_bins:
            bins = feature_bins[feature]
            bin_labels = []
            
            for value in X_train[feature]:
                for bin_id, (min_val, max_val) in enumerate(bins):
                    if min_val <= value <= max_val:
                        bin_labels.append(bin_id)
                        break
            
            X_train_binned[f'{feature}_bin'] = bin_labels
        else:
            # 对于没有指定分箱规则的特征，使用原始值
            X_train_binned[feature] = X_train[feature]
    
    # 添加教师模型的预测概率作为一个特征
    X_train_binned['teacher_prob'] = teacher_model.predict_proba(X_train)[:, 1]
    
    # 训练逻辑回归模型作为评分卡
    scorecard_model = LogisticRegression(random_state=42)
    scorecard_model.fit(X_train_binned, y_train)
    
    # 生成评分卡规则
    scorecard_rules = {
        'intercept': scorecard_model.intercept_[0],
        'coefficients': dict(zip(X_train_binned.columns, scorecard_model.coef_[0]))
    }
    
    return scorecard_model, scorecard_rules, X_train_binned

# 为多个特征创建分箱规则
feature_bins = {}
for feature in ['income', 'age', 'years_employed', 'debt_ratio']:
    bins, _, _ = create_probability_driven_bins(
        X_train_with_probs[feature],
        X_train_with_probs['teacher_prob'],
        n_bins=3
    )
    feature_bins[feature] = bins

# 构建评分卡模型
scorecard_model, scorecard_rules, X_train_binned = build_scorecard(
    X_train, y_train, teacher_model, feature_bins
)

# 评估评分卡模型性能
X_test_binned = pd.DataFrame(index=X_test.index)
for feature in X_test.columns:
    if feature in feature_bins:
        bins = feature_bins[feature]
        bin_labels = []
        for value in X_test[feature]:
            for bin_id, (min_val, max_val) in enumerate(bins):
                if min_val <= value <= max_val:
                    bin_labels.append(bin_id)
                    break
        X_test_binned[f'{feature}_bin'] = bin_labels
    else:
        X_test_binned[feature] = X_test[feature]

X_test_binned['teacher_prob'] = teacher_model.predict_proba(X_test)[:, 1]
y_pred_scorecard = scorecard_model.predict_proba(X_test_binned)[:, 1]

print(f"\n评分卡模型 AUC: {roc_auc_score(y_test, y_pred_scorecard):.4f}")

# ======================================
# 7. 打印评分卡规则示例
# ======================================
print("\n评分卡规则示例:")
print(f"基础分: {scorecard_rules['intercept']:.2f}")
for feature, coef in scorecard_rules['coefficients'].items():
    print(f"{feature} 系数: {coef:.4f}")

# ======================================
# 8. 比较不同分箱方法
# ======================================
# 传统分箱（等距分箱）
def traditional_binning(feature, n_bins=3):
    """传统等距分箱"""
    bins = pd.cut(feature, bins=n_bins, retbins=True)[1]
    return bins

# 比较两种分箱方法的预测能力
plt.figure(figsize=(10, 6))

# 教师模型驱动的分箱
plt.subplot(1, 2, 1)
for bin_id, (min_val, max_val) in enumerate(bins):
    bin_data = X_train_with_probs[
        (X_train_with_probs[feature_name] >= min_val) & 
        (X_train_with_probs[feature_name] <= max_val)
    ]
    plt.hist(bin_data['teacher_prob'], bins=10, alpha=0.7,
             label=f'分箱 {bin_id} ({min_val:.1f}-{max_val:.1f}千元)')

plt.title('基于教师模型预测概率的分箱')
plt.xlabel('教师模型预测概率')
plt.ylabel('样本数')
plt.legend()

# 传统等距分箱
plt.subplot(1, 2, 2)
traditional_bin_edges = traditional_binning(X_train_with_probs[feature_name], n_bins=3)
for i in range(len(traditional_bin_edges)-1):
    min_val, max_val = traditional_bin_edges[i], traditional_bin_edges[i+1]
    bin_data = X_train_with_probs[
        (X_train_with_probs[feature_name] >= min_val) & 
        (X_train_with_probs[feature_name] < max_val)
    ]
    plt.hist(bin_data['teacher_prob'], bins=10, alpha=0.7,
             label=f'分箱 {i} ({min_val:.1f}-{max_val:.1f}千元)')

plt.title('传统等距分箱')
plt.xlabel('教师模型预测概率')
plt.ylabel('样本数')
plt.legend()

plt.tight_layout()
plt.savefig('binning_comparison.png', dpi=300)
plt.show()