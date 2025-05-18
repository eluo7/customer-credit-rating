import pandas as pd
import featuretools as ft
from featuretools.primitives import Mean, Count, Day, Month

# 创建示例数据
users = pd.DataFrame({
    'user_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'signup_date': pd.to_datetime(['2023-01-01', '2023-01-15', '2023-02-01'])
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105],
    'user_id': [1, 1, 2, 3, 3],
    'product_id': ['A', 'B', 'A', 'C', 'B'],
    'amount': [100.50, 200.00, 150.75, 300.25, 120.00],
    'order_date': pd.to_datetime(['2023-01-10', '2023-01-15', '2023-01-20', '2023-02-05', '2023-02-10'])
})

products = pd.DataFrame({
    'product_id': ['A', 'B', 'C', 'D'],
    'category': ['Electronics', 'Clothing', 'Books', 'Electronics'],
    'price': [50.25, 100.00, 30.50, 200.75]
})

# 创建实体集
es = ft.EntitySet(id="ecommerce")

# 添加数据框
es = es.add_dataframe(
    dataframe_name="users",
    dataframe=users,
    index="user_id",
    time_index="signup_date"
)

es = es.add_dataframe(
    dataframe_name="orders",
    dataframe=orders,
    index="order_id",
    time_index="order_date",
    logical_types={"product_id": "Categorical"}
)

es = es.add_dataframe(
    dataframe_name="products",
    dataframe=products,
    index="product_id",
    logical_types={"category": "Categorical"}
)

# 添加关系（新版本写法）
es = es.add_relationship("users", "user_id", "orders", "user_id")
es = es.add_relationship("products", "product_id", "orders", "product_id")

# 执行深度特征合成
features, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="users",
    agg_primitives=[Mean, Count],
    trans_primitives=[Day, Month],
    max_depth=2,
    verbose=True
)

print("\n生成的特征:")
print(features.to_csv(sep='\t', na_rep='nan'))