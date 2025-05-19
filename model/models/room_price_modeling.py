import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import logging
import sys
from datetime import datetime
import os

# 设置日志记录
log_filename = f'room_price_modeling_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

# 创建模型结果文件夹，如果不存在
if not os.path.exists('models_result'):
    os.makedirs('models_result')

# 1. 读取数据
logging.info("Reading data...")
df = pd.read_csv('cleaned_with_count.csv')
logging.info(f"Data shape: {df.shape}")

# 2. 数据分割
logging.info("\nSplitting data...")
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 特征工程
logging.info("\nPerforming feature engineering...")

# 3.1 处理分类特征
categorical_features = ['Room Type', 'State', 'Location Detail']
le = LabelEncoder()

# 合并训练集和测试集进行编码，以确保测试集不会包含训练集未见过的标签
for feature in categorical_features:
    # 先合并训练集和测试集
    all_data = pd.concat([X_train[feature], X_test[feature]], axis=0)

    # 对整个合并后的数据进行fit_transform
    le.fit(all_data.astype(str))

    # 对训练集和测试集分别进行transform
    X_train[feature + '_encoded'] = le.transform(X_train[feature].astype(str))
    X_test[feature + '_encoded'] = le.transform(X_test[feature].astype(str))  # Apply same transformation to test set

# 3.2 设施相关特征（二进制特征）
facility_features = [
    'Air-Conditioning', 'Washing Machine', 'Wifi / Internet Access',
    'Cooking Allowed', 'TV', 'Share Bathroom', 'Private Bathroom',
    'Near KTM / LRT', 'Near LRT / MRT', 'Near KTM', 'Near LRT', 'Near MRT',
    'Near Train', 'Near Bus stop', '24 hours security', 'Swimming Pools',
    'Gymnasium Facility', 'OKU Friendly', 'Multi-purpose hall', 'Playground',
    'Covered car park', 'Surau', 'Mini Market', 'Co-Living'
]

# 3.3 选择最终特征
final_features = ['Latitude', 'Longitude'] + \
                 [f + '_encoded' for f in categorical_features] + \
                 facility_features

# 3.4 准备建模数据
X_train = X_train[final_features]
X_test = X_test[final_features]

# 3.5 特征标准化（对于某些模型需要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# 4. 特征选择 - 使用F回归选择最重要的特征（仅在训练集上进行）
logging.info("\nPerforming feature selection using F-regression...")

# 使用F-regression特征选择方法
selector = SelectKBest(f_regression, k=5)
selector.fit(X_train_scaled, y_train)  # 仅在训练集上进行特征选择

# 获取特征分数和p值
feature_scores = pd.DataFrame({
    'Feature': X_train.columns,
    'F-Score': selector.scores_,
    'P-value': selector.pvalues_
})
feature_scores = feature_scores.sort_values('F-Score', ascending=False)

logging.info("\nTop 5 features selected by F-regression:")
logging.info("\n" + str(feature_scores.head(5)))

# 获取选定的特征
selected_features = feature_scores.head(5)['Feature'].values
logging.info(f"Selected features: {selected_features}")

# 创建选定特征的数据集
X_train_selected = X_train_scaled[selected_features]
X_test_selected = X_test_scaled[selected_features]

# 5. 保存预处理后的数据，供model_tuning.py使用
logging.info("Saving preprocessed data for model tuning...")
preprocessed_data = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'X_train_selected': X_train_selected,
    'X_test_selected': X_test_selected,
    'selected_features': selected_features
}
with open('models_result/preprocessed_data.pkl', 'wb') as f:
    pickle.dump(preprocessed_data, f)

# 6. 模型训练和评估
def evaluate_model(y_true, y_pred, model_name):
    # 回归指标
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    logging.info(f"\n{model_name} Metrics:")
    logging.info(f"R2 Score: {r2:.4f}")
    logging.info(f"RMSE: {rmse:.2f}")
    logging.info(f"MAE: {mae:.2f}")
    logging.info(f"MAPE: {mape:.2f}%")

    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

# 定义所有模型
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'LightGBM': lgb.LGBMRegressor(random_state=42),
    'SVR': SVR(kernel='rbf'),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

# 在模型训练之前添加数据可视化部分
def create_visualizations(df, y_test, best_pred):
    logging.info("\nCreating visualizations...")

    # 1. 价格分布图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='Price', bins=50)
    plt.title('Price Distribution')
    plt.savefig('models_result/price_distribution.png')

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['Price'])
    plt.title('Price Box Plot')
    plt.tight_layout()
    plt.savefig('models_result/price_box_plot.png')

    # 2. 相关性热力图
    plt.figure(figsize=(15, 12))
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('models_result/correlation_heatmap.png')

    # 3. 不同地区的价格箱线图
    plt.figure(figsize=(15, 6))
    sns.boxplot(x='State', y='Price', data=df)
    plt.xticks(rotation=45)
    plt.title('Price Distribution by State')
    plt.tight_layout()
    plt.savefig('models_result/price_by_state.png')

    # 4. 不同房型的价格对比
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Room Type', y='Price', data=df)
    plt.xticks(rotation=45)
    plt.title('Price Distribution by Room Type')
    plt.tight_layout()
    plt.savefig('models_result/price_by_room_type.png')

    # 5. 设施与价格的关系
    facility_price_means = pd.DataFrame()
    for facility in facility_features:
        facility_price_means[facility] = [
            df[df[facility] == 0]['Price'].mean(),
            df[df[facility] == 1]['Price'].mean()
        ]

    plt.figure(figsize=(15, 6))
    facility_price_means.T.plot(kind='bar')
    plt.legend(['Without Facility', 'With Facility'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Price by Facilities')
    plt.tight_layout()
    plt.savefig('models_result/price_by_facilities.png')

    # 6. 预测值与实际值的对比散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, best_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.tight_layout()
    plt.savefig('models_result/actual_vs_predicted.png')

    # 7. 预测误差分布图
    errors = y_test - best_pred
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(errors, bins=50)
    plt.title('Prediction Error Distribution')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_test, y=errors)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Price')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Error vs Actual Price')
    plt.tight_layout()
    plt.savefig('models_result/prediction_errors.png')

    # 8. 地理位置与价格的关系
    plt.figure(figsize=(12, 8))
    plt.scatter(df['Longitude'], df['Latitude'],
                c=df['Price'], cmap='viridis',
                alpha=0.6)
    plt.colorbar(label='Price')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographic Distribution of Prices')
    plt.tight_layout()
    plt.savefig('models_result/geographic_prices.png')

# 训练和评估所有模型
logging.info("\nTraining and evaluating models...")
model_metrics = {}
best_r2 = -float('inf')
best_model = None
best_model_name = None

for name, model in models.items():
    logging.info(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, name)
    model_metrics[name] = metrics

    # 更新最佳模型
    if metrics['r2'] > best_r2:
        best_r2 = metrics['r2']
        best_model = model
        best_model_name = name
        best_pred = y_pred  # 保存最佳预测结果

# 创建模型比较DataFrame
models_comparison = pd.DataFrame({
    'Model': list(model_metrics.keys()),
    'R2 Score': [metrics['r2'] for metrics in model_metrics.values()],
    'RMSE': [metrics['rmse'] for metrics in model_metrics.values()],
    'MAE': [metrics['mae'] for metrics in model_metrics.values()],
    'MAPE': [metrics['mape'] for metrics in model_metrics.values()]
})

# 按R2分数排序
models_comparison = models_comparison.sort_values('R2 Score', ascending=False)
logging.info("\nModel Comparison:")
logging.info("\n" + str(models_comparison))
models_comparison.to_csv('models_result/model_comparison.csv', index=False)

# 可视化模型比较
plt.figure(figsize=(15, 8))
sns.barplot(x='Model', y='R2 Score', data=models_comparison)
plt.xticks(rotation=45, ha='right')
plt.title('Model Comparison - R2 Scores')
plt.tight_layout()
plt.savefig('models_result/model_comparison.png')

# 如果是随机森林模型，显示特征重要性
if isinstance(best_model, RandomForestRegressor):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig('models_result/feature_importance.png')

# 在保存最佳模型之前添加可视化调用
create_visualizations(df, y_test, best_pred)

# 保存最佳模型
logging.info(f"\nSaving the best model ({best_model_name})...")
with open('models_result/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# 获取前三个表现最好的模型
top_models = models_comparison.head(3)['Model'].values
logging.info(f"\nTop 3 models for tuning: {top_models}")
logging.info("To tune these models, please run model_tuning.py")

logging.info("\nModeling complete! Results have been saved.")
logging.info(f"Detailed log saved to {log_filename}")
