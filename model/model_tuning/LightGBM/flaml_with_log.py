import sys
sys.stdout = open('flaml_output_log.txt', 'w', encoding='utf-8')
sys.stderr = sys.stdout
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from flaml import AutoML

# 加载预处理数据
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

def evaluate_all(y_true, y_pred, name="FLAML"):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n{name} Evaluation Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    return r2, rmse, mae, mape

# 初始化 FLAML
automl = AutoML()

settings = {
    "time_budget": 300,
    "metric": 'r2',
    "task": 'regression',
    "log_file_name": "flaml_lgbm.log",
    "estimator_list": ["lgbm"],
}

automl.fit(X_train=X_train, y_train=y_train, **settings)

# 预测与评估
y_pred = automl.predict(X_test)
r2, rmse, mae, mape = r2, rmse, mae, mape = evaluate_all(y_test, y_pred, "LGBM FLAML")

# 特征重要性可视化（如果支持）
try:
    import lightgbm as lgb
    booster = automl.model.estimator.booster_
    lgb.plot_importance(booster, max_num_features=20)
    plt.title("FLAML LightGBM Feature Importance")
    plt.tight_layout()
    plt.savefig("flaml_feature_importance.png")
    plt.show()
except Exception as e:
    print("Feature importance plot skipped:", e)

# 保存模型
with open("lgbm_flaml_model.pkl", "wb") as f:
    pickle.dump(automl, f)

# 将最终评估指标附加写入同一日志文件
with open("flaml_output_log.txt", "a", encoding="utf-8") as f_log:
    f_log.write("\nFinal Evaluation:\n")
    f_log.write(f"R² Score: {r2:.4f}\n")
    f_log.write(f"RMSE: {rmse:.2f}\n")
    f_log.write(f"MAE: {mae:.2f}\n")
    f_log.write(f"MAPE: {mape:.2f}%\n")
