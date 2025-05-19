
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys
import time
import datetime

# === FLAML style logging ===
def save_detailed_log(method_name, best_params, cv_r2, test_r2, rmse, mae, mape, filename, duration=None):
    now = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
    log_lines = []

    if duration:
        log_lines.append(f"[{method_name.lower()}.logger: {now}] {{2688}} INFO - retrained model: {best_params}")
        log_lines.append(f"[{method_name.lower()}.logger: {now}] {{1985}} INFO - fit succeeded")
        log_lines.append(f"[{method_name.lower()}.logger: {now}] {{1986}} INFO - Time taken to find the best model: {duration:.3f}")
        log_lines.append("")

    log_lines.append(f"{method_name} Evaluation Metrics:")
    log_lines.append(f"R² Score: {test_r2:.4f}")
    log_lines.append(f"RMSE: {rmse:.2f}")
    log_lines.append(f"MAE: {mae:.2f}")
    log_lines.append(f"MAPE: {mape:.2f}%")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

# 输出 log 到 txt 文件
sys.stdout = open("rf_random_output_log.txt", "w", encoding="utf-8")

# 加载数据
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

# 随机搜索参数网格
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 15, 20, 25, 30],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8, 12],
    'max_features': ['sqrt', 'log2', 0.7],
    'bootstrap': [True],
    'oob_score': [True, False],
    'ccp_alpha': [0.0, 0.001, 0.01]
}

# 建模并调参
model = RandomForestRegressor(random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(model, param_dist, n_iter=50, cv=5, scoring='r2', verbose=2, n_jobs=-1, random_state=42)

start = time.time()
random_search.fit(X_train, y_train)
duration = time.time() - start

# 获取最佳模型与评估结果
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)

test_r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# 保存 FLAML 风格日志
save_detailed_log("RandomSearchCV", random_search.best_params_, random_search.best_score_,
                  test_r2, rmse, mae, mape, "rf_random_output_log.txt", duration)

# 保存模型
with open("rf_random_model.pkl", "wb") as f:
    pickle.dump(best_rf, f)

# 输出特征重要性图
importances = best_rf.feature_importances_
plt.figure(figsize=(12,6))
plt.bar(range(len(importances)), importances)
plt.title("Random Forest Feature Importance (Randomized Search)")
plt.savefig("rf_random_feature_importance.png")
plt.close()
