from flaml import AutoML
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time

with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

automl = AutoML()
settings = {
    "time_budget": 300,  # 5分钟（可调整）
    "task": "regression",
    "estimator_list": ["rf"],
    "log_file_name": "rf_flaml_log.log",
    "eval_method": "cv",
    "n_jobs": -1,
    "seed": 42
}

start = time.time()
automl.fit(X_train=X_train, y_train=y_train, **settings)
duration = time.time() - start

with open("rf_flaml_output_log.txt", "w", encoding="utf-8") as f:
    f.write(f"[flaml.automl.logger] INFO - Best config: {automl.best_config}\n")
    f.write(f"[flaml.automl.logger] INFO - Best loss: {automl.best_loss}\n")
    f.write(f"[flaml.automl.logger] INFO - Time taken to find the best model: {duration:.3f}\n")

# 保存模型
with open("rf_flaml_model.pkl", "wb") as f:
    pickle.dump(automl.model, f)

# 评估
y_pred = automl.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# 日志追加写入评价指标到txt
with open("rf_flaml_output_log.txt", "a", encoding="utf-8") as f:
    f.write(f"\nFLAML RF Evaluation Metrics:\n")
    f.write(f"R² Score: {test_r2:.4f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"MAPE: {mape:.2f}%\n")

importances = automl.model.feature_importances_
plt.figure(figsize=(12,6))
plt.bar(range(len(importances)), importances)
plt.title("Random Forest Feature Importance (FLAML)")
plt.savefig("rf_flaml_feature_importance.png")
plt.close()