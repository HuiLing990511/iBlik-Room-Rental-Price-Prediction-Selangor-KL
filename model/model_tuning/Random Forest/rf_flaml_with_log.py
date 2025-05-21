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

with open("rf_flaml_model.pkl", "wb") as f:
    pickle.dump(automl.model, f)

y_pred = automl.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

with open("rf_flaml_output_log.txt", "a", encoding="utf-8") as f:
    f.write(f"\nFLAML RF Evaluation Metrics:\n")
    f.write(f"R² Score: {test_r2:.4f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"MAPE: {mape:.2f}%\n")
    # 输出全部主流参数
    f.write("\nFull Model Parameters:\n")
    params = automl.model.get_params()
    main_keys = [
        'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
        'max_features', 'bootstrap', 'oob_score', 'ccp_alpha', 'max_leaf_nodes', 'max_samples'
    ]
    for k in main_keys:
        if k in params:
            f.write(f"{k}: {params[k]}\n")
    for k, v in params.items():
        if k not in main_keys:
            f.write(f"{k}: {v}\n")

importances = automl.model.feature_importances_
feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(X_train.shape[1])]

sorted_idx = np.argsort(importances)

plt.figure(figsize=(8, 6))
bars = plt.barh(range(len(importances)), np.array(importances)[sorted_idx], align='center')
plt.yticks(range(len(importances)), np.array(feature_names)[sorted_idx])
plt.xlabel("Feature importance")
plt.ylabel("Features")
plt.title("FLAML Random Forest Feature Importance")

for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.002, bar.get_y() + bar.get_height()/2,
             f"{width:.3f}", va='center', fontsize=8)

plt.tight_layout()
plt.savefig("rf_flaml_feature_importance.png")
plt.show()
