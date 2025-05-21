import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import optuna
import sys
import time
import datetime

def save_detailed_log(method_name, best_params, cv_r2, test_r2, rmse, mae, mape, filename, duration=None):
    now = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
    log_lines = []
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

sys.stdout = open("rf_optuna_output_log.txt", "w", encoding="utf-8")

with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 10, 30, step=5),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20, step=2),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 12, step=2),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.7]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True]),
        'oob_score': trial.suggest_categorical('oob_score', [True, False]),
        'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.01)
    }
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    return cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()

start = time.time()
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
duration = time.time() - start

best_params = study.best_params
cv_r2 = study.best_value
best_rf = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)

test_r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

save_detailed_log("Optuna", best_params, cv_r2, test_r2, rmse, mae, mape, "rf_optuna_output_log.txt", duration)

with open("rf_optuna_model.pkl", "wb") as f:
    pickle.dump(best_rf, f)

importances = best_rf.feature_importances_
feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"Feature {{i}}" for i in range(X_train.shape[1])]
sorted_idx = np.argsort(importances)

plt.figure(figsize=(8, 6))
bars = plt.barh(range(len(importances)), np.array(importances)[sorted_idx], align='center')
plt.yticks(range(len(importances)), np.array(feature_names)[sorted_idx])
plt.xlabel("Feature importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance (Optuna)")

for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.002, bar.get_y() + bar.get_height()/2,
             f"{width:.3f}", va='center', fontsize=8)

plt.tight_layout()
plt.savefig("rf_optuna_feature_importance.png")
plt.show()
