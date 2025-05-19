import sys
sys.stdout = open('optuna_output_log.txt', 'w', encoding='utf-8')
sys.stderr = sys.stdout
import pandas as pd
import numpy as np
import pickle
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

def evaluate_all(y_true, y_pred, name="Model"):
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

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'feature_pre_filter': False,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[valid_data]
    )
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("Best parameters:", study.best_params)

best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'feature_pre_filter': False
})

final_model = lgb.train(
    best_params,
    train_data,
    num_boost_round=300,
    valid_sets=[valid_data]
)

y_pred = final_model.predict(X_test)
r2, rmse, mae, mape = evaluate_all(y_test, y_pred, "Optuna LGBM Final (Fixed)")

lgb.plot_importance(final_model, max_num_features=20)
plt.title("Optuna LightGBM Feature Importance")
plt.tight_layout()
plt.savefig("optuna_final_fixed_feature_importance.png")
plt.show()

with open("lgbm_optuna_final_fixed_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open("optuna_output_log.txt", "a", encoding="utf-8") as f_log:
    f_log.write("\nFinal Evaluation:\n")
    f_log.write(f"R² Score: {r2:.4f}\n")
    f_log.write(f"RMSE: {rmse:.2f}\n")
    f_log.write(f"MAE: {mae:.2f}\n")
    f_log.write(f"MAPE: {mape:.2f}%\n")
