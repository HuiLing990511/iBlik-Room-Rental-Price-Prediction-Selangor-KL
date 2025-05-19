
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from contextlib import redirect_stdout

with open('grid_output_log.txt', 'w', encoding='utf-8') as log_file:
    with redirect_stdout(log_file):
        # 加载数据
        with open('preprocessed_data.pkl', 'rb') as f:
            data = pickle.load(f)
            X_train = data['X_train']
            X_test = data['X_test']
            y_train = data['y_train']
            y_test = data['y_test']

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

        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [4, 6, 8],
            'num_leaves': [31, 50],
            'min_child_samples': [10, 20],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0.0, 0.5],
            'reg_lambda': [0.0, 0.5],
            'n_estimators': [100, 200]
        }

        model = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression', verbosity=-1, random_state=42)

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='r2',
            cv=3,
            verbose=2,
            n_jobs=1  # 单线程确保日志捕获
        )

        grid.fit(X_train, y_train)

        print("Best parameters:", grid.best_params_)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        r2, rmse, mae, mape = evaluate_all(y_test, y_pred, "GridSearch LGBM Final (Correct Log)")

        with open("lgbm_grid_final_model.pkl", "wb") as f_model:
            pickle.dump(best_model, f_model)

        lgb.plot_importance(best_model, max_num_features=20)
        plt.title("GridSearch LightGBM Feature Importance")
        plt.tight_layout()
        plt.savefig("grid_final_feature_importance.png")
        plt.close()
