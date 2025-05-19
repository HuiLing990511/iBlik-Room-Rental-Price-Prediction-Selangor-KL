
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import lightgbm as lgb
from contextlib import redirect_stdout

with open('random_output_log.txt', 'w', encoding='utf-8') as log_file:
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

        # 参数空间
        param_dist = {
            'learning_rate': uniform(0.01, 0.09),
            'max_depth': randint(4, 10),
            'num_leaves': randint(20, 100),
            'min_child_samples': randint(5, 50),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0.0, 1.0),
            'reg_lambda': uniform(0.0, 1.0),
            'n_estimators': randint(100, 400)
        }

        model = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression', verbosity=-1, random_state=42)

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=30,
            scoring='r2',
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=1  # 避免多进程导致日志丢失
        )

        random_search.fit(X_train, y_train)

        print("Best parameters:", random_search.best_params_)

        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)
        r2, rmse, mae, mape = evaluate_all(y_test, y_pred, "RandomSearch LGBM Final (Correct Log)")

        with open("lgbm_random_final_model.pkl", "wb") as f_model:
            pickle.dump(best_model, f_model)

        lgb.plot_importance(best_model, max_num_features=20)
        plt.title("RandomSearch LightGBM Feature Importance")
        plt.tight_layout()
        plt.savefig("random_final_feature_importance.png")
        plt.close()
