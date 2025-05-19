import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt import fmin, tpe, hp, Trials
import optuna

# 加载数据
with open('models_result/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']  # 用所有特征
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

output_dir = 'xgboost_models_result'
os.makedirs(output_dir, exist_ok=True)

def plot_feature_importance(model, feature_names, method_name):
    importance = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    df_imp = df_imp.sort_values(by='Importance', ascending=False).head(20)
    plt.figure(figsize=(10,6))
    sns.barplot(x='Importance', y='Feature', data=df_imp)
    plt.title(f'Feature Importance - {method_name}')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_{method_name}.png')
    plt.close()

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return r2, rmse, mae, mape

def random_search_tuning(X_train, y_train, X_test, y_test, feature_names):
    print("Running Random Search tuning...")
    xgb_model = xgb.XGBRegressor(random_state=42)
    param_dist = {
        'n_estimators': np.arange(100, 1000, 100),
        'max_depth': np.arange(3, 15),
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 3, 5]
    }
    rs = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=50, cv=3,
                            scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
    rs.fit(X_train, y_train)
    best_params = rs.best_params_
    model = xgb.XGBRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    r2, rmse, mae, mape = evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, feature_names, 'random_search')
    return {'Method':'Random Search', **best_params, 'R2 Score':r2, 'RMSE':rmse, 'MAE':mae, 'MAPE (%)':mape}

def bayesian_optimization_tuning(X_train, y_train, X_test, y_test, feature_names):
    print("Running Bayesian Optimization tuning...")
    def objective(params):
        params = {
            'n_estimators': int(params['n_estimators']),
            'max_depth': int(params['max_depth']),
            'learning_rate': params['learning_rate'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'gamma': params['gamma'],
            'min_child_weight': int(params['min_child_weight']),
            'random_state': 42,
            'objective': 'reg:squarederror'
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return rmse
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
        'max_depth': hp.quniform('max_depth', 3, 15, 1),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'subsample': hp.uniform('subsample', 0.7, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
        'gamma': hp.uniform('gamma', 0, 0.2),
        'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1)
    }
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
    best_params = {
        'n_estimators': int(best['n_estimators']),
        'max_depth': int(best['max_depth']),
        'learning_rate': best['learning_rate'],
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree'],
        'gamma': best['gamma'],
        'min_child_weight': int(best['min_child_weight']),
        'random_state': 42,
        'objective': 'reg:squarederror'
    }
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    r2, rmse, mae, mape = evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, feature_names, 'bayesian_optimization')
    # 不存seed和objective
    best_params.pop('random_state')
    best_params.pop('objective')
    return {'Method':'Bayesian Optimization', **best_params, 'R2 Score':r2, 'RMSE':rmse, 'MAE':mae, 'MAPE (%)':mape}

def xgb_cv_grid_search_tuning(X_train, y_train, X_test, y_test, feature_names):
    print("Running xgb.cv + grid search tuning...")

    from sklearn.model_selection import ParameterGrid

    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 3]
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    best_rmse = float('inf')
    best_params = None
    best_n_estimators = None

    for params in ParameterGrid(param_grid):
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=3,
            early_stopping_rounds=50,
            metrics='rmse',
            seed=42,
            verbose_eval=False
        )
        mean_rmse = cv_results['test-rmse-mean'].min()
        n_estimators = cv_results['test-rmse-mean'].idxmin()
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params
            best_n_estimators = n_estimators

    print(f"Best params from xgb.cv grid search: {best_params}")
    print(f"Best n_estimators: {best_n_estimators}")

    model = xgb.XGBRegressor(
        n_estimators=best_n_estimators,
        random_state=42,
        **best_params
    )
    model.fit(X_train, y_train)

    r2, rmse, mae, mape = evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, feature_names, 'xgb_cv_grid_search')

    return {'Method': 'XGB CV Grid Search', 'n_estimators': best_n_estimators, **best_params,
            'R2 Score': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape}

def optuna_tuning(X_train, y_train, X_test, y_test, feature_names):
    print("Running Optuna tuning...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.2),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'objective': 'reg:squarederror',
            'random_state': 42,
            'verbosity': 0
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    best_params = study.best_params
    model = xgb.XGBRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    r2, rmse, mae, mape = evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, feature_names, 'optuna')

    return {'Method': 'Optuna', **best_params, 'R2 Score': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape}

if __name__ == "__main__":
    feature_names = X_train.columns.tolist()

    results = []
    results.append(random_search_tuning(X_train, y_train, X_test, y_test, feature_names))
    results.append(bayesian_optimization_tuning(X_train, y_train, X_test, y_test, feature_names))
    results.append(xgb_cv_grid_search_tuning(X_train, y_train, X_test, y_test, feature_names))
    results.append(optuna_tuning(X_train, y_train, X_test, y_test, feature_names))

    df_results = pd.DataFrame(results)
    df_results.to_csv(f'{output_dir}/xgboost_all_automatic_tuning_results.csv', index=False)
    print(f"All tuning results saved to {output_dir}/xgboost_all_automatic_tuning_results.csv")
