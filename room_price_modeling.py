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

# Create output directory
output_dir = 'modeling_outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up logging
log_filename = os.path.join(output_dir, f'room_price_modeling_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

# 1. Read data
logging.info("Reading data...")
df = pd.read_csv('cleaned_with_count.csv')
logging.info(f"Data shape: {df.shape}")

# 2. Feature Engineering
logging.info("\nPerforming feature engineering...")

# 2.1 Process categorical features
categorical_features = ['Room Type', 'State', 'Location Detail']
le = LabelEncoder()
for feature in categorical_features:
    df[feature + '_encoded'] = le.fit_transform(df[feature].astype(str))

# 2.2 Facility-related features (binary features)
facility_features = [
    'Air-Conditioning', 'Washing Machine', 'Wifi / Internet Access',
    'Cooking Allowed', 'TV', 'Share Bathroom', 'Private Bathroom',
    'Near KTM / LRT', 'Near LRT / MRT', 'Near KTM', 'Near LRT', 'Near MRT',
    'Near Train', 'Near Bus stop', '24 hours security', 'Swimming Pools',
    'Gymnasium Facility', 'OKU Friendly', 'Multi-purpose hall', 'Playground',
    'Covered car park', 'Surau', 'Mini Market', 'Co-Living'
]

# 2.3 Select final features
final_features = ['Price', 'Latitude', 'Longitude'] + \
                 [f + '_encoded' for f in categorical_features] + \
                 facility_features

# 2.4 Prepare modeling data
X = df[final_features].drop('Price', axis=1)
y = df['Price']

# 2.5 Feature standardization (required for some models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 2.6 Feature selection - using F-regression to select most important features
logging.info("\nPerforming feature selection using F-regression...")

# Use F-regression feature selection method
selector = SelectKBest(f_regression, k=5)
selector.fit(X, y)

# Get feature scores and p-values
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'F-Score': selector.scores_,
    'P-value': selector.pvalues_
})
feature_scores = feature_scores.sort_values('F-Score', ascending=False)

logging.info("\nTop 5 features selected by F-regression:")
logging.info("\n" + str(feature_scores.head(5)))

# Get selected features
selected_features = feature_scores.head(5)['Feature'].values
logging.info(f"Selected features: {selected_features}")

# Create dataset with selected features
X_selected = X[selected_features]
X_selected_scaled = X_scaled[selected_features]

# Visualize F-scores of selected features
plt.figure(figsize=(12, 6))
sns.barplot(x='F-Score', y='Feature', data=feature_scores.head(10))
plt.title('Top 10 Features by F-Regression')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_selection.png'))

# 3. Data splitting
logging.info("\nSplitting data...")
# Split data with all features
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Split data with selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Save preprocessed data for model tuning
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
with open(os.path.join(output_dir, 'preprocessed_data.pkl'), 'wb') as f:
    pickle.dump(preprocessed_data, f)

# 4. Model training and evaluation
def evaluate_model(y_true, y_pred, model_name):
    # Regression metrics
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

# Define all models
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

# Add data visualization before model training
def create_visualizations(df, y_test, best_pred):
    logging.info("\nCreating visualizations...")

    # 1. Price distribution plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='Price', bins=50)
    plt.title('Price Distribution')

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['Price'])
    plt.title('Price Box Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'price_distribution.png'))

    # 2. Correlation heatmap
    plt.figure(figsize=(15, 12))
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))

    # 3. Price distribution by state
    plt.figure(figsize=(15, 6))
    sns.boxplot(x='State', y='Price', data=df)
    plt.xticks(rotation=45)
    plt.title('Price Distribution by State')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'price_by_state.png'))

    # 4. Price comparison by room type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Room Type', y='Price', data=df)
    plt.xticks(rotation=45)
    plt.title('Price Distribution by Room Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'price_by_room_type.png'))

    # 5. Relationship between facilities and price
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
    plt.savefig(os.path.join(output_dir, 'price_by_facilities.png'))

    # 6. Actual vs Predicted prices scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, best_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))

    # 7. Prediction error distribution
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
    plt.savefig(os.path.join(output_dir, 'prediction_errors.png'))

    # 8. Geographic distribution of prices
    plt.figure(figsize=(12, 8))
    plt.scatter(df['Longitude'], df['Latitude'],
                c=df['Price'], cmap='viridis',
                alpha=0.6)
    plt.colorbar(label='Price')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographic Distribution of Prices')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'geographic_prices.png'))

# Train and evaluate all models
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

    # Update best model
    if metrics['r2'] > best_r2:
        best_r2 = metrics['r2']
        best_model = model
        best_model_name = name
        best_pred = y_pred  # Save best predictions

# Create model comparison DataFrame
models_comparison = pd.DataFrame({
    'Model': list(model_metrics.keys()),
    'R2 Score': [metrics['r2'] for metrics in model_metrics.values()],
    'RMSE': [metrics['rmse'] for metrics in model_metrics.values()],
    'MAE': [metrics['mae'] for metrics in model_metrics.values()],
    'MAPE': [metrics['mape'] for metrics in model_metrics.values()]
})

# Sort by R2 score
models_comparison = models_comparison.sort_values('R2 Score', ascending=False)
logging.info("\nModel Comparison:")
logging.info("\n" + str(models_comparison))
models_comparison.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

# Visualize model comparison
plt.figure(figsize=(15, 8))
sns.barplot(x='Model', y='R2 Score', data=models_comparison)
plt.xticks(rotation=45, ha='right')
plt.title('Model Comparison - R2 Scores')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_comparison.png'))

# If best model is Random Forest, show feature importance
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
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))

# Add visualization call before saving best model
create_visualizations(df, y_test, best_pred)

# Save best model
logging.info(f"\nSaving the best model ({best_model_name})...")
with open(os.path.join(output_dir, 'best_model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)

# Get top 3 performing models
top_models = models_comparison.head(3)['Model'].values
logging.info(f"\nTop 3 models for tuning: {top_models}")
logging.info("To tune these models, please run model_tuning.py")

logging.info("\nModeling complete! Results have been saved.")
logging.info(f"Detailed log saved to {log_filename}")
logging.info(f"All outputs saved in directory: {output_dir}")
