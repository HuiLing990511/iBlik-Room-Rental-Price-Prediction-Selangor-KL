import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Create models_result folder if it doesn't exist
if not os.path.exists('deployment'):
    os.makedirs('deployment')

# 1. Read data
df = pd.read_csv('cleaned_with_count.csv')
print(f"Data shape: {df.shape}")

# 2. Split data (same way as training)
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Engineering
categorical_features = ['Room Type', 'State', 'Location Detail']
encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    all_data = pd.concat([X_train[feature], X_test[feature]], axis=0)
    le.fit(all_data.astype(str))
    
    # Apply encoding (needed for scaler)
    X_train[feature + '_encoded'] = le.transform(X_train[feature].astype(str))
    X_test[feature + '_encoded'] = le.transform(X_test[feature].astype(str))
    
    # Save encoder
    encoders[feature] = le

# 4. Prepare final features (same as training)
facility_features = [
    'Air-Conditioning', 'Washing Machine', 'Wifi / Internet Access',
    'Cooking Allowed', 'TV', 'Share Bathroom', 'Private Bathroom',
    'Near KTM / LRT', 'Near LRT / MRT', 'Near KTM', 'Near LRT', 'Near MRT',
    'Near Train', 'Near Bus stop', '24 hours security', 'Swimming Pools',
    'Gymnasium Facility', 'OKU Friendly', 'Multi-purpose hall', 'Playground',
    'Covered car park', 'Surau', 'Mini Market', 'Co-Living'
    ]

final_features = ['Latitude', 'Longitude'] + \
                 [f + '_encoded' for f in categorical_features] + \
                 facility_features

X_train_final = X_train[final_features]
X_test_final = X_test[final_features]

# 5. Create and fit scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_final.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_final.columns)

deployment_assets = {
    'encoders': encoders,
    'scaler': scaler,
    'final_features': final_features,
    'categorical_features': categorical_features,
    'facility_features': facility_features
}

# Save main deployment file
with open('deployment/deployment_assets.pkl', 'wb') as f:
    pickle.dump(deployment_assets, f)

# Save individual components for convenience
pickle.dump(encoders, open('deployment/label_encoders.pkl', 'wb'))
pickle.dump(scaler, open('deployment/scaler.pkl', 'wb'))

print("Successfully saved:")
print("✅ deployment_assets.pkl")
print("✅ label_encoders.pkl") 
print("✅ scaler.pkl")