import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Load Data
df = pd.read_csv('churn_data.csv')
X = df.drop(columns=['Churn'])
y = df['Churn']

# 2. Define Features
categorical_cols = ['Connection_Type', 'Payment_Method']
numerical_cols = ['Tenure_Months', 'Monthly_Bill_LKR', 'Support_Tickets_30d']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# 3. Preprocess Features
X_transformed = preprocessor.fit_transform(X)
ohe_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
all_feature_names = ohe_feature_names + numerical_cols

X_df = pd.DataFrame(X_transformed, columns=all_feature_names)

# 4. Train XGBoost Model
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 5. Save Artifacts
joblib.dump(model, 'model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(all_feature_names, 'feature_names.pkl')

print("✅ Model trained & saved with Sri Lankan dataset!")