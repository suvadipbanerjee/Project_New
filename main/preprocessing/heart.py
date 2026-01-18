import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../data/raw/heart.csv')
df.columns = df.columns.str.strip()

numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X.to_csv('../data/processed/heart_features.csv', index=False)
y.to_csv('../data/processed/heart_target.csv', index=False)

print("Heart dataset preprocessing complete. Features and target saved.")
