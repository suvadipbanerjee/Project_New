import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../data/raw/diabetes.csv')
df.columns = df.columns.str.strip()

df.drop('id', axis=1, inplace=True)
categorical_cols = ['location','gender','frame']
numeric_cols = [col for col in df.columns if col not in categorical_cols + ['time.ppn']]

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop('time.ppn', axis=1)
y = df['time.ppn']  

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X.to_csv('../data/processed/diabetes_features.csv', index=False)
y.to_csv('../data/processed/diabetes_target.csv', index=False)

print("Diabetes dataset preprocessing complete. Features and target saved.")
