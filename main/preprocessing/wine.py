import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../data/raw/winequalityN.csv')
df.columns = df.columns.str.strip()
df['type'] = df['type'].map({'white': 0, 'red': 1})

numeric_cols = df.drop('quality', axis=1).columns.tolist()

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

X = df.drop('quality', axis=1)
y = df['quality']

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
X.to_csv('../data/processed/winequality_features.csv', index=False)
y.to_csv('../data/processed/winequality_target.csv', index=False)

print("Wine Quality dataset preprocessing complete. Features and target saved.")
