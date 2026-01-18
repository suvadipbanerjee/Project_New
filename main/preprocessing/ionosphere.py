import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../data/raw/ionosphere_data.csv')
df.columns = df.columns.str.strip()

bool_cols = ['column_a', 'column_b']

for col in bool_cols:
    df[col] = df[col].map({'true': 1, 'false': 0})


numeric_cols = [col for col in df.columns if col not in bool_cols + ['column_ai']]
df[numeric_cols] = df[numeric_cols].astype(float)

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

df['column_ai'] = df['column_ai'].replace({'g': 1, 'b': 0})

X = df.drop('column_ai', axis=1)
y = df['column_ai']

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X.to_csv('../data/processed/ionosphere_features.csv', index=False)
y.to_csv('../data/processed/ionosphere_target.csv', index=False)

print("Ionosphere dataset preprocessing complete. Features and target saved.")
