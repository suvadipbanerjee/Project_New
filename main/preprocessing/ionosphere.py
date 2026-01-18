import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('../data/raw/ionosphere_data.csv')
df.columns = df.columns.str.strip()
bool_cols = ['column_a', 'column_b']

for col in bool_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({'true': 1, 'false': 0})
    )

df[bool_cols] = df[bool_cols].fillna(df[bool_cols].mode().iloc[0])
df['column_ai'] = df['column_ai'].str.strip().replace({'g': 1, 'b': 0})

numeric_cols = [col for col in df.columns if col not in bool_cols + ['column_ai']]
df[numeric_cols] = df[numeric_cols].astype(float)

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

assert df.isnull().sum().sum() == 0, "NaNs still present in dataframe!"

X = df.drop('column_ai', axis=1)
y = df['column_ai']

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X.to_csv('../data/processed/ionosphere_features.csv', index=False)
y.to_csv('../data/processed/ionosphere_target.csv', index=False)

print("Ionosphere preprocessing completed successfully with zero NaNs.")
