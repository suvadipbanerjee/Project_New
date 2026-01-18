import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../data/raw/sonar.csv')

df.columns = df.columns.str.strip()

df['Class'] = df['Class'].map({'R': 0, 'M': 1})

numeric_cols = [col for col in df.columns if col != 'Class']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X.to_csv('../data/processed/sonar_features.csv', index=False)
y.to_csv('../data/processed/sonar_target.csv', index=False)

print("Sonar dataset preprocessing complete. Features and target saved.")
