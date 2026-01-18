import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../data/raw/Breast_cancer.csv')
df.columns = df.columns.str.strip()

categorical_cols = ['Race','Marital Status','T Stage','N Stage','6th Stage',
                    'differentiate','Grade','A Stage','Estrogen Status','Progesterone Status']
numeric_cols = ['Age','Tumor Size','Regional Node Examined','Reginol Node Positive','Survival Months']

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)


df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df['Status'] = df['Status'].apply(lambda x: 1 if x.strip().lower() == 'alive' else 0)

X = df.drop('Status', axis=1)
y = df['Status']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X.to_csv('../data/processed/breast_cancer_features.csv', index=False)
y.to_csv('../data/processed/breast_cancer_target.csv', index=False)

print("Preprocessing completed. Features and target saved.")
