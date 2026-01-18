import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../data/raw/credit.csv')
df.columns = df.columns.str.strip()

categorical_cols = ['checking_balance','credit_history','purpose','savings_balance',
                    'employment_length','personal_status','other_debtors','property',
                    'installment_plan','housing','job','telephone','foreign_worker']
numeric_cols = ['months_loan_duration','amount','installment_rate','residence_history',
                'age','existing_credits','dependents']

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df['default'] = df['default'].apply(lambda x: 1 if str(x).strip().lower() == '1' else 0)

X = df.drop('default', axis=1)
y = df['default']

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
X.to_csv('../data/processed/credit_features.csv', index=False)
y.to_csv('../data/processed/credit_target.csv', index=False)

print("Credit dataset preprocessing complete. Features and target saved.")
