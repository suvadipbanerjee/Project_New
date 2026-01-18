import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv('../data/raw/adult.csv')

df.replace('?', pd.NA, inplace=True)

categorical_cols = ['workclass','education','marital.status','occupation',
                    'relationship','race','sex','native.country']
numeric_cols = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')  
    df[col].fillna(df[col].median(), inplace=True)


df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  
df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

X = df.drop('income', axis=1)
y = df['income']

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X.to_csv('../data/processed/adult_features.csv', index=False)
y.to_csv('../data/processed/adult_target.csv', index=False)

print("Preprocessing completed. Features and target saved.")
