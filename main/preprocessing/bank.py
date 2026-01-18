import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('../data/raw/bank_marketing.csv', sep=';')

df.replace('unknown', pd.NA, inplace=True)

categorical_cols = ['job','marital','education','default','housing','loan',
                    'contact','month','day_of_week','poutcome']
numeric_cols = ['age','duration','campaign','pdays','previous',
                'emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')  
    df[col].fillna(df[col].median(), inplace=True)

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

X = df.drop('y', axis=1)
y = df['y']

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X.to_csv('../data/processed/bank_marketing_features.csv', index=False)
y.to_csv('../data/processed/bank_marketing_target.csv', index=False)

print("Preprocessing completed. Features and target saved.")
