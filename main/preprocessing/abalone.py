import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv('../data/raw/abalonedata.csv')  
categorical_cols = ['Sex']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  

numeric_cols = ['Length','Diameter','Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight']

df[numeric_cols] = df[numeric_cols].replace(0, pd.NA)
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

X = df.drop(['Rings', 'id'] if 'id' in df.columns else ['Rings'], axis=1)  
y = df['Rings']

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X.to_csv('../data/processed/abalone_features.csv', index=False)
y.to_csv('../data/processed/abalone_target.csv', index=False)

print("Preprocessing completed. Features and target saved.")
