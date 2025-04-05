import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle

df = pd.read_csv("data/bagpack/train.csv")

X = df.drop(columns=['Price'])
y = df['Price']

# Feature Engineering:
X_dummy = pd.get_dummies(X)

imputer = SimpleImputer(strategy='mean')
X_dummy_imputed = imputer.fit_transform(X_dummy)

X_train, X_test, y_train, y_test = train_test_split(X_dummy_imputed, y, test_size=0.1, shuffle=True)

X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)

X_train_df['Price'] = y_train.values
X_test_df['Price'] = y_test.values

X_train_df.to_csv("data/processed_train_data.csv", index=False)
X_test_df.to_csv("data/processed_test_data.csv", index=False)

with open("data/preprocessor.pkl", 'wb') as f:
    pickle.dump(imputer, f)

print("Preprocessing complete! Processed data saved.")