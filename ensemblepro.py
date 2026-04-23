# 1. Import Libraries
import pandas as pd
from google.colab import files
uploaded = files.upload()
df=pd.read_csv("data.csv")
print(df)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. Load Dataset
df = pd.read_csv("data.csv")
print("Dataset:\n", df)

#  3. Split Features & Target
X = df[['age', 'estimated_salary']]
y = df['purchased']

#  4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  RANDOM FOREST

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))


#  DECISION TREE

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, dt_pred))


#  LOGISTIC REGRESSION

lr = LogisticRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, lr_pred))

import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(rf, f)   # rf = your Random Forest model
