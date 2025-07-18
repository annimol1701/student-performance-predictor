
# student.vegeta.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Sample student data
data = {
    'hours_studied': [2, 3, 4, 5, 6, 7, 8],
    'attendance': [60, 65, 70, 75, 80, 85, 90],
    'passed': [0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['hours_studied', 'attendance']]
y = df['passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
sample = [[6, 80]]  # 6 hours studied, 80% attendance
prediction = model.predict(sample)
print("Pass" if prediction[0] == 1 else "Fail")
