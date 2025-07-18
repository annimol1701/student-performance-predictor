import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Sample dataset
data = {
    'hours_studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'pass_exam':     [0, 0, 0, 1, 1, 1, 1, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Split into features and label
X = df[['hours_studied']]
y = df['pass_exam']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict for a student who studied 4.5 hours
hours = [[4.5]]
prediction = model.predict(hours)
print("Pass" if prediction[0] else "Fail")
