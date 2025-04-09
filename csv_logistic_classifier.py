import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("study_data_logistic.csv")

# Split features and labels
X = data[["Hours"]]
y = data["Passed"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Predict on new examples
test_hours = [[2], [4], [5]]
predictions = model.predict(test_hours)
probabilities = model.predict_proba(test_hours)

# Output results
for i, hours in enumerate(test_hours):
    result = "Pass" if predictions[i] == 1 else "Fail"
    confidence = probabilities[i][predictions[i]] * 100
    print(f"Study hours: {hours[0]} → Predicted: {result} (Confidence: {confidence:.2f}%)")


