import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Load CSV
data = pd.read_csv("study_data_logistic.csv")
X = data[['Hours']]
y = data['Passed']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Plot data points
plt.scatter(X, y, color='blue', label='Data')

# Plot decision boundary
x_values = np.linspace(0, 7, 300).reshape(-1, 1)
y_probs = model.predict_proba(x_values)[:, 1]
plt.plot(x_values, y_probs, color='red', label='Logistic Curve')

# Labels
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression Curve')
plt.legend()
plt.grid(True)
plt.show()
