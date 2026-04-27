# Linear Regression Example - Predict Salary based on Experience

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([30000,40000,50000,60000,70000])

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
pred = model.predict([[6]])
print("Predicted Salary for 6 years experience:", pred)

# Visualization
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Linear Regression")
plt.show()
