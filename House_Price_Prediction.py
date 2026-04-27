# House Price Prediction using Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Area":[1000,1200,1500,1800,2000,2200],
    "Bedrooms":[2,2,3,3,4,4],
    "Price":[20,25,30,35,40,45]
}

df = pd.DataFrame(data)

# Features & Target
X = df[["Area","Bedrooms"]]
y = df["Price"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict([[1600,3]])
print("Predicted Price:", prediction)
