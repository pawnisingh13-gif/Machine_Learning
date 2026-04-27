# Stock Price Prediction

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    'day':[1,2,3,4,5],
    'price':[100,105,110,108,115]
})

X = df[['day']]
y = df['price']

model = LinearRegression()
model.fit(X,y)

print("Next Day Price:", model.predict([[6]]))
