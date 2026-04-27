import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
 'distance':[5,10,15],
 'time':[15,30,45]
})

model = LinearRegression().fit(df[['distance']], df['time'])
print(model.predict([[12]]))
