import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.DataFrame({
    'glucose':[80,120,140,90,160],
    'bp':[70,80,90,75,95],
    'diabetes':[0,1,1,0,1]
})

X = df[['glucose','bp']]
y = df['diabetes']

model = LogisticRegression().fit(X,y)

print(model.predict([[130,85]]))
