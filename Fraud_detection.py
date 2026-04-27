# Fraud Detection

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Dataset
data = {
    'amount':[100,2000,150,3000,500,7000,120],
    'time':[10,200,15,300,20,400,12],
    'fraud':[0,1,0,1,0,1,0]
}

df = pd.DataFrame(data)

X = df[['amount','time']]
y = df['fraud']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

model = RandomForestClassifier()
model.fit(X_train,y_train)

print(classification_report(y_test, model.predict(X_test)))

# Prediction
print(model.predict([[2500,250]]))
