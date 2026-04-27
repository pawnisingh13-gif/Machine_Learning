# Weather Prediction ML Project (Single File)

import pandas as pd

# ML tools
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Sample Dataset (Create manually)
# Outlook: Sunny=0, Overcast=1, Rain=2
# Temp: Hot=0, Mild=1, Cool=2
# Humidity: High=0, Normal=1
# Wind: Weak=0, Strong=1
# PlayWeather: No=0, Yes=1
# -----------------------------

data = {
    'outlook': [0,0,1,2,2,2,1,0,0,2,0,1,1,2],
    'temp':    [0,0,0,1,2,2,2,1,2,1,1,1,0,1],
    'humidity':[0,0,0,0,1,1,1,0,1,1,1,0,1,0],
    'wind':    [0,1,0,0,0,1,1,0,0,0,1,1,0,1],
    'play':    [0,0,1,1,1,0,1,0,1,1,1,1,1,0]
}

df = pd.DataFrame(data)

# -----------------------------
# 2. Split Data
# -----------------------------
X = df.drop('play', axis=1)
y = df['play']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Train Model
# -----------------------------
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# -----------------------------
# 4. Accuracy
# -----------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# 5. Prediction Function
# -----------------------------
def predict_weather(outlook, temp, humidity, wind):
    # input must be numeric as defined above
    prediction = model.predict([[outlook, temp, humidity, wind]])
    return "Good Weather" if prediction[0] == 1 else "Bad Weather"

# -----------------------------
# 6. Test Example
# -----------------------------
# Example: Sunny, Hot, High Humidity, Weak Wind
print(predict_weather(0, 0, 0, 0))
