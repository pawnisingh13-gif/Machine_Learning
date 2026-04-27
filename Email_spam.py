# Email Spam Detection - Single File Project

import pandas as pd

# ML tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
# Dataset format: label, message
# ham = 0, spam = 1

df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['label', 'message']]
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# -----------------------------
# 2. Split Data
# -----------------------------
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Text Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 4. Train Model
# -----------------------------
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# -----------------------------
# 5. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# -----------------------------
# 6. Custom Prediction Function
# -----------------------------
def predict_spam(message):
    msg_vec = vectorizer.transform([message])
    result = model.predict(msg_vec)

    return "Spam" if result[0] == 1 else "Not Spam"

# -----------------------------
# 7. Test Examples
# -----------------------------
print(predict_spam("Congratulations! You won a free iPhone"))
print(predict_spam("Let's meet tomorrow at college"))
