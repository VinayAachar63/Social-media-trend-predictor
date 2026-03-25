import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("Social Media Engagement Dataset.csv")
df.columns = df.columns.str.strip().str.lower()

# -------------------------------
# Detect Text Column
# -------------------------------
text_col = None
for col in df.columns:
    if "text" in col or "tweet" in col or "post" in col:
        text_col = col
        break

if text_col is None:
    text_col = df.columns[0]

df['text'] = df[text_col].astype(str)

# -------------------------------
# Create Sentiment Score
# -------------------------------
df['sentiment_score'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# -------------------------------
# CREATE TREND COLUMN (IMPORTANT)
# -------------------------------
def get_trend(score):
    if score >= 0:
        return "Positive"
    else:
        return "Negative"

df['trend'] = df['sentiment_score'].apply(get_trend)

print("Trend Distribution:")
print(df['trend'].value_counts())

# -------------------------------
# Encode Target
# -------------------------------
df['target'] = df['trend'].map({
    "Positive": 1,
    "Negative": 0
})

# -------------------------------
# TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X = vectorizer.fit_transform(df['text'])
y = df['target']

# -------------------------------
# Train Model
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# Save Files
# -------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained with trend column!")