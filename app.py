import streamlit as st
import pandas as pd
import re
from textblob import TextBlob
from collections import Counter
import matplotlib.pyplot as plt
from prophet import Prophet

# -------------------------------
# TITLE
# -------------------------------
st.title("📊 Social Media Trend Predictor for Local Businesses")

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("social_media_advanced_dataset_1000.csv")
df.columns = df.columns.str.strip().str.lower()
df['date'] = pd.to_datetime(df['date'])

# -------------------------------
# SENTIMENT SCORE ✅
# -------------------------------
df['sentiment_score'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# -------------------------------
# TREND FUNCTION (USING SCORE)
# -------------------------------
def predict_trend(text, score):
    text_lower = text.lower()

    positive_words = ["good","great","love","excellent","amazing","best","awesome","perfect","delicious"]
    negative_words = ["bad","worst","hate","poor","terrible","awful","disappointed","waste"]

    if any(word in text_lower for word in positive_words):
        return "Positive"
    if any(word in text_lower for word in negative_words):
        return "Negative"

    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

# -------------------------------
# ADD TREND COLUMN ✅
# -------------------------------
df['trend'] = df.apply(lambda x: predict_trend(x['text'], x['sentiment_score']), axis=1)

st.success("✅ Dataset Loaded Successfully")

# -------------------------------
# 📂 DISPLAY FULL DATASET
# -------------------------------
st.subheader("📂 Dataset with Sentiment Score & Trend")
st.dataframe(df)

# -------------------------------
# 🔍 SEARCH & FILTER
# -------------------------------
st.subheader("🔍 Search & Filter Data")

search_text = st.text_input("Search text")

location_filter = st.selectbox("Filter by Location", ["All"] + list(df['location'].unique()))
business_filter = st.selectbox("Filter by Business Type", ["All"] + list(df['business_type'].unique()))

filtered_df = df.copy()

if search_text:
    filtered_df = filtered_df[filtered_df['text'].str.contains(search_text, case=False)]

if location_filter != "All":
    filtered_df = filtered_df[filtered_df['location'] == location_filter]

if business_filter != "All":
    filtered_df = filtered_df[filtered_df['business_type'] == business_filter]

st.dataframe(filtered_df)

# -------------------------------
# 📊 TREND DISTRIBUTION
# -------------------------------
st.subheader("📊 Trend Distribution")

trend_counts = df['trend'].value_counts()
st.bar_chart(trend_counts)

# -------------------------------
# 📊 SENTIMENT SCORE STATS
# -------------------------------
st.subheader("📊 Sentiment Analysis")

st.write("Average Sentiment Score:", df['sentiment_score'].mean())

fig_score, ax_score = plt.subplots()
ax_score.hist(df['sentiment_score'], bins=20)
ax_score.set_title("Sentiment Score Distribution")

st.pyplot(fig_score)

# -------------------------------
# 🔥 TOP TRENDING WORDS
# -------------------------------
def process_text(text):
    return re.findall(r'\b[a-z]+\b', text.lower())

all_words = []
for text in df['text']:
    all_words.extend(process_text(text))

trend_words = Counter(all_words).most_common(10)

st.subheader("🔥 Top Trending Keywords")

for word, count in trend_words:
    st.write(f"{word} - {count}")

# -------------------------------
# 📈 TIME SERIES
# -------------------------------
st.subheader("📈 Daily Activity")

daily_posts = df.groupby(df['date'].dt.date).size()

fig, ax = plt.subplots()
ax.plot(daily_posts.index, daily_posts.values)
ax.set_xlabel("Date")
ax.set_ylabel("Posts Count")

st.pyplot(fig)

# -------------------------------
# 🔮 FORECASTING
# -------------------------------
st.subheader("🔮 Future Trend Forecast")

ts_data = daily_posts.reset_index()
ts_data.columns = ['ds', 'y']

model = Prophet()
model.fit(ts_data)

future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)

fig2 = model.plot(forecast)
st.pyplot(fig2)

# -------------------------------
# 🔮 PREDICT NEW TEXT
# -------------------------------
st.subheader("🔮 Predict Trend from New Post")

user_input = st.text_input("Enter a social media post")

if st.button("Predict"):
    if user_input:
        score = TextBlob(user_input).sentiment.polarity
        result = predict_trend(user_input, score)

        st.write("Sentiment Score:", score)

        if result == "Positive":
            st.success("🔥 Positive Trend")
        elif result == "Negative":
            st.error("❌ Negative Trend")
        else:
            st.info("⚖️ Neutral Trend")


# -------------------------------
# 💡 BUSINESS INSIGHTS
# -------------------------------
st.subheader("💡 Business Insights")

positive = trend_counts.get("Positive", 0)
negative = trend_counts.get("Negative", 0)

if positive > negative:
    st.success("📈 Customers are satisfied. Promote offers!")
else:
    st.warning("⚠️ Improve service quality!")

# -------------------------------
# 📄 AUTO REPORT
# -------------------------------
st.subheader("📄 Automated Report")

report = f"""
📊 Social Media Trend Report

Total Posts: {len(df)}
Positive: {positive}
Negative: {negative}

Average Sentiment Score: {df['sentiment_score'].mean():.2f}

Top Keywords:
"""

for word, count in trend_words:
    report += f"- {word} ({count})\n"

st.text(report)