import streamlit as st
# Streamlit page configuration (must be called first and only once)
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import os
import emoji

# Set plot style
plt.style.use('ggplot')

# Define data and model paths
INPUT_FILE = "Reviews.csv"  # Ensure this file is in the same directory
OUTPUT_FILE = "Sentiment_Analysis_Results.csv"

# Function to check if results already exist
def load_or_process_data(input_file, output_file):
    if os.path.exists(output_file):
        return pd.read_csv(output_file)

    # If results are not available, process the data
    print("Processing data...")
    df = pd.read_csv(input_file)
    
    # Load pre-trained model and tokenizer
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    # Define RoBERTa polarity scoring function
    def polarity_scores_roberta(text):
        encoded_text = tokenizer(text, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores_dict = {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        }
        return scores_dict

    # Process reviews and compile results
    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            text = row['Text']
            myid = row['Id']

            # VADER sentiment analysis
            from nltk.sentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            vader_result = sia.polarity_scores(text)
            vader_result_rename = {f"vader_{key}": value for key, value in vader_result.items()}

            # RoBERTa sentiment analysis
            roberta_result = polarity_scores_roberta(text)

            # Combine results
            both = {**vader_result_rename, **roberta_result}
            res[myid] = both
        except RuntimeError as e:
            print(f"Error for id {myid}: {e}")

    # Create a results DataFrame
    results_df = pd.DataFrame(res).T
    results_df = results_df.reset_index().rename(columns={"index": "Id"})
    results_df = results_df.merge(df, how="left")

    # Save results to file
    results_df.to_csv(output_file, index=False)
    print(f"Sentiment analysis completed. Results saved to {output_file}.")
    return results_df

# Load or process data
results_df = load_or_process_data(INPUT_FILE, OUTPUT_FILE)

# Add sentiment labels with emojis
def get_sentiment_emoji(row):
    if row["roberta_pos"] > max(row["roberta_neg"], row["roberta_neu"]):
        return "😊 Positive"
    elif row["roberta_neg"] > max(row["roberta_neu"], row["roberta_pos"]):
        return "😡 Negative"
    else:
        return "😐 Neutral"

if "Sentiment" not in results_df.columns:
    results_df["Sentiment"] = results_df.apply(get_sentiment_emoji, axis=1)

# Streamlit Dashboard
# Title and description
st.title("Sentiment Analysis Dashboard")
st.markdown(
    """
    This app shows sentiment analysis results from customer reviews using the **RoBERTa** model.  
    Each review is categorized as **Positive**, **Neutral**, or **Negative**, with scores and an emoji-based summary.
    """
)

# Sidebar filters
st.sidebar.header("Filters")
score_filter = st.sidebar.multiselect(
    "Filter by Review Stars",
    options=results_df["Score"].unique(),
    default=results_df["Score"].unique()
)

sentiment_filter = st.sidebar.multiselect(
    "Filter by Sentiment",
    options=results_df["Sentiment"].unique(),
    default=results_df["Sentiment"].unique()
)

# Apply filters to data
filtered_df = results_df[
    (results_df["Score"].isin(score_filter)) &
    (results_df["Sentiment"].isin(sentiment_filter))
]

# Display filtered data
st.subheader("Filtered Reviews")
st.write(f"Showing {len(filtered_df)} reviews based on your filters.")
st.dataframe(filtered_df[["Id", "Text", "Score", "roberta_pos", "roberta_neu", "roberta_neg", "Sentiment"]])

# Sentiment distribution bar chart
st.subheader("Sentiment Distribution")
sentiment_counts = filtered_df["Sentiment"].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis", ax=ax)
ax.set_title("Sentiment Distribution")
ax.set_ylabel("Count")
st.pyplot(fig)

# Review details
st.subheader("Detailed Review Sentiment")
selected_review = st.selectbox(
    "Select a Review to Analyze",
    options=filtered_df["Text"]
)

selected_row = filtered_df[filtered_df["Text"] == selected_review].iloc[0]
st.write(f"**Review Text:** {selected_review}")
st.write(f"**Sentiment:** {selected_row['Sentiment']}")
st.write(
    f"""
    - **Scores:**  
      - Positive: {selected_row['roberta_pos']:.2f}  
      - Neutral: {selected_row['roberta_neu']:.2f}  
      - Negative: {selected_row['roberta_neg']:.2f}
    """
)
st.markdown(
    f"### Emoji Representation: {emoji.emojize(':thumbs_up:') if 'Positive' in selected_row['Sentiment'] else emoji.emojize(':neutral_face:') if 'Neutral' in selected_row['Sentiment'] else emoji.emojize(':thumbs_down:')}"
)

# Add a download button for the filtered data
@st.cache_data
def convert_df(filtered_df):
    return filtered_df.to_csv(index=False)


csv = convert_df(filtered_df)

st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name='filtered_data.csv',
    mime='text/csv',
)
