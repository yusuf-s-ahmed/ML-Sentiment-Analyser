from fastapi import FastAPI
from pydantic import BaseModel
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import uvicorn # Import uvicorn

# Ensure NLTK's VADER lexicon is downloaded
nltk.download("vader_lexicon")

# Initialize FastAPI app
app = FastAPI()

# Initialize sentiment analyser
analyser = SIA()

# Request model for receiving text
class TextInput(BaseModel):
    text: str

# Function to classify sentiment
def classify_sentiment(text: str):
    sentiment_score = analyser.polarity_scores(text)["compound"]

    if sentiment_score <= -0.6:
        return "Extremely Negative", 1
    elif -0.6 < sentiment_score <= -0.2:
        return "Moderately Negative", 2
    elif -0.2 < sentiment_score <= 0.2:
        return "Neutral", 3
    elif 0.2 < sentiment_score <= 0.5:
        return "Moderately Positive", 4
    else:
        return "Extremely Positive", 5

# Additional rule for specific negative phrases
def detect_specific_negatives(text: str):
    negative_phrases = ["could have been better", "not great", "disappointed", "needs improvement"]
    if any(phrase in text.lower() for phrase in negative_phrases):
        return "Moderately Negative", 2
    return None

# API endpoint for sentiment analysis
@app.post("/analyse/")
async def analyse_sentiment(input_text: TextInput):
    specific_neg_result = detect_specific_negatives(input_text.text)
    if specific_neg_result:
        sentiment, score = specific_neg_result
    else:
        sentiment, score = classify_sentiment(input_text.text)

    return {"sentiment": sentiment, "score (1-5):": score}

# Conditional execution:  Only run the server if the script is run directly
if __name__ == "__main__":
    # Option 1: Run with uvicorn directly (for development)
    uvicorn.run(app="app:app", host="127.0.0.1", port=8000, reload=True) 