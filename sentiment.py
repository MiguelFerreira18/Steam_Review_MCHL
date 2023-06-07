import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download('vader_lexicon')

# Example text
text = "One of the best games I've ever played, simple, yet with a lot of depth and progress you can make to experience each run differently. Tons of replayability (look at my play time). The game could use a big optimization boost in the late game, though (without spoiling), when you start farming. I played the game once on my Steam deck, but my battery was almost empty after an hour of playing, so I only play it on the computer now. The only thing I would like to add is that I would love to have drop cards, save your current runs, and resume them later."

# Create a SentimentIntensityAnalyzer object
sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis
sentiment_scores = sid.polarity_scores(text)

# Print the sentiment scores
print("Sentiment Scores:", sentiment_scores)

# Determine the overall sentiment
if sentiment_scores['compound'] >= 0.05:
    sentiment = "Positive"
elif sentiment_scores['compound'] <= -0.05:
    sentiment = "Negative"
else:
    sentiment = "Neutral"

# Print the overall sentiment
print("Overall Sentiment:", sentiment)
