# Import Dependencies 
import pandas as pd
from flask import Flask, request, jsonify, render_template, make_response
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords


app = Flask(__name__)

# load the vectorizer
loaded_vectorizer = pickle.load(open('Data/vectorizer.pickle', 'rb'))
 # load the model
loaded_model = pickle.load(open('Data/classification.model', 'rb'))

# Preprocess Tweets
def preprocess_tweet(tweet):
    tweet.lower()
    # Remove urls
    tweet = re.sub(r'http\S+', "", tweet)
    # Remove tweet mentions
    tweet = re.sub(r'@\w+', '', tweet)
    # Remove punctuations
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # Remove numbers
    tweet = re.sub(r'[0-9]+', '', tweet)
   # Remove stopwords
    STOPWORDS = set(stopwords.words('english'))
    filtered_words = [word for word in str(tweet).split() if word not in STOPWORDS]
    return " ".join(filtered_words)

def predict_tweet(tweet):
    tweet_vec= loaded_vectorizer.transform([tweet])
    
    # make prediction
    tweet_prediction = loaded_model.predict(tweet_vec.toarray())
    return tweet_prediction


@app.route('/')
def models():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_sentiment():

    if request.method == 'POST': 
    # Get user input from HTML form
     input_tweet = request.form['input_tweet']
     print(preprocess_tweet(input_tweet))
     prediction = predict_tweet(input_tweet)
     sentiment = 'Positive Sentiment' if prediction == 0 else 'Negative Sentiment'
     return render_template('index.html', predict_text= sentiment)
 
if __name__ == "__main__":
    app.run(debug=True)

