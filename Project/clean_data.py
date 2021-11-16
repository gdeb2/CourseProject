import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

from string import punctuation
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

import csv



def createCleanCSV():
    with open('cleanReviews.csv', 'w') as cleancsv:
        writer = csv.writer(cleancsv)
        writer.writerow(['category', 'rating', 'label', 'original_text', 'clean_text', 'word_count', 'character_count', 'capital_letters_count', 'digit_count', 'punctuation_count', 'sentiment_score'])


def appendCleanReview(data):
    with open('cleanReviews.csv', 'a') as cleancsv:
        writer = csv.writer(cleancsv)
        writer.writerow(data)

createCleanCSV()

def readCSV():
    with open('fake_reviews_dataset.csv', newline='') as csvfile:
         reader = csv.DictReader(csvfile)
         sent_analyzer = SentimentIntensityAnalyzer()
         stop_words = set(stopwords.words('english'))
         lemmatizer = WordNetLemmatizer()

         for row in reader:
            category = row['category']
            rating = row['rating']
            label = row['label']
            review = row['text_']

            # count length of review
            character_count = len(review)

            # count occurences of digits
            digit_count = sum(char.isdigit() for char in review)

            # count occurences of capital letters
            letters = list(filter(str.isalpha, review)) 
            capital_letters_count = sum(map(str.isupper, letters))

            # tokenize
            tokenized_review = word_tokenize(review)

            # remove stop words from reviews
            review_no_stopwords = [word for word in tokenized_review if word.lower() not in stop_words]

            # remove punctuation and lemmatize reviews
            review_no_punctuation_and_lemmatized = [lemmatizer.lemmatize(word.lower()) for word in review_no_stopwords if word.lower() not in punctuation]
            clean_text = ' '.join(review_no_punctuation_and_lemmatized)

            # count occurences of punctuation
            punctuation_count = len([char for char in review if char in punctuation])

            # word count in sentence
            word_count = len(tokenized_review) - punctuation_count

            # calculate sentiment of review
            sentiment_score = sent_analyzer.polarity_scores("review")["compound"]
            
            row = [category,rating,label,review,clean_text,word_count,character_count,capital_letters_count,digit_count,punctuation_count,sentiment_score]
            appendCleanReview(row)


readCSV()


