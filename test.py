from preprocess import pre_process
from train import add_feature
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
message = sys.argv[1]

message = message.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress')
# Remove punctuation
message = message.replace(r'[^\w\d\s]', ' ')
# Replace whitespace between terms with a single space
message = message.replace(r'\s+', ' ')
# Remove leading and trailing whitespace
message = message.replace(r'^\s+|\s+?$', '')
message = message.lower()
# remove stopwords from messages
stop_words = set(stopwords.words('english'))
for words in message.split(" "):
  temp = []
  if words not in stop_words:
    temp.append(lemmatizer.lemmatize(words))
  message = " ".join(temp)

df = pd.Series(message)

with open("./vect.pkl","rb") as f:
  vect = pickle.load(f)

message_transformed = vect.transform(df)

message_len = df.apply(len)
message_digits = df.str.count(r'\d')
message_non = df.str.count(r'\W')

message1 = add_feature(message_transformed,message_len)
message2 = add_feature(message1,message_digits)
message_vect = add_feature(message2,message_non)

with open('./model.pkl', 'rb') as f:
    clf = pickle.load(f)

prediction = clf.predict(message_vect)
print("Prediction : ",prediction)
if(prediction==0):
  print("Message is not a spam!")

else:
  print("Message is a SPAM!")