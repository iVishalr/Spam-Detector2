import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Begin Pre Processing
def pre_process(df):
  #Replace different emailIDs with emailaddress 
  lemmatizer = WordNetLemmatizer()
  df = df.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress')
  # Replace URLs with 'webaddress'
  df = df.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')
  # Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
  df = df.str.replace(r'£|\$', 'moneysymb')
  # Replace 10 digit phone numbers (formats include parenthesis, spaces, no spaces, dashes) with 'phonenumber'
  df = df.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumber')
  # Replace numbers with 'number'
  df = df.str.replace(r'\d+(\.\d+)?', 'number')
  # Remove punctuation
  df = df.str.replace(r'[^\w\d\s]', ' ')
  # Replace whitespace between terms with a single space
  df = df.str.replace(r'\s+', ' ')
  # Remove leading and trailing whitespace
  df = df.str.replace(r'^\s+|\s+?$', '')
  df = df.str.lower()
  # remove stopwords from messages
  stop_words = set(stopwords.words('english'))
  messages_list = []
  for messages in df.tolist():
    message = messages.split(" ")
    temp = []
    for words in message:
      if words not in stop_words:
        temp.append(lemmatizer.lemmatize(words))
    message = " ".join(temp)
    messages_list.append(message)
  df = messages_list
  #tokenize the messages
  # df['target'] = np.where(df['target']=='ham',0,1)
  return df