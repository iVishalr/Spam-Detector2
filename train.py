import pandas as pd
import numpy as np
import pickle
from preprocess import pre_process
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,roc_auc_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def load_dataset():
  df = pd.read_csv("./spamham.csv",delimiter='\t')
  df = df.reset_index()
  df = df.rename(columns={'index':'target'})
  # print("percentage of spam messages in dataset : ",100*len(df[df['target']=='spam'])/len(df))
  valid_df = df[df['target']=='ham']
  spam_df = df[df['target']=='spam']
  new_valid_df = valid_df[:len(spam_df)+2000]
  # print(spam_df,new_valid_df)
  DF = pd.DataFrame({'Messages': spam_df['Messages'].tolist() + new_valid_df['Messages'].tolist(),'target':spam_df['target'].tolist() + new_valid_df['target'].tolist()})
  DF = DF.sample(frac=1).reset_index(drop=True)
  # print("percentage of spam messages in new dataset : ",100*len(DF[DF['target']=='spam'])/len(DF))
  return DF

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

def train_clf():
  df = load_dataset()

  X = df['Messages']
  y = df['target']

  X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
  X_train = pd.Series(pre_process(X_train))
  y_train = np.where(y_train=='ham',0,1)
  vect = TfidfVectorizer(min_df=5,ngram_range=(2,5)).fit(X_train)
  with open("./vect.pkl",'wb') as f:
    pickle.dump(vect,f)
    
  X_train_vectorized = vect.transform(X_train)

  X_test = pd.Series(pre_process(X_test))
  y_test = np.where(y_test=='ham',0,1)
  X_test_vectorized = vect.transform(X_test)

  X_train_len = X_train.apply(len)
  X_train_digits = X_train.str.count(r'\d')
  X_train_non = X_train.str.count(r'\W')

  X_test_len = X_test.apply(len)
  X_test_digits = X_test.str.count(r'\d')
  X_test_non = X_test.str.count(r'\W')
  
  X_train_1 = add_feature(X_train_vectorized,X_train_len)
  X_train_2 = add_feature(X_train_1,X_train_digits)
  X_train_vect = add_feature(X_train_2,X_train_non)
  
  X_test_1 = add_feature(X_test_vectorized,X_test_len)
  X_test_2 = add_feature(X_test_1,X_test_digits)
  X_test_vect = add_feature(X_test_2,X_test_non)

  # clf = SVC(C=100,kernel="linear").fit(X_train_vect,y_train)
  # clf = LogisticRegression(C=10000,solver='lbfgs').fit(X_train_vect,y_train)
  clf = MLPClassifier(solver='adam',activation='relu',hidden_layer_sizes=(600,150),learning_rate_init=3e-4,batch_size=256,random_state=0).fit(X_train_vect,y_train)
  y_predictions = clf.predict(X_test_vect)
  train_accuracy = clf.score(X_train_vect,y_train)
  test_accuracy = clf.score(X_test_vect,y_test)
  precision = precision_score(y_test,y_predictions)
  recall = recall_score(y_test,y_predictions)
  auc = roc_auc_score(y_test,y_predictions)
  confusion_Matrix = confusion_matrix(y_test,y_predictions)

  print("Train : ",train_accuracy)
  print("Test : ",test_accuracy)
  print("Precision : ",precision)
  print("Recall : ",recall)
  print("AUC : ",auc)
  print("Confusion Matrix : \n",confusion_Matrix)

  with open('./model.pkl', 'wb') as f:
    pickle.dump(clf, f)   

if __name__ == "__main__":
    train_clf()