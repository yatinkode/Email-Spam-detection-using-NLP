import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot

import re
import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import string

from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split

import time

#Load dataset
dataset = pd.read_csv("C:\\Users\\kode surendra aba\\Desktop\\Data science\\python\\Sample projects\\emails_spam_ham_Nlp\\emails.csv")

#explore data
list(dataset)
dataset.head()

dataset.shape #5728,2

dataset.drop_duplicates(inplace = True)

dataset.shape #(5695, 2)

print (pd.DataFrame(dataset.isnull().sum()))

#split text column into array of words


dataset['text'] = dataset['text'].map(lambda text:re.sub('\W+', ' ',text)).apply(lambda x: (x.lower()).split())

dataset['text']=dataset['text'].map(lambda text: text[1:])

#Remove stpwords

stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text

dataset['text'] = dataset['text'].apply(lambda x: remove_stopwords(x))

# Stemming
#def stemming(tokenized_text):
#    text = [ps.stem(word) for word in tokenized_text]
#    return text

#dataset['text'] = dataset['text'].apply(lambda x: stemming(x))

wn = nltk.WordNetLemmatizer()

def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

#dataset['text'] = dataset['text'].apply(lambda x: lemmatizing(x))

dataset.head(10)

#count vectorizer ######################
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer=lemmatizing)
X_counts = count_vect.fit_transform(dataset['text'])
print(X_counts.shape)
print(count_vect.get_feature_names())

X_counts_df = pd.DataFrame(X_counts.toarray())

X_counts_df.columns = count_vect.get_feature_names()

X_counts_df.head()
X_counts_df.shape

#Tfidf vectororizer #########################
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(analyzer=lemmatizing)
X_tfidf = tfidf_vect.fit_transform(dataset['text'])
print(X_tfidf.shape)
print(tfidf_vect.get_feature_names())

X_tfidf_df = pd.DataFrame(X_tfidf.toarray())

X_tfidf_df.columns = tfidf_vect.get_feature_names()

X_tfidf_df.head()
X_tfidf_df.shape


########### Create features #################

import string

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100
    
dataset['body_len'] = dataset['text'].apply(lambda x: len(x) - x.count(" "))
dataset['punct%'] = dataset['text'].apply(lambda x: count_punct(x))
dataset.head()

#plot the features

from matplotlib import pyplot

#plot for body length
bins = np.linspace(0, 200, 40)

pyplot.hist(dataset['body_len'], bins)
pyplot.title("Body Length Distribution")
pyplot.show()

bins = np.linspace(0, 200, 40)

pyplot.hist(dataset[dataset['spam']==1]['body_len'], bins, alpha=0.5, normed=True, label='spam')
pyplot.hist(dataset[dataset['spam']==0]['body_len'], bins, alpha=0.5, normed=True, label='ham')
pyplot.legend(loc='upper left')
pyplot.show()


#plot for punctuation count
bins = np.linspace(0, 50, 40)

pyplot.hist(dataset['punct%'], bins)
pyplot.title("Punctuation % Distribution")
pyplot.show()

bins = np.linspace(0, 200, 40)

pyplot.hist(dataset[dataset['spam']==1]['punct%'], bins, alpha=0.5, normed=True, label='spam')
pyplot.hist(dataset[dataset['spam']==0]['punct%'], bins, alpha=0.5, normed=True, label='ham')
pyplot.legend(loc='upper left')
pyplot.show()

#Body length cannot help us distinguish between spam and ham
#But ham messeges contain more punctuation than spam

### Random forest classifier ###########################

#concat a set of features which we will use as independent variables
#Python does not allow to cbind directly, first we need to ignore indexes to do a cbind or else it does an outer join


############### Using count vectorizer lets see the accuracy of the model ##################
#Taking independent variables separately
X_features = pd.concat([dataset['punct%'].reset_index(drop=True),dataset['body_len'].reset_index(drop=True),X_counts_df.reset_index(drop=True)], axis=1)

X_features.shape


#Divide data in train and test
X_train, X_test, y_train, y_test = train_test_split(X_features, dataset['spam'], test_size=0.2)

rf = RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1)
rf_model = rf.fit(X_train, y_train)

sorted(zip(rf_model.feature_importances_, X_train.columns), reverse=True)[0:10]

y_pred = rf_model.predict(X_test)
precision, recall, fscore, support = score(y_test, y_pred, pos_label=0, average='binary')

print('Precision: {} / Recall: {} / Accuracy: {}'.format(round(precision, 3),
                                                        round(recall, 3),
                                                        round((y_pred==y_test).sum() / len(y_pred),3)))


# Precision: 0.882 / Recall: 1.0 / Accuracy: 0.899


############### Using tfidf vectorizer lets see the accuracy of the model ##################
#Taking independent variables separately
X_features = pd.concat([dataset['punct%'].reset_index(drop=True),dataset['body_len'].reset_index(drop=True),X_tfidf_df.reset_index(drop=True)], axis=1)

X_features.shape


#Divide data in train and test
X_train, X_test, y_train, y_test = train_test_split(X_features, dataset['spam'], test_size=0.2)

rf = RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1)
rf_model = rf.fit(X_train, y_train)

sorted(zip(rf_model.feature_importances_, X_train.columns), reverse=True)[0:10]

y_pred = rf_model.predict(X_test)
precision, recall, fscore, support = score(y_test, y_pred, pos_label=0, average='binary')

print('Precision: {} / Recall: {} / Accuracy: {}'.format(round(precision, 3),
                                                        round(recall, 3),
                                                        round((y_pred==y_test).sum() / len(y_pred),3)))

# Precision: 0.901 / Recall: 1.0 / Accuracy: 0.915

#Here we can definetly seethat using tfidf vectorization we get a etter reslt so we choose tfidf vectorizer ahead

#############################################################################################################
#Checing best hyperparameter value to choose for better results
def train_RF(n_est, depth):
    rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, n_jobs=-1)
    rf_model = rf.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    precision, recall, fscore, support = score(y_test, y_pred, pos_label=1, average='binary')
    print('Est: {} / Depth: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(
        n_est, depth, round(precision, 3), round(recall, 3),
        round((y_pred==y_test).sum() / len(y_pred), 3)))
    
for n_est in [10, 50, 100]:
    for depth in [10, 20, 30, None]:
        train_RF(n_est, depth)
        


#Best parameters are 
#Est: 50 / Depth: None ---- Precision: 0.992 / Recall: 0.907 / Accuracy: 0.976

##################### Final Evaluation #############################


X_train, X_test, y_train, y_test = train_test_split(dataset[['text', 'body_len', 'punct%']], dataset['spam'], test_size=0.2)

tfidf_vect = TfidfVectorizer(analyzer=lemmatizing)
tfidf_vect_fit = tfidf_vect.fit(X_train['text'])

tfidf_train = tfidf_vect_fit.transform(X_train['text'])
tfidf_test = tfidf_vect_fit.transform(X_test['text'])

X_train_vect = pd.concat([X_train[['body_len', 'punct%']].reset_index(drop=True), 
           pd.DataFrame(tfidf_train.toarray())], axis=1)
X_test_vect = pd.concat([X_test[['body_len', 'punct%']].reset_index(drop=True), 
           pd.DataFrame(tfidf_test.toarray())], axis=1)

X_train_vect.head()

#Choosing best parameters
rf = RandomForestClassifier(n_estimators=50, max_depth=None, n_jobs=-1)

start = time.time()
rf_model = rf.fit(X_train_vect, y_train)
end = time.time()
fit_time = (end - start)

start = time.time()
y_pred = rf_model.predict(X_test_vect)
end = time.time()
pred_time = (end - start)

precision, recall, fscore, train_support = score(y_test, y_pred, pos_label=1, average='binary')
print('Fit time: {} / Predict time: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(fit_time, 3), round(pred_time, 3), round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))