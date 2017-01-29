
# coding: utf-8

# # Siraj's 03 Challenge
# 
# The challenge for this video is to train a model on this dataset of video game reviews from IGN.com. Then, given some new video game title it should be able to classify it. You can use pandas to parse this dataset. 
# 
# Right now each review has a label that's either Amazing, Great, Good, Mediocre, Painful, or Awful. These are the emotions. Using the existing labels is extra credit. 
# 
# The baseline is that you can just convert the labels so that there are only 2 emotions (positive or negative). Ideally you can use an RNN via TFLearn like the one in this example, but I'll accept other types of ML models as well.
# 
# You'll learn how to parse data, select appropriate features, and use a neural net on an IRL problem. 
# 
# Remember to ask each other questions for help in the Slack channel. Good luck!

# In[ ]:

n_epoch = int(input('Enter no. of Epoch for RNN training: '))


# In[ ]:

print('n_epoch:', n_epoch)


# In[ ]:

import pandas as pd
pd.set_option('display.max_colwidth', 1000)


# # Load IGN Dataset as `original_ign`

# In[ ]:

original_ign = pd.read_csv('ign.csv')
original_ign.head(10)


# ### Check out the `shape` of the IGN Dataset

# In[ ]:

original_ign.shape


# ### Check out all the unique `score_phrase` as well as their `counts`

# In[ ]:

original_ign.score_phrase.value_counts()


# # Data Preprocessing
# 
# As always, we gotta perform preprocessing on our Dataset before training our Model.

# ### Convert `score_phrase` to binary sentiments and add a new column called `sentiment`

# In[ ]:

bad_phrases = ['Bad', 'Awful', 'Painful', 'Unbearable', 'Disaster']
original_ign['sentiment'] = original_ign.score_phrase.isin(bad_phrases).map({True: 'Negative', False: 'Positive'})


# In[ ]:

# Remove "Disaster"
# original_ign = original_ign[original_ign['score_phrase'] != 'Disaster']


# In[ ]:

original_ign.shape


# In[ ]:

original_ign.head()


# ### No. of Positive Sentiments VS No. of Negative Seniments

# In[ ]:

original_ign.sentiment.value_counts(normalize=True)


# ### Check for null elements

# In[ ]:

original_ign.isnull().sum()


# ### Fill all null elements with an empty string

# In[ ]:

original_ign.fillna(value='', inplace=True)


# In[ ]:

# original_ign[ original_ign['genre'] == '' ].shape


# # Create a new DataFrame called `ign`

# In[ ]:

ign = original_ign[ ['sentiment', 'score_phrase', 'title', 'platform', 'genre', 'editors_choice'] ].copy()
ign.head(10)


# ### Create a new column called `is_editors_choice`

# In[ ]:

ign['is_editors_choice'] = ign['editors_choice'].map({'Y': 'editors_choice', 'N': ''})
ign.head()


# ### Create a new column called `text` which contains contents of several columns

# In[ ]:

ign['text'] = ign['title'].str.cat(ign['platform'], sep=' ').str.cat(ign['genre'], sep=' ').str.cat(ign['is_editors_choice'], sep=' ')


# In[ ]:

ign.shape


# In[ ]:

ign.head(10)


# ![http://www.westernbands.net/userdata/news_picupload/pic_sid1070-0-norm.jpg](http://www.westernbands.net/userdata/news_picupload/pic_sid1070-0-norm.jpg)

# In[ ]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


# In[ ]:

X = ign.text
y = ign.sentiment  # ign.score_phrase


# # Model #0: The DUMMY Classifier (Always Choose the Most Frequent Class)

# In[ ]:

from sklearn.dummy import DummyClassifier


# In[ ]:

vect = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w{2,}\b')
dummy = DummyClassifier(strategy='most_frequent', random_state=0)

dummy_pipeline = make_pipeline(vect, dummy)


# In[ ]:

dummy_pipeline.named_steps


# In[ ]:

# Cross Validation
cv = cross_val_score(dummy_pipeline, X, y, scoring='accuracy', cv=10, n_jobs=-1)
print(cv.mean())


# # Model #1: MultinomialNB Sentiment Classifier

# In[ ]:

from sklearn.naive_bayes import MultinomialNB        


# ### Pipeline = TfidfVectorizer + MultinomialNB

# In[ ]:

vect = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w{2,}\b')
clf = MultinomialNB()

pipeline = make_pipeline(vect, clf)


# In[ ]:

pipeline.named_steps


# In[ ]:

# Cross Validation
cv = cross_val_score(pipeline, X, y, scoring='accuracy', cv=10, n_jobs=-1)
cv.mean()  


# ### Now, let's tune our parameters using `scikit-learn`'s Grid Search

# In[ ]:

from sklearn.model_selection import GridSearchCV     


# In[ ]:

param_dict = {'tfidfvectorizer__min_df'     : [1], 
              'tfidfvectorizer__max_df'     : [0.1],
              'tfidfvectorizer__ngram_range': [(1,2)],
              'multinomialnb__alpha'        : [2]
              }

estimator = GridSearchCV( pipeline, param_dict, scoring='accuracy', cv=10, n_jobs=-1 )
estimator.fit(X, y)


# In[ ]:

estimator.best_estimator_


# In[ ]:

estimator.best_params_


# In[ ]:

estimator.best_score_


# ### Prediction

# In[ ]:

ids_of_titles = range(1, 4)


# In[ ]:

for i in ids_of_titles:
    print(X[i])
    print(estimator.predict([X[i]]))
    print()


# # Model #2: RNN Sentiment Classifier

# In[ ]:

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb


# In[ ]:

from sklearn.model_selection import train_test_split


# In[ ]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)


# ### Create `X_train_dtm`

# In[ ]:

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(ngram_range=(1,1), token_pattern=r'\b\w{1,}\b')


# In[ ]:

X_train_dtm = vect.fit_transform(X_train)


# In[ ]:

X_train_dtm_df = pd.DataFrame(X_train_dtm.toarray())
X_train_dtm_df.head()


# In[ ]:

X_train_dtm_df.shape


# ### Create `X_test_dtm`

# In[ ]:

X_test_dtm = vect.transform(X_test)


# In[ ]:

X_test_dtm_df = pd.DataFrame(X_test_dtm.toarray())
X_test_dtm_df.head()


# In[ ]:

X_test_dtm_df.shape


# ### Sequence Padding

# In[ ]:

def helper_func(some_list):
    output = list()
    for idx, elem in enumerate(some_list):
        if elem != 0:
            output.append(idx)
    return output


# In[ ]:

X_test_dtm_df.apply(lambda x: helper_func(x), axis=1).head(10)


# In[ ]:

X_train_padded_sequences = pad_sequences(X_train_dtm_df.apply(lambda x: helper_func(x), axis=1), maxlen=100, value=0)
X_test_padded_sequences  = pad_sequences(X_test_dtm_df.apply( lambda x: helper_func(x), axis=1), maxlen=100, value=0)


# In[ ]:

print(X_train_padded_sequences.shape)
print(X_test_padded_sequences.shape)


# In[ ]:

pd.DataFrame(X_train_padded_sequences).head()


# In[ ]:

pd.DataFrame(X_test_padded_sequences).head()


# ### Convert (y) labels to vectors

# In[ ]:

y_train = to_categorical(y_train.map({'Positive':1, 'Negative':0}), nb_classes=2)
y_test  = to_categorical(y_test.map( {'Positive':1, 'Negative':0}), nb_classes=2)


# In[ ]:

y_train


# In[ ]:

print(y_train.shape)
print(y_test.shape)


# ### Network Building

# In[ ]:

size_of_input_dim = X_train_dtm_df.shape[1]
size_of_input_dim


# In[ ]:

net = tflearn.input_data([None, 100]) # The first element is the "batch size" which we set to "None"
net = tflearn.embedding(net, input_dim=size_of_input_dim, output_dim=128) # input_dim: Vocabulary size (number of ids)
net = tflearn.lstm(net, 128, dropout=0.5) # Long Short Term Memory Recurrent Layer
net = tflearn.fully_connected(net, 2, activation='softmax') # relu or softmax
net = tflearn.regression(net, 
                         optimizer='adam', 
                         learning_rate=1e-4,
                         loss='categorical_crossentropy')


# ### Train the Model

# In[ ]:

model = tflearn.DNN(net, tensorboard_verbose=0)

model.fit(X_train_padded_sequences, y_train, 
          n_epoch=n_epoch,
          validation_set=(X_test_padded_sequences, y_test), 
          show_metric=True, 
          batch_size=64)


# ### Save Model

# In[ ]:

# Manually save model
model.save("model.tfl")

print('\nModel Saved!\n')


# ### Use `scikit-learn`'s metric for evaluation

# In[ ]:

import numpy as np
from sklearn import metrics


# ### Accuracy

# In[ ]:

pred_class = [np.argmax(i) for i in model.predict(X_test_padded_sequences)]
true_class = [np.argmax(i) for i in y_test]

print('Accuracy:', metrics.accuracy_score(true_class, pred_class))


# ### Show some predicted samples (focus on True Negatives & False Negatives -- as the "Negatives" are the minority class)

# In[ ]:

ids_of_titles = range(X_test.shape[0]) # range(0,10)

n_of_true_negatives = 0
n_of_false_negatives = 0

for i in ids_of_titles:
    pred_class = np.argmax(model.predict([X_test_padded_sequences[i]]))
    true_class = np.argmax(y_test[i])
    
    #True Negative
    if pred_class == true_class == 0:
        if n_of_true_negatives > 5:
            continue
        print('TRUE NEGATIVE [%d of 5]' % n_of_true_negatives)
        print(X_test.values[i])
        print('Pred: %s | Actual: %s' % (pred_class, true_class) )
        print()
        n_of_true_negatives += 1
        
    elif ((pred_class == 0) and (true_class == 1)):
        if n_of_false_negatives > 5:
            continue
        print('*FALSE* NEGATIVE [%d of 5]' % n_of_false_negatives)
        print(X_test.values[i])
        print('Pred: %s | Actual: %s' % (pred_class, true_class) )
        print()
        n_of_false_negatives += 1


# In[ ]:

# ign[ ign.text.str.contains('Attack of the Movies 3D Xbox') ]

