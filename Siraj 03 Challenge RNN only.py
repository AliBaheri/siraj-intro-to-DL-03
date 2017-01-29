
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

# In[1]:

n_epoch = int(input('Enter no. of Epoch for RNN training: '))


# In[2]:

import pandas as pd
pd.set_option('display.max_colwidth', 1000)


# # Load IGN Dataset as `original_ign`

# In[3]:

original_ign = pd.read_csv('ign.csv')
original_ign.head(10)


# ### Check out the `shape` of the IGN Dataset

# In[4]:

original_ign.shape


# ### Check out all the unique `score_phrase` as well as their `counts`

# In[5]:

original_ign.score_phrase.value_counts()


# # Data Preprocessing
# 
# As always, we gotta perform preprocessing on our Dataset before training our Model.

# ### Convert `score_phrase` to binary sentiments and add a new column called `sentiment`

# In[6]:

bad_phrases = ['Bad', 'Awful', 'Painful', 'Unbearable', 'Disaster']
original_ign['sentiment'] = original_ign.score_phrase.isin(bad_phrases).map({True: 'Negative', False: 'Positive'})


# In[7]:

# Remove "Disaster"
original_ign = original_ign[original_ign['score_phrase'] != 'Disaster']


# In[8]:

original_ign.shape


# In[9]:

original_ign.head()


# ### No. of Positive Sentiments VS No. of Negative Seniments

# In[10]:

original_ign.sentiment.value_counts(normalize=True)


# ### Check for null elements

# In[11]:

original_ign.isnull().sum()


# ### Fill all null elements with an empty string

# In[12]:

original_ign.fillna(value='', inplace=True)


# In[13]:

# original_ign[ original_ign['genre'] == '' ].shape


# # Create a new DataFrame called `ign`

# In[14]:

ign = original_ign[ ['sentiment', 'score_phrase', 'title', 'platform', 'genre', 'editors_choice'] ].copy()
ign.head(10)


# ### Create a new column called `is_editors_choice`

# In[15]:

ign['is_editors_choice'] = ign['editors_choice'].map({'Y': 'editors_choice', 'N': ''})
ign.head()


# ### Create a new column called `text` which contains contents of several columns

# In[16]:

ign['text'] = ign['title'].str.cat(ign['platform'], sep=' ').str.cat(ign['genre'], sep=' ').str.cat(ign['is_editors_choice'], sep=' ')


# In[17]:

print('Shape of \"ign\" DataFrame:', ign.shape)


# In[18]:

ign.head(10)


# ![http://www.westernbands.net/userdata/news_picupload/pic_sid1070-0-norm.jpg](http://www.westernbands.net/userdata/news_picupload/pic_sid1070-0-norm.jpg)

# In[19]:

X = ign.text
y = ign.score_phrase


# # Model #0: The DUMMY Classifier (Always Choose the Most Frequent Class)

# In[20]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


# In[21]:

vect = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w{2,}\b')
dummy = DummyClassifier(strategy='most_frequent', random_state=0)

dummy_pipeline = make_pipeline(vect, dummy)


# In[22]:

dummy_pipeline.named_steps


# In[23]:

# Cross Validation
cv = cross_val_score(dummy_pipeline, X, y, scoring='accuracy', cv=10, n_jobs=-1)
print('Dummy Classifier\'s Accuracy: %0.5f' % cv.mean())


# # Model #1: RNN Sentiment Classifier

# In[24]:

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb


# ### Train-Test-Split

# In[25]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# ### Create the `vocab` (so that we can create `X_word_ids` from `X`)

# In[26]:

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(ngram_range=(1,1), token_pattern=r'\b\w{1,}\b')


# In[27]:

vect.fit(X_train)
vocab = vect.vocabulary_


# In[28]:

def convert_X_to_X_word_ids(X):
    return X.apply( lambda x: [vocab[w] for w in [w.lower().strip() for w in x.split()] if w in vocab] )


# In[29]:

X_train_word_ids = convert_X_to_X_word_ids(X_train)
X_test_word_ids  = convert_X_to_X_word_ids(X_test)


# ### Difference between X(_train/_test) and X(_train_word_ids/test_word_ids)

# In[30]:

X_train.head()


# In[31]:

X_train_word_ids.head()


# In[32]:

print(X_train_word_ids.shape)
print(X_test_word_ids.shape)


# ### Sequence Padding

# In[33]:

X_train_padded_seqs = pad_sequences(X_train_word_ids, maxlen=20, value=0)
X_test_padded_seqs  = pad_sequences(X_test_word_ids , maxlen=20, value=0)


# In[34]:

print(X_train_padded_seqs.shape)
print(X_test_padded_seqs.shape)


# In[35]:

pd.DataFrame(X_train_padded_seqs).head()


# In[36]:

pd.DataFrame(X_test_padded_seqs).head()


# ### Convert (y) labels to vectors

# In[37]:

unique_y_labels = list(y_train.value_counts().index)
unique_y_labels


# In[38]:

len(unique_y_labels)


# In[39]:

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(unique_y_labels)


# In[40]:

print(unique_y_labels)
print(le.transform(unique_y_labels))


# In[41]:

for label_id, label_name in zip(le.transform(unique_y_labels), unique_y_labels):
    print('%d: %s' % (label_id, label_name))


# In[42]:

y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), nb_classes=len(unique_y_labels))
y_test  = to_categorical(y_test.map(lambda x:  le.transform([x])[0]), nb_classes=len(unique_y_labels))


# In[43]:

y_train[0:3]


# In[44]:

print(y_train.shape)
print(y_test.shape)


# ### Network Building

# In[45]:

size_of_each_vector = X_train_padded_seqs.shape[1]
vocab_size = len(vocab)
no_of_unique_y_labels = len(unique_y_labels)


# In[46]:

print(size_of_each_vector)
print(vocab_size)
print(no_of_unique_y_labels)


# In[47]:

net = tflearn.input_data([None, size_of_each_vector]) # The first element is the "batch size" which we set to "None"
net = tflearn.embedding(net, input_dim=vocab_size, output_dim=128) # input_dim: Vocabulary size (number of ids)
net = tflearn.lstm(net, 128, dropout=0.5) # Long Short Term Memory Recurrent Layer
net = tflearn.fully_connected(net, no_of_unique_y_labels, activation='softmax') # relu or softmax
net = tflearn.regression(net, 
                         optimizer='adam', 
                         learning_rate=1e-4,
                         loss='categorical_crossentropy')


# ### Train the Model

# In[48]:

model = tflearn.DNN(net, tensorboard_verbose=0, checkpoint_path='model.tfl.ckpt')

model.fit(X_train_padded_seqs, y_train, 
          validation_set=(X_test_padded_seqs, y_test), 
          n_epoch=n_epoch,
          show_metric=True, 
          batch_size=100)


# ### Manually Save the Model

# In[61]:

model.save("model.tfl")
print('Model Saved!')


# ### Manually Load the Model

# In[63]:

# model.load("model.tfl")
# print('Model Loaded!')a


# ### Use `scikit-learn`'s metric for evaluation

# In[50]:

import numpy as np
from sklearn import metrics


# ### Accuracy

# In[52]:

pred_class = [np.argmax(i) for i in model.predict(X_test_padded_seqs)]
true_class = [np.argmax(i) for i in y_test]

print('Accuracy (RNN model):', metrics.accuracy_score(true_class, pred_class))


# ### Show some predicted samples (focus on True Negatives & False Negatives -- as the "Negatives" are the minority class)

# In[60]:

ids_of_titles = range(0,10) # range(X_test.shape[0]) 

n_of_true_negatives = 0
n_of_false_negatives = 0

for i in ids_of_titles:
    pred_class = np.argmax(model.predict([X_test_padded_seqs[i]]))
    true_class = np.argmax(y_test[i])
    
    print(X_test.values[i])
    print('pred_class:', le.inverse_transform(pred_class))
    print('true_class:', le.inverse_transform(true_class))
    print('')

