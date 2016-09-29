# # Tutorial: Machine Learning with Text in scikit-learn

# ## Agenda
# 
# 1. Model building in scikit-learn (refresher)
# 2. Representing text as numerical data
# 3. Reading a text-based dataset into pandas
# 4. Vectorizing our dataset
# 5. Building and evaluating a model
# 6. Comparing models

# for Python 2: use print only as a function
from __future__ import print_function


# ## Part 1: Model building in scikit-learn (refresher)

# load the iris dataset as an example
from sklearn.datasets import load_iris
iris = load_iris()


# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target


# **"Features"** are also known as predictors, inputs, or attributes. The **"response"** is also known as the target, label, or output.

# check the shapes of X and y
print(X.shape)
print(y.shape)


# **"Observations"** are also known as samples, instances, or records.

# examine the first 5 rows of the feature matrix (including the feature names)
import pandas as pd
pd.DataFrame(X, columns=iris.feature_names).head()


# examine the response vector
print(y)


# In order to **build a model**, the features must be **numeric**, and every observation must have the **same features in the same order**.

# import the class
from sklearn.neighbors import KNeighborsClassifier

# instantiate the model (with the default parameters)
knn = KNeighborsClassifier()

# fit the model with data (occurs in-place)
knn.fit(X, y)


# In order to **make a prediction**, the new observation must have the **same features as the training observations**, both in number and meaning.

# predict the response for a new observation
knn.predict([[3, 5, 4, 2]])


# ## Part 2: Representing text as numerical data

# example text for model training (SMS messages)
simple_train = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']


# example response vector
is_desperate = [0, 0, 1]


# From the [scikit-learn documentation](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction):
# 
# > Text Analysis is a major application field for machine learning algorithms. However the raw data, a sequence of symbols cannot be fed directly to the algorithms themselves as most of them expect **numerical feature vectors with a fixed size** rather than the **raw text documents with variable length**.
# 
# We will use [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) to "convert text into a matrix of token counts":

# import and instantiate CountVectorizer (with the default parameters)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()


# learn the 'vocabulary' of the training data (occurs in-place)
vect.fit(simple_train)


# examine the fitted vocabulary
vect.get_feature_names()


# transform training data into a 'document-term matrix'
simple_train_dtm = vect.transform(simple_train)
simple_train_dtm


# convert sparse matrix to a dense matrix
simple_train_dtm.toarray()


# examine the vocabulary and document-term matrix together
pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())


# From the [scikit-learn documentation](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction):
# 
# > In this scheme, features and samples are defined as follows:
# 
# > - Each individual token occurrence frequency (normalized or not) is treated as a **feature**.
# > - The vector of all the token frequencies for a given document is considered a multivariate **sample**.
# 
# > A **corpus of documents** can thus be represented by a matrix with **one row per document** and **one column per token** (e.g. word) occurring in the corpus.
# 
# > We call **vectorization** the general process of turning a collection of text documents into numerical feature vectors. This specific strategy (tokenization, counting and normalization) is called the **Bag of Words** or "Bag of n-grams" representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document.

# check the type of the document-term matrix
type(simple_train_dtm)


# examine the sparse matrix contents
print(simple_train_dtm)


# From the [scikit-learn documentation](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction):
# 
# > As most documents will typically use a very small subset of the words used in the corpus, the resulting matrix will have **many feature values that are zeros** (typically more than 99% of them).
# 
# > For instance, a collection of 10,000 short text documents (such as emails) will use a vocabulary with a size in the order of 100,000 unique words in total while each document will use 100 to 1000 unique words individually.
# 
# > In order to be able to **store such a matrix in memory** but also to **speed up operations**, implementations will typically use a **sparse representation** such as the implementations available in the `scipy.sparse` package.

# build a model to predict desperation
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(simple_train_dtm, is_desperate)


# example text for model testing
simple_test = ["please don't call me"]


# In order to **make a prediction**, the new observation must have the **same features as the training observations**, both in number and meaning.

# transform testing data into a document-term matrix (using existing vocabulary)
simple_test_dtm = vect.transform(simple_test)
simple_test_dtm.toarray()


# examine the vocabulary and document-term matrix together
pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names())


# predict whether simple_test is desperate
knn.predict(simple_test_dtm)


# **Summary:**
# 
# - `vect.fit(train)` **learns the vocabulary** of the training data
# - `vect.transform(train)` uses the **fitted vocabulary** to build a document-term matrix from the training data
# - `vect.transform(test)` uses the **fitted vocabulary** to build a document-term matrix from the testing data (and **ignores tokens** it hasn't seen before)

# ## Part 3: Reading a text-based dataset into pandas

# read file into pandas from the working directory
sms = pd.read_table('sms.tsv', header=None, names=['label', 'message'])


# alternative: read file into pandas from a URL
# url = 'https://raw.githubusercontent.com/justmarkham/pydata-dc-2016-tutorial/master/sms.tsv'
# sms = pd.read_table(url, header=None, names=['label', 'message'])


# examine the shape
sms.shape


# examine the first 10 rows
sms.head(10)


# examine the class distribution
sms.label.value_counts()


# convert label to a numerical variable
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})


# check that the conversion worked
sms.head(10)


# how to define X and y (from the iris data) for use with a MODEL
X = iris.data
y = iris.target
print(X.shape)
print(y.shape)


# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
X = sms.message
y = sms.label_num
print(X.shape)
print(y.shape)


# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## Part 4: Vectorizing our dataset

# instantiate the vectorizer
vect = CountVectorizer()


# learn training data vocabulary, then use it to create a document-term matrix
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)


# equivalently: combine fit and transform into a single step
X_train_dtm = vect.fit_transform(X_train)


# examine the document-term matrix
X_train_dtm


# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm


# ## Part 5: Building and evaluating a model
# 
# We will use [multinomial Naive Bayes](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html):
# 
# > The multinomial Naive Bayes classifier is suitable for classification with **discrete features** (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.

# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# train the model using X_train_dtm
nb.fit(X_train_dtm, y_train)


# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)


# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)


# print message text for the false positives (ham incorrectly classified as spam)


# print message text for the false negatives (spam incorrectly classified as ham)


# example false negative
X_test[3132]


# calculate predicted probabilities for X_test_dtm (poorly calibrated)
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
y_pred_prob


# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)


# ## Part 6: Comparing models
# 
# We will compare multinomial Naive Bayes with [logistic regression](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression):
# 
# > Logistic regression, despite its name, is a **linear model for classification** rather than regression. Logistic regression is also known in the literature as logit regression, maximum-entropy classification (MaxEnt) or the log-linear classifier. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.

# import and instantiate a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# train the model using X_train_dtm
logreg.fit(X_train_dtm, y_train)


# make class predictions for X_test_dtm
y_pred_class = logreg.predict(X_test_dtm)


# calculate predicted probabilities for X_test_dtm (well calibrated)
y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]
y_pred_prob


# calculate accuracy
metrics.accuracy_score(y_test, y_pred_class)


# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)
