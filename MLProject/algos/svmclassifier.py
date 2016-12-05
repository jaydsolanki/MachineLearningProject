from .getdata import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# from sklearn.metrics import zero_one_score # NOT FOUND
from sklearn.metrics import precision_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import numpy as np
from NewsGroups20.models import *


def train(train_data, ngram_range=(1,1), learning_rate=0.0001, num_iterations=5):
    count_vect = CountVectorizer(analyzer='word', ngram_range=ngram_range, stop_words='english')
    X_train_counts = count_vect.fit_transform(train_data.data)
    clf = SGDClassifier(loss="hinge", alpha=learning_rate, n_iter=num_iterations, random_state=42).fit(X_train_counts, train_data.target)
    return clf,count_vect


def test(test_data, clf, count_vect):
    X_new_counts = count_vect.transform(test_data.data)
    predicted = clf.predict(X_new_counts)
    y_score = clf.decision_function(X_new_counts)
    score=np.transpose(y_score)
    fpr, tpr, thresholds = metrics.roc_curve(test_data.target, score[2], pos_label=2)
    cnf_matrix = confusion_matrix(test_data.target, predicted)
    recall = recall_score(test_data.target, predicted, average='weighted')
    precision = precision_score(test_data.target, predicted, average='weighted')
    accuracy = accuracy_score(test_data.target, predicted, normalize=True)
    error_rate = 1 - accuracy
    return predicted, cnf_matrix, recall, precision, accuracy, error_rate, fpr, tpr, thresholds


def train_with_tf_idf(train_data, ngram_range=(1,1), learning_rate=0.0001, num_iterations=5):
    count_vect = CountVectorizer(analyzer='word', ngram_range=ngram_range, stop_words='english')
    X_train_counts = count_vect.fit_transform(train_data.data)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = SGDClassifier(loss="hinge", alpha=learning_rate, n_iter=num_iterations, random_state=42).fit(X_train_tfidf, train_data.target)
    return clf, tfidf_transformer,count_vect


def test_tf_idf(test_data, clf,tfidf_transformer, count_vect):
    X_new_counts = count_vect.transform(test_data.data)
    X_new_tf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tf)
    y_score = clf.decision_function(X_new_tf)
    score=np.transpose(y_score)
    fpr, tpr, thresholds = metrics.roc_curve(test_data.target, score[2], pos_label=2)
    cnf_matrix = confusion_matrix(test_data.target, predicted)
    recall = recall_score(test_data.target, predicted, average='weighted')
    precision = precision_score(test_data.target, predicted, average='weighted')
    accuracy = accuracy_score(test_data.target, predicted, normalize=True)
    error_rate = 1 - accuracy
    return predicted, cnf_matrix, recall, precision, accuracy, error_rate, fpr, tpr, thresholds


def make_model(ngram_range, learning_rate, num_iterations):
    model_id = "svm_"+str(ngram_range[0])+"_"+str(learning_rate)+"_"+str(num_iterations)
    a = MlModels.objects.get(id=model_id)
    train_data = get_train_data(random_state=42)
    test_data = get_test_data(random_state=42)
    clf, count_vect = train(train_data, ngram_range=ngram_range, learning_rate=learning_rate, num_iterations=num_iterations)
    predicted, cnf_matrix, recall, precision, accuracy, error_rate, fpr, tpr, thresholds=test(test_data, clf, count_vect)
    return predicted, cnf_matrix, recall, precision, accuracy, error_rate, fpr, tpr, thresholds


def make_model_tf_idf(ngram_range, learning_rate, num_iterations):
    train_data = get_train_data(random_state=42)
    test_data = get_test_data(random_state=42)
    clf, tfidf_transformer, count_vect = train_with_tf_idf(train_data, ngram_range=ngram_range, learning_rate=learning_rate, num_iterations=num_iterations)
    predicted, cnf_matrix, recall, precision, accuracy, error_rate, fpr, tpr, thresholds=test_tf_idf(test_data, clf,tfidf_transformer, count_vect)
    return predicted, cnf_matrix, recall, precision, accuracy, error_rate, fpr, tpr, thresholds


