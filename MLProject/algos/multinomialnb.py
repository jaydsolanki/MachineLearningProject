from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn import metrics
import numpy as np
from .getdata import *
from NewsGroups20.models import *
from NewsGroups20.ml_models import *


def train(train_data, ngram_range=(1,1), alpha=2):
    count_vect = CountVectorizer(analyzer='word', ngram_range=ngram_range,stop_words='english')
    X_train_counts = count_vect.fit_transform(train_data.data)
    clf = MultinomialNB(alpha=alpha).fit(X_train_counts, train_data.target)
    return clf, count_vect


def test(test_data, clf, count_vect):
    X_new_counts = count_vect.transform(test_data.data)
    predicted = clf.predict(X_new_counts)
    predicted_pa = clf.predict_proba(X_new_counts)
    score=np.transpose(predicted_pa)
    fpr, tpr, thresholds = metrics.roc_curve(test_data.target, score[2], pos_label=2)
    cnf_matrix = confusion_matrix(test_data.target, predicted)
    recall = recall_score(test_data.target, predicted, average='weighted')
    precision = precision_score(test_data.target, predicted, average='weighted')
    accuracy = accuracy_score(test_data.target, predicted, normalize=True)
    error_rate = 1 - accuracy
    return predicted, cnf_matrix, recall, precision, accuracy, error_rate, fpr, tpr, thresholds


def train_with_tf_idf(train_data, ngram_range=(1,1), alpha=2):
    count_vect = CountVectorizer(analyzer='word', ngram_range=ngram_range,stop_words='english')
    X_train_counts = count_vect.fit_transform(train_data.data)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB(alpha=alpha).fit(X_train_tfidf, train_data.target)
    return clf, tfidf_transformer, count_vect


def test_tf_idf(test_data, clf, tfidf_transformer, count_vect):
    X_new_counts = count_vect.transform(test_data.data)
    X_new_tf = tfidf_transformer.transform(X_new_counts)
    predicted_pa = clf.predict_proba(X_new_tf)
    score=np.transpose(predicted_pa)
    fpr, tpr, thresholds = metrics.roc_curve(test_data.target, score[2], pos_label=2)
    predicted = clf.predict(X_new_tf)
    cnf_matrix = confusion_matrix(test_data.target, predicted)
    recall = recall_score(test_data.target, predicted, average='weighted')
    precision = precision_score(test_data.target, predicted, average='weighted')
    accuracy = accuracy_score(test_data.target, predicted, normalize=True)
    error_rate = 1-accuracy
    return predicted,cnf_matrix,recall,precision,accuracy,error_rate, fpr, tpr, thresholds

def get_metrics_for_roc(algo_type,ngram_range,alpha,target_class_index):
    test_data = get_test_data(random_state=42)
    if algo_type=='word_count':
        algo='wc'
    elif algo_type=='tfidf':
        algo='tfidf'
    model_id = "naive_bayes_"+algo+"_" + str(ngram_range[0]) + "_" + str(ngram_range[1]) + "_" + str(alpha)
    model = retrieve_from_db(model_id)
    clf = model.param1
    if algo=='wc':
        count_vect = model.param2
        X_new_counts = count_vect.transform(test_data.data)
        predicted_pa = clf.predict_proba(X_new_counts)
    elif algo=='tfidf':
        tfidf_transformer = model.param2
        count_vect = model.param3
        X_new_counts = count_vect.transform(test_data.data)
        X_new_tf = tfidf_transformer.transform(X_new_counts)
        predicted_pa = clf.predict_proba(X_new_tf)
    score = np.transpose(predicted_pa)
    fpr, tpr, thresholds = metrics.roc_curve(test_data.target, score[target_class_index], pos_label=target_class_index)
    return fpr, tpr, thresholds


def make_model(ngram_range, alpha):
    model_id = "naive_bayes_wc_"+str(ngram_range[0])+"_"+str(ngram_range[1])+"_"+str(alpha)
    model = retrieve_from_db(model_id)
    if model:
        clf = model.param1
        count_vect = model.param2
    else:
        train_data = get_train_data(random_state=42)
        clf, count_vect = train(train_data, ngram_range=ngram_range, alpha=alpha)
        pow = PickleObjectWrapper()
        pow.param1 = clf
        pow.param2 = count_vect
        dump_to_db(model_id, pow)
    test_data = get_test_data(random_state=42)
    predicted, cnf_matrix, recall, precision, accuracy, error_rate, fpr, tpr, thresholds=test(test_data, clf, count_vect)
    return predicted, cnf_matrix, recall, precision, accuracy, error_rate, fpr, tpr, thresholds


def make_model_tf_idf(ngram_range, alpha):
    model_id = "naive_bayes_tfidf_"+str(ngram_range[0])+"_"+str(ngram_range[1])+"_"+str(alpha)
    model = retrieve_from_db(model_id)
    if model:
        clf = model.param1
        tfidf_transformer = model.param2
        count_vect = model.param3
    else:
        train_data = get_train_data(random_state=42)
        clf, tfidf_transformer, count_vect = train_with_tf_idf(train_data, ngram_range=ngram_range, alpha=alpha)
        pow = PickleObjectWrapper()
        pow.param1 = clf
        pow.param2 = tfidf_transformer
        pow.param3 = count_vect
        dump_to_db(model_id, pow)
    test_data = get_test_data(random_state=42)
    predicted, cnf_matrix, recall, precision, accuracy, error_rate, fpr, tpr, thresholds=test_tf_idf(test_data, clf,
                                                                                     tfidf_transformer, count_vect)
    return predicted, cnf_matrix, recall, precision, accuracy, error_rate, fpr, tpr, thresholds

