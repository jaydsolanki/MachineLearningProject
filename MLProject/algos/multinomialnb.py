from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# from sklearn.metrics import zero_one_score # NOT FOUND
from sklearn.metrics import precision_score

from .getdata import *


def train(train_data, ngram_range=(1,1)):
    count_vect = CountVectorizer(analyzer='word', ngram_range=ngram_range,stop_words='english')
    X_train_counts = count_vect.fit_transform(train_data.data)
    clf = MultinomialNB(alpha=2).fit(X_train_counts, train_data.target)
    return clf, count_vect


def test(test_data, clf, count_vect):
    X_new_counts = count_vect.transform(test_data.data)
    predicted = clf.predict(X_new_counts)
    predict_list = predicted.tolist()
    for j in range(len(predict_list)):
        predict_dict={}
        for i in range(0, len(clf.classes_)):
            predict_dict[clf.classes_[i]]=predict_list[j][i]
        sorted_predict_dict = sorted(predict_dict.items(), key=lambda x: x[1], reverse=True)
        print (str(test_data.data[j])+" : "+str(total_categories[sorted_predict_dict[0][0]]))
        print ("\n\n\n\n")
    return sorted_predict_dict


def train_with_tf_idf(train_data, ngram_range=(1,1)):
    count_vect = CountVectorizer(analyzer='word', ngram_range=ngram_range,stop_words='english')
    X_train_counts = count_vect.fit_transform(train_data.data)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB(alpha=2).fit(X_train_tfidf, train_data.target)
    return clf, tfidf_transformer, count_vect


def test_tf_idf(train_data, test_data, clf, tfidf_transformer, count_vect):
    X_new_counts = count_vect.transform(test_data.data)
    X_new_tf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tf)
    predict_list = predicted.tolist()
    cnf_matrix = confusion_matrix(test_data.target, predicted)
    recall = recall_score(test_data.target, predicted, average='weighted')
    precision_score = precision_score(test_data.target, predicted, average='weighted')
    accuracy = accuracy_score(test_data.target, predicted, normalize=True)
    error_rate = 1-accuracy
    for j in range(len(predict_list)):
        predict_dict={}
        for i in range(0, len(clf.classes_)):
            predict_dict[clf.classes_[i]]=predict_list[j][i]
        sorted_predict_dict = sorted(predict_dict.items(), key=lambda x: x[1], reverse=True)
        print (str(test_data.data[j])+" : "+str(total_categories[sorted_predict_dict[0][0]]))
        print ("\n\n\n\n")
    return sorted_predict_dict


def make_model():
    train_data = get_train_data(random_state=42)
    test_data = get_test_data(random_state=42)
    clf, count_vect = train(train_data, ngram_range=(1,5))
    test(train_data, test_data, clf, count_vect)


def make_model_tf_idf():
    train_data = get_train_data(random_state=42)
    test_data = get_test_data(random_state=42)
    clf, tfidf_transformer, count_vect = train_with_tf_idf(train_data, ngram_range=(1,5))
    test_tf_idf(train_data, test_data, clf, tfidf_transformer, count_vect)
