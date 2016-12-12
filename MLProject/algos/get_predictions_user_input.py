import pickle


def get_predictions(algo,text):
    dict_predicted={}
    new_docs=[text]
    if algo=='naive_bayes':
        model_id = algo + "_tfidf_1_1_2"
    elif algo=='svm':
        model_id = algo + "_tfidf_1_3_0.0001_5"
    model = pickle.load(open('algos/'+model_id,'rb'))
    clf = model.param1
    tfidf_transformer = model.param2
    count_vect = model.param3
    X_new_counts = count_vect.transform(new_docs)
    X_new_tf = tfidf_transformer.transform(X_new_counts)
    if algo == 'naive_bayes':
        y_score = clf.predict_proba(X_new_tf)
        predict_list = y_score.tolist()
    elif algo == 'svm':
        y_score = [normalize_data(clf.decision_function(X_new_tf).tolist()[0])]
        predict_list = y_score
    predict_dict = {}
    for i in range(0, len(clf.classes_)):
        predict_dict[clf.classes_[i]] = predict_list[0][i]
    sorted_predict_dict = sorted(predict_dict.items(), key=lambda x: x[1], reverse=True)
    dict_predicted[text]=sorted_predict_dict
    return sorted_predict_dict



class PickleObjectWrapper:
    def __init__(self):
        self.param1 = None
        self.param2 = None
        self.param3 = None
        self.param4 = None
        self.param5 = None
        self.param6 = None


def normalize_data(x):
    min_x = min(x)
    max_x = max(x)
    z = x[:]
    for i in range(len(x)):
        z[i] = (x[i]-min_x)/(max_x-min_x)
    return z