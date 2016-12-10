from NewsGroups20.ml_models import *
def get_predictions(algo):
    total_categories = [
                    'atheism', 'christian',
                    'computer graphics', 'medicine',
                    'Microsoft',
                    'IBM',
                    'mac','Sale','Automobiles',
                    'motorcycles',
                    'baseball',
                    'hockey','politics',
                    'guns',
                    'mideast','cryptography',
                    'electronics',
                    'space','religion'
                  ]
    dict_predicted={}
    for i in total_categories:
        new_docs=[]
        result={}
        with open(total_categories[i] + '.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                new_docs.append(line)
            if algo=='naive_bayes':
                model_id = algo + "_tfidf_1_1_2"
            elif algo=='svm':
                model_id = algo + "_tfidf_1_3_0.0001_5"
            model = retrieve_from_db(model_id)
            clf = model.param1
            tfidf_transformer = model.param2
            count_vect = model.param3
            X_new_counts = count_vect.transform(new_docs)
            X_new_tf = tfidf_transformer.transform(X_new_counts)
            if algo == 'naive_bayes':
                y_score = clf.predict_proba(X_new_tf)
            elif algo == 'svm':
                y_score = clf.decision_function(X_new_tf)
            predict_list = y_score.tolist()
            predict_dict = {}
            for j in range(0,len(new_docs)):
                for k in range(0, len(clf.classes_)):
                    predict_dict[clf.classes_[k]] = predict_list[j][k]
                sorted_predict_dict = sorted(predict_dict.items(), key=lambda x: x[1], reverse=True)
                result[new_docs[j]]=sorted_predict_dict
        dict_predicted[total_categories[i]]=result
    return dict_predicted