import pickle
import mysql.connector
import os


def get_predictions():
    total_categories = [
        'atheism', 'christian',
        'computer graphics', 'medicine',
        'Microsoft',
        'IBM',
        'mac', 'Sale', 'Automobiles',
        'motorcycles',
        'baseball',
        'hockey', 'politics',
        'guns',
        'mideast', 'cryptography',
        'electronics',
        'space', 'religion'
    ]
    conn = mysql.connector.connect(user='root', database='ml_project')
    cursor = conn.cursor()
    for i in range(len(total_categories)):
        new_docs = []
        if not os.path.exists(total_categories[i] + '.txt'):
            continue
        with open(total_categories[i] + '.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                new_docs.append(line)
            model_id_nb = "naive_bayes" + "_tfidf_1_1_2"
            model_id_svm = "svm"+ "_tfidf_1_3_0.0001_5"
            model_nb = pickle.load(open(model_id_nb, 'rb'))
            model_svm = pickle.load(open(model_id_svm, 'rb'))
            clf_nb = model_nb.param1
            clf_svm = model_svm.param1
            tfidf_transformer_nb = model_nb.param2
            count_vect_nb = model_nb.param3
            tfidf_transformer_svm = model_svm.param2
            count_vect_svm = model_svm.param3
            X_new_counts_nb = count_vect_nb.transform(new_docs)
            X_new_tf_nb = tfidf_transformer_nb.transform(X_new_counts_nb)
            X_new_counts_svm = count_vect_svm.transform(new_docs)
            X_new_tf_svm = tfidf_transformer_svm.transform(X_new_counts_svm)
            y_score_nb = clf_nb.predict_proba(X_new_tf_nb)
            predict_list_nb = y_score_nb.tolist()
            y_score_svm = clf_svm.decision_function(X_new_tf_svm).tolist()
            # y_score_svm = [normalize_data(clf_svm.decision_function(X_new_tf_svm).tolist()[0])]
            predict_list_svm = []
            for arr in y_score_svm:
                predict_list_svm.append(normalize_data(arr))
            predict_dict_nb = {}
            predict_dict_svm = {}
            for j in range(0, len(new_docs)):
                insert_news = "INSERT INTO alchemy_news (content, category) VALUES ('" + new_docs[j].replace("'", "''") + "', '" + total_categories[i] + "')"
                cursor.execute(insert_news)
                conn.commit()
                cursor.execute("select max(id) from alchemy_news")
                alchemy_news_id = cursor.fetchall()[0][0]
                for k in range(0, len(clf_nb.classes_)):
                    predict_dict_nb[clf_nb.classes_[k]] = predict_list_nb[j][k]
                for k in range(0, len(clf_svm.classes_)):
                    predict_dict_svm[clf_svm.classes_[k]] = predict_list_svm[j][k]
                sorted_predict_dict_nb = sorted(predict_dict_nb.items(), key=lambda x: x[1], reverse=True)
                sorted_predict_dict_svm = sorted(predict_dict_svm.items(), key=lambda x: x[1], reverse=True)
                for k in range(len(sorted_predict_dict_nb)):
                    cursor.execute("INSERT INTO alchemy_news_classification (alchemy_news_id, category, score, algorithm) values (" + str(alchemy_news_id) + "," + str(sorted_predict_dict_nb[k][0]) + "," + str(
                        sorted_predict_dict_nb[k][1]) + ",'naive_bayes')")
                for k in range(len(sorted_predict_dict_svm)):
                    cursor.execute("INSERT INTO alchemy_news_classification (alchemy_news_id, category, score, algorithm) values (" + str(alchemy_news_id) + "," + str(sorted_predict_dict_svm[k][0]) + "," + str(
                        sorted_predict_dict_svm[k][1]) + ",'svm')")
                conn.commit()

    cursor.close()
    conn.close()


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
        z[i] = (x[i] - min_x) / (max_x - min_x)
    return z


get_predictions()
