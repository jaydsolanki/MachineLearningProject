import requests
import json
import mysql.connector
import pickle

conn = mysql.connector.connect(user='root', password='', database='ml_project')
cursor = conn.cursor()

def normalize_data(x):
    min_x = min(x)
    max_x = max(x)
    z = x[:]
    for i in range(len(x)):
        z[i] = (x[i] - min_x) / (max_x - min_x)
    return z


while True:
    try:
        r = requests.get('https://newsapi.org/v1/articles?source=cnn&sortBy=top&apiKey=3a47760a55b34dfaa23d897c5d475972')
        res = r.json()
        results = res['articles']
        for result in results:
            description = result['description']
            title = result['title']
            if type(title)==tuple:
                title = title[0]
            publishedAt = result['publishedAt']
            print("TITLE: "+str(title))
            print("DESCRIPTION: "+str(description))
            print("PUBLISHEDAT: "+str(publishedAt))
            if publishedAt:
                published_date, published_time = publishedAt.split("T")
                published_date = "'"+published_date.replace("'","''")+"'" if published_date else 'null'
                published_date_val = '='+ published_date
                published_time = "'"+published_time.replace("'","''").replace('Z','')+"'" if published_time else 'null'
            else:
                published_date_val = ' is null'
                published_date = 'null'
                published_time = 'null'
            description = "'"+description.replace("'","''")+"'" if description else 'null'
            title = "'"+title.replace("'","''")+"'" if title\
                else 'null'
            title_val = '='+title if title!='null' else 'is null'
            cursor.execute("SELECT 1 from live_news where title"+title_val+" and published_date"+published_date_val)
            data = cursor.fetchall()
            if len(data)>0:
                continue
            ####
            new_docs = [title+" "+description]
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
                insert_query = "INSERT INTO live_news (title, published_date, published_time, description) VALUES ("+title+","+published_date+","+published_time+","+description+")"
                cursor.execute(insert_query)
                conn.commit()
                cursor.execute("SELECT max(id) from live_news")
                live_news_id = cursor.fetchall()[0][0]
                for k in range(0, len(clf_nb.classes_)):
                    predict_dict_nb[clf_nb.classes_[k]] = predict_list_nb[j][k]
                for k in range(0, len(clf_svm.classes_)):
                    predict_dict_svm[clf_svm.classes_[k]] = predict_list_svm[j][k]
                sorted_predict_dict_nb = sorted(predict_dict_nb.items(), key=lambda x: x[1], reverse=True)
                sorted_predict_dict_svm = sorted(predict_dict_svm.items(), key=lambda x: x[1], reverse=True)
                for k in range(len(sorted_predict_dict_nb)):
                    cursor.execute("INSERT INTO live_news_classification (live_news_id, category, score, algorithm) values (" + str(live_news_id) + "," + str(sorted_predict_dict_nb[k][0]) + "," + str(
                        sorted_predict_dict_nb[k][1]) + ",'naive_bayes')")
                for k in range(len(sorted_predict_dict_svm)):
                    cursor.execute("INSERT INTO live_news_classification (live_news_id, category, score, algorithm) values (" + str(live_news_id) + "," + str(sorted_predict_dict_svm[k][0]) + "," + str(
                        sorted_predict_dict_svm[k][1]) + ",'svm')")
                conn.commit()

            #####
            print(insert_query)
            # with open('streaming_news.json', 'a') as f:
            #     f.write(json.dumps(dict_news))
            #     f.write('\n')
    except Exception as e:
        print("Exception encountered: " + str(e))
        cursor.close()
        conn.close()
        break
