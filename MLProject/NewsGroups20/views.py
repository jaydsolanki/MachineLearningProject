from django.shortcuts import render
from django.http import HttpResponse
import json
import algos.multinomialnb as naive_bayes
import algos.svmclassifier as svm
from algos.getdata import total_categories, total_categories_display
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import algos.get_predictions_user_input as user_news
from django.views.decorators.csrf import csrf_exempt
from .models import *
import algos.kmeans as kmean_clustering
import algos.ldm as lda_clustering


def index(request):
    context = {"title": "Home"}
    return render(request, 'index.html', context)


def classification(request):
    if request.method == "GET":
        context = {"title": "Classification"}
        context['total_categories'] = total_categories
        return render(request, "classification.html", context)
    else:
        classifier = request.POST.get('classifier')
        response = {}
        if classifier == "naive_bayes":
            smoothing_factor = int(request.POST.get('smoothing_factor', '2'))
            approach = request.POST.get('approach')
            n_gram = request.POST.get('n_gram', '1,3').split(",")
            n_gram_range = (int(n_gram[0]), int(n_gram[1]))
            # print("Smoothing: "+str(smoothing_factor)+"; approach: "+str(approach)+"; n_gram: "+str(n_gram_range))

            if approach == "tfidf":
                predicted, cnf_matrix, recall, precision, accuracy, error_rate = naive_bayes.make_model_tf_idf(n_gram_range, smoothing_factor)
                # predicted, cnf_matrix, recall, precision, accuracy, error_rate = make_model_tf_idf((1,3), 2)
            else:
                predicted, cnf_matrix, recall, precision, accuracy, error_rate = naive_bayes.make_model(n_gram_range, smoothing_factor)
                # predicted, cnf_matrix, recall, precision, accuracy, error_rate = (1,2,3,4,5,6)
        else:
            learning_rate = float(request.POST.get('learning_rate'))
            num_iterations = int(request.POST.get('num_iterations'))
            approach = request.POST.get('approach')
            n_gram = request.POST.get('n_gram', '1,3').split(",")
            n_gram_range = (int(n_gram[0]), int(n_gram[1]))
            # print("Smoothing: "+str(smoothing_factor)+"; approach: "+str(approach)+"; n_gram: "+str(n_gram_range))
            if approach == "tfidf":
                predicted, cnf_matrix, recall, precision, accuracy, error_rate = svm.make_model_tf_idf(n_gram_range, learning_rate, num_iterations)
                # predicted, cnf_matrix, recall, precision, accuracy, error_rate = make_model_tf_idf((1,3), 2)
            else:
                predicted, cnf_matrix, recall, precision, accuracy, error_rate = svm.make_model(n_gram_range, learning_rate, num_iterations)
                # predicted, cnf_matrix, recall, precision, accuracy, error_rate = (1,2,3,4,5,6)
        response = {"recall": recall, "precision": precision, "accuracy": accuracy, "error_rate": error_rate, "confusion_matrix": cnf_matrix.tolist()
            , "labels": total_categories_display}
        return HttpResponse(json.dumps(response), content_type="application/json", status=200)


def clustering(request):
    context = {"title": "Clustering"}
    return render(request, 'clustering.html', context)


def alchemy_api_search(request):
    categories = ["atheism",
                  "christian",
                  "computer graphics",
                  "medicine",
                  "Microsoft",
                  "IBM",
                  "mac",
                  "Sale",
                  "Automobiles",
                  "motorcycles",
                  "baseball",
                  "hockey",
                  "politics",
                  "guns",
                  "mideast",
                  "cryptography",
                  "electronics",
                  "space",
                  "religion"]
    context = {"title": "Alchemy API Test Results", "categories": categories}
    return render(request, 'alchemy_api_search.html', context)


def alchemy_api_test_results(request):
    alchemy_news = AlchemyNews.objects.filter(category=request.POST['category'])
    result = []
    for news in alchemy_news:
        scores_svm = AlchemyNewsClassification.objects.filter(alchemy_news_id=news.id, algorithm='svm').values_list('category', 'score')
        scores_with_labels_svm_1 = map(lambda x: (total_categories[x[0]], x[1]), scores_svm)
        scores_with_labels_svm = []
        for i in scores_with_labels_svm_1:
            scores_with_labels_svm.append([i[0], i[1]])
        scores_nb = AlchemyNewsClassification.objects.filter(alchemy_news_id=news.id, algorithm='naive_bayes').values_list('category', 'score')
        scores_with_labels_nb_1 = map(lambda x: (total_categories[x[0]], x[1]), scores_nb)
        scores_with_labels_nb = []
        for i in scores_with_labels_nb_1:
            scores_with_labels_nb.append([i[0], i[1]])
        result.append({"content": news.content, "category": news.category, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
    return render(request, "alchemy_api.html", {"title": "Alchemy API Test Results", "news": result})


def live_news(request):
    live_news_objs = LiveNews.objects.all()
    result = []
    main_result = []
    for i in range(20):
        main_result.append([])
    for news in live_news_objs:
        scores_svm = LiveNewsClassification.objects.filter(live_news_id=news.id, algorithm='svm').values_list('category', 'score')
        scores_with_labels_svm_1 = map(lambda x: (total_categories[x[0]], x[1]), scores_svm)
        scores_with_labels_svm = []
        for i in scores_with_labels_svm_1:
            scores_with_labels_svm.append([i[0], i[1]])
        scores_nb = LiveNewsClassification.objects.filter(live_news_id=news.id, algorithm='naive_bayes').values_list('category', 'score')
        scores_with_labels_nb_1 = map(lambda x: (total_categories[x[0]], x[1]), scores_nb)
        scores_with_labels_nb = []
        for i in scores_with_labels_nb_1:
            scores_with_labels_nb.append([i[0], i[1]])
        if scores_with_labels_svm[0][0] == "alt.atheism":
            main_result[0].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "comp.graphics":
            main_result[1].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "comp.os.ms-windows.misc":
            main_result[2].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "comp.sys.ibm.pc.hardware":
            main_result[3].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "comp.sys.mac.hardware":
            main_result[4].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "comp.windows.x":
            main_result[5].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "misc.forsale":
            main_result[6].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "rec.autos":
            main_result[7].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "rec.motorcycles":
            main_result[8].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "rec.sport.baseball":
            main_result[9].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "rec.sport.hockey":
            main_result[10].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "sci.crypt":
            main_result[11].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "sci.electronics":
            main_result[12].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "sci.med":
            main_result[13].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "sci.space":
            main_result[14].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "soc.religion.christian":
            main_result[15].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "talk.politics.guns":
            main_result[16].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "talk.politics.mideast":
            main_result[17].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "talk.politics.misc":
            main_result[18].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
        elif scores_with_labels_svm[0][0] == "talk.religion.misc":
            main_result[19].append({"content": news.description, "title": news.title, "scores_svm": scores_with_labels_svm, "scores_nb": scores_with_labels_nb})
    return render(request, "live_news.html", {"title": "Live News", "all_news": main_result, "total_categories": total_categories})


def classify_news(request):
    return render(request, "classify_news.html", {"title": "Classify News"})


def roc_curve(request):
    if request.method == "POST":
        classifier = request.POST.get('classifier')
        if classifier == "naive_bayes":
            smoothing_factor = request.POST.get('smoothing_factor')
            approach = request.POST.get('approach')
            n_gram = request.POST.get('n_gram')
            target_class = request.POST.get("target_class")
            fpr, tpr, thresholds = naive_bayes.get_metrics_for_roc(approach, n_gram.split(","), smoothing_factor, target_class)
            plt.figure()
            plt.plot(fpr, tpr, color="blue", lw=2, label='ROC curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            sio = BytesIO()
            plt.savefig(sio, format="png")
            plt.clf()
            plt.close()
            response = {"roc_curve": base64.b64encode(sio.getvalue()).decode('ascii')}
            return HttpResponse(json.dumps(response), content_type="application/json", status=200)
        else:
            approach = request.POST.get('approach')
            n_gram = request.POST.get('n_gram')
            learning_rate = request.POST.get('learning_rate')
            num_iterations = request.POST.get('num_iterations')
            target_class = request.POST.get("target_class")
            fpr, tpr, thresholds = svm.get_metrics_for_roc(approach, n_gram.split(","), learning_rate, num_iterations, target_class)
            plt.figure()
            plt.plot(fpr, tpr, color="blue", lw=2, label='ROC curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            sio = BytesIO()
            plt.savefig(sio, format="png")
            plt.clf()
            plt.close()
            response = {"roc_curve": base64.b64encode(sio.getvalue()).decode('ascii')}
            return HttpResponse(json.dumps(response), content_type="application/json", status=200)


@csrf_exempt
def user_news_classification(request):
    if request.method == "POST":
        classifier = request.POST.get("classifier")
        news = request.POST.get("news")
        class_dict = user_news.get_predictions(classifier, news)
        class_names = []
        scores = []
        count = 20
        for data in class_dict:
            count -= 1
            class_names.append(total_categories[data[0]])
            scores.append(round(data[1] * 100, 2))
            if count == 0:
                break
        return HttpResponse(status=200, content_type="application/json", content=json.dumps({"scores": scores, "class_names": class_names}))


@csrf_exempt
def kmeans(request):
    homogeneity_score, completeness_score, v_measure_score, counts, total_labels, total_count = kmean_clustering.cluster(int(request.POST.get('num_features', 10000)), int(request.POST.get('random_state', 3)))
    return HttpResponse(status=200, content_type='application/json', content=json.dumps({'scores': [homogeneity_score, completeness_score, v_measure_score], "counts": counts, "total_labels": total_labels, "total_count": total_count}))


@csrf_exempt
def lda(request):
    data, charts = lda_clustering.topic_modelling(int(request.POST.get('num_features', 10000)), int(request.POST.get('top_words', 10)))
    return HttpResponse(status=200, content_type='application/json', content=json.dumps({"data": data, "charts": charts}))
