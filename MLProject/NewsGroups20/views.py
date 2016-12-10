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


def index(request):
    context = {"title":"Home"}
    return render(request, 'index.html', context)


def classification(request):
    if request.method=="GET":
        context = {"title":"Classification"}
        context['total_categories'] = total_categories
        return render(request, "classification.html", context)
    else:
        classifier = request.POST.get('classifier')
        response = {}
        if classifier=="naive_bayes":
            smoothing_factor = int(request.POST.get('smoothing_factor','2'))
            approach = request.POST.get('approach')
            n_gram = request.POST.get('n_gram', '1,3').split(",")
            n_gram_range = (int(n_gram[0]), int(n_gram[1]))
            # print("Smoothing: "+str(smoothing_factor)+"; approach: "+str(approach)+"; n_gram: "+str(n_gram_range))

            if approach=="tfidf":
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
            if approach=="tfidf":
                predicted, cnf_matrix, recall, precision, accuracy, error_rate = svm.make_model_tf_idf(n_gram_range, learning_rate, num_iterations)
                # predicted, cnf_matrix, recall, precision, accuracy, error_rate = make_model_tf_idf((1,3), 2)
            else:
                predicted, cnf_matrix, recall, precision, accuracy, error_rate = svm.make_model(n_gram_range, learning_rate, num_iterations)
                # predicted, cnf_matrix, recall, precision, accuracy, error_rate = (1,2,3,4,5,6)
        response = {"recall": recall, "precision": precision, "accuracy": accuracy, "error_rate": error_rate, "confusion_matrix":cnf_matrix.tolist()
                        ,"labels": total_categories_display }
        return HttpResponse(json.dumps(response), content_type="application/json", status=200)


def clustering(request):
    context = {"title":"Clustering"}
    return render(request, 'clustering.html', context)


def alchemy_api_test_results(request):
    return render(request, "index.html", {"title":"Alchemy API Test Results"})


def live_news(request):
    return render(request, "index.html", {"title":"Live News"})


def classify_news(request):
    return render(request, "classify_news.html", {"title":"Classify News"})


def roc_curve(request):
    if request.method=="POST":
        classifier = request.POST.get('classifier')
        if classifier=="naive_bayes":
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
    if request.method=="POST":
        classifier = request.POST.get("classifier")
        news = request.POST.get("news")
        class_dict = user_news.get_predictions(classifier, news)
        class_names = []
        scores = []
        count = 5
        for data in class_dict:
            count-=1
            class_names.append(total_categories[data[0]])
            scores.append(data[1])
            if count==0:
                break
        return HttpResponse(status=200, content_type="application/json", content=json.dumps({"scores": scores, "class_names": class_names}))