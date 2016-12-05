from django.shortcuts import render
from django.http import HttpResponse
import json
import algos.multinomialnb as naive_bayes
import algos.svmclassifier as svm
from algos.getdata import total_categories, total_categories_display
import matplotlib.pyplot as plt
from io import BytesIO
import base64


def index(request):
    context = {"title":"Home"}
    return render(request, 'index.html', context)


def classification(request):
    if request.method=="GET":
        context = {"title":"Classification"}
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

            if approach=="tf_idf":
                predicted, cnf_matrix, recall, precision, accuracy, error_rate, fpr, tpr, thresholds = naive_bayes.make_model_tf_idf(n_gram_range, smoothing_factor)
                # predicted, cnf_matrix, recall, precision, accuracy, error_rate = make_model_tf_idf((1,3), 2)
            else:
                predicted, cnf_matrix, recall, precision, accuracy, error_rate, fpr, tpr, thresholds = naive_bayes.make_model(n_gram_range, smoothing_factor)
                # predicted, cnf_matrix, recall, precision, accuracy, error_rate = (1,2,3,4,5,6)
        else:
            learning_rate = float(request.POST.get('learning_rate'))
            num_iterations = int(request.POST.get('num_iterations'))
            approach = request.POST.get('approach')
            n_gram = request.POST.get('n_gram', '1,3').split(",")
            n_gram_range = (int(n_gram[0]), int(n_gram[1]))
            # print("Smoothing: "+str(smoothing_factor)+"; approach: "+str(approach)+"; n_gram: "+str(n_gram_range))
            if approach=="tf_idf":
                predicted, cnf_matrix, recall, precision, accuracy, error_rate, fpr, tpr, thresholds = svm.make_model_tf_idf(n_gram_range, learning_rate, num_iterations)
                # predicted, cnf_matrix, recall, precision, accuracy, error_rate = make_model_tf_idf((1,3), 2)
            else:
                predicted, cnf_matrix, recall, precision, accuracy, error_rate, fpr, tpr, thresholds = svm.make_model(n_gram_range, learning_rate, num_iterations)
                # predicted, cnf_matrix, recall, precision, accuracy, error_rate = (1,2,3,4,5,6)
        print ("FPR: "+str(fpr.tolist()))
        print ("TPR: "+str(tpr.tolist()))
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
        response = {"recall": recall, "precision": precision, "accuracy": accuracy, "error_rate": error_rate, "confusion_matrix":cnf_matrix.tolist()
                        ,"labels": total_categories_display, "roc_curve": base64.b64encode(sio.getvalue()).decode('ascii') }
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