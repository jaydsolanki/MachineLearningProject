from django.shortcuts import render
from django.http import HttpResponse
import json
import algos.multinomialnb as naive_bayes
from algos.getdata import total_categories


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
                predicted, cnf_matrix, recall, precision, accuracy, error_rate = naive_bayes.make_model_tf_idf(n_gram_range, smoothing_factor)
                #redicted, cnf_matrix, recall, precision, accuracy, error_rate = make_model_tf_idf((1,3), 2)
            else:
                predicted, cnf_matrix, recall, precision, accuracy, error_rate = naive_bayes.make_model(n_gram_range, smoothing_factor)
            # predicted, cnf_matrix, recall, precision, accuracy, error_rate = (1,2,3,4,5,6)
            response = {"recall": recall, "precision": precision, "accuracy": accuracy, "error_rate": error_rate, "confusion_matrix":cnf_matrix.tolist()
                        ,"labels": total_categories}
        else:
            pass
        return HttpResponse(json.dumps(response), content_type="application/json", status=200)