from django.shortcuts import render
from django.http import JsonResponse, Http404
import json
from neuron import operate_nn as op
from neuron import data_creation as dc
from neuron import learn_nn as ln


def get_json(request):
    try:
        requestJson = json.loads(request.body.decode())
    except Exception:
        return JsonResponse({"error": 404})
    #else:
    #    return JsonResponse({"error": 404 })
    #print(request.body)

    if (request.method != "POST" or requestJson is None):
        return JsonResponse({"error": 404})
    #TODO: something with json
    return JsonResponse(requestJson)


def get_check(request):
    try:
        requestJson = json.loads(request.body.decode())
    except Exception:
        return JsonResponse({"error": 404})
    # else:
    #    return JsonResponse({"error": 404 })
    # print(request.body)

    if (request.method != "POST" or requestJson is None):
        return JsonResponse({"error": 404})
    # TODO: something with json
    
    return JsonResponse(op.operate(requestJson["dataSet"]))


def get_train(request):
    try:
        requestJson = json.loads(request.body.decode())
    except Exception:
        return JsonResponse({"error": 404})
    # else:
    #    return JsonResponse({"error": 404 })
    # print(request.body)

    if (request.method != "POST" or requestJson is None):
        return JsonResponse({"error": 404})
    # TODO: something with json
    
    dc.makeDataFile(requestJson["trainingSet"])
    
    return JsonResponse(ln.train())

