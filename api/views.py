from django.shortcuts import render
from django.http import JsonResponse, Http404
import json

def get_json(request):
	try:
	    requestJson = json.loads(request.body.decode())
	except Exception:
		return JsonResponse({"error": 404 })
	#else:
	#    return JsonResponse({"error": 404 })
	#print(request.body)

	if (request.method != "POST" or requestJson is None):
		return JsonResponse({"error": 404 })
	#TODO: something with json
	return JsonResponse(requestJson)

