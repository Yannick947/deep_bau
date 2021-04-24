from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from django.http import JsonResponse

import datetime
import json
import pandas as pd

import base64

import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

@csrf_exempt
def index(request):
    return render(request, "index.html")


@csrf_exempt
def predict(request):
    """View to predict output for selected prediction model

    Args:
        request (json): prediction model input (and parameters)

    Returns:
        json: prediction output
    """
    projects = [{"name":"Erschließung Ob den Häusern Stadt Tengen", "id":101227},
                    {"name":"Stadtbauamt Bräunlingen Feldweg", "id":101205}] 
    
    
    if request.method == "GET":
        
        context = {"projects": projects}
        return render(request, "app/predict.html", context)

    elif request.method == "POST":
        
        
        

        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        context = {"projects": projects,
                   "image": image_data}

        return render(request, 'app/predict/index.html', context)