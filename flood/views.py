from django.shortcuts import render
import numpy as np
# from notebook import Linear_Regression
from joblib import load
model = load('./model.joblib')

def predict(request):
    if request.method == 'POST':
        rainfall_amount = request.POST['rainfall_value']
        river_amount = request.POST['river_value']
        flood_predict = model.predict(np.array([[float(rainfall_amount), float(river_amount)]]))
        
        if flood_predict[0] == 0:
            flood_predict = f'No Flood {flood_predict}'

        else:
            flood_predict = f'Flood {flood_predict}'
        return render(request,'index.html',{'output': flood_predict})
    return render(request,'index.html')