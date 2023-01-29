from django.shortcuts import render
import numpy as np
# from notebook import Linear_Regression
from joblib import load
model = load('./model.joblib')

def predict(request):
    if request.method == 'POST':
        number1 = request.POST['num1']
        number2 = request.POST['num2']
        y_pred = model.predict(np.array([[float(number1), float(number2)]]))
        
        if y_pred[0] == 0:
            y_pred = f'No Flood {y_pred}'

        else:
            y_pred = f'Flood {y_pred}'
        return render(request,'index.html',{'output': y_pred})
    return render(request,'index.html')