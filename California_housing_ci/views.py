from anaconda_navigator.widgets.manager import model
from django.shortcuts import render

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

def home(request):
    return render(request, "home.html")

def predict(request):
    return render(request, "predict.html")

def result(request):
    # load dataset
    data = pd.read_csv(r'C:\Users\USER\Documents\housing.csv')

    # dropping records with null values
    data.dropna(inplace=True)

    if 'ocean_proximity' in data.columns:
        data.drop('ocean_proximity', axis=1, inplace=True)
    else:
        print("'ocean_proximity' column does not exist in the DataFrame")

    # splitting dataset as training and testing
    from sklearn.model_selection import train_test_split

    x = data.drop(['median_house_value'], axis=1)
    y = data['median_house_value']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    # Model Selection and Training
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)

    y_pred = rf_model.predict(x_test)

    # Function to preprocess input data, make predictions, and return the predicted house prices
    def predict_house_price(data):
        data = pd.DataFrame(data, index=[0])
        data[['total_bedrooms']] = data[['total_bedrooms']].apply(lambda x: np.log(x + 1))

        return rf_model.predict(data)

    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])
    var6 = float(request.GET['n6'])
    var7 = float(request.GET['n7'])
    var8 = float(request.GET['n8'])


    pred = rf_model.predict(np.array([var1, var2, var3, var4, var5, var6, var7, var8]).reshape(1,-1))
    pred = round(pred[0])

    price = "The predicted price is $"+str(pred)


    return render(request, "predict.html", {"result2":price})


