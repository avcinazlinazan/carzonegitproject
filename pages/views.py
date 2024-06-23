from django.shortcuts import render
from .models import Team
from cars.models import Car




############ Mdel için kütüphaneler ##################
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

########################################

# Create your views here.

def home(request):
    teams = Team.objects.all()
    featured_cars = Car.objects.order_by('-created_date').filter(is_featured=True)
    all_cars = Car.objects.order_by('-created_date')
    
    model_search= Car.objects.values_list('model', flat=True).distinct()
    city_search= Car.objects.values_list('city', flat=True).distinct()
    year_search = Car.objects.values_list('year', flat=True).distinct()
    body_style_search= Car.objects.values_list('body_style', flat=True).distinct()
    
    data = {
        'teams': teams,
        'featured_cars': featured_cars,
        'all_cars': all_cars, 
        
        'model_search':model_search,
        'city_search':city_search,
        'year_search':year_search,
        'body_style_search':body_style_search,
    }
    return render(request, 'pages/home.html',data)
# def predict(request):
#     return render(request, 'pages/predict.html')
def about(request):
    teams = Team.objects.all()
    
    data = {
        'teams': teams,
    }
    return render(request,'pages/about.html',data)

def services(request):
    return render(request,'pages/services.html')

def contact(request):
    return render(request,'pages/contact.html')

def predict(request): 
    df = pd.read_csv(r"otomobil_AutoScout.csv")

    df.drop_duplicates(keep='first', inplace=True)
    df.drop(index=[2614], inplace =True)

    X= df[['make_model','body_type','km','Fuel','age','hp_kW' , 'Gearing_Type','Displacement_cc','Weight_kg','cons_comb']]
    
    y= df.price
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=101)




    cat_features = X.select_dtypes("object").columns
    ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    #ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    column_trans = make_column_transformer((ord_enc, cat_features), remainder='passthrough')

    #operations = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestRegressor(random_state=101))]
    operations = [("OrdinalEncoder", column_trans), ("LGBM_model", LGBMRegressor(random_state=101, objective='regression'))]

    pipe_model = Pipeline(steps=operations)

    pipe_model.fit(X_train, y_train)

    operations = [("OrdinalEncoder", column_trans), ("LGBM_model", LGBMRegressor(random_state=101,
                            objective='regression',
                            colsample_bytree=0.8,
                            learning_rate=0.1,
                            max_depth=5,
                            subsample=0.8,
                            num_leaves = 32,
                            n_estimators=500,
                            verbose=-1))]

    
    pipe_model = Pipeline(steps=operations)

    pipe_model.fit(X_train, y_train)



    make_model = (request.GET['make_model']) 
    body_type = (request.GET['body_type'])
    km = float(request.GET['km'])
    Fuel = (request.GET['Fuel'])
    age= float(request.GET['age']) 
    hp_kW = float(request.GET['hp_kW']) 
    Gearing_Type = (request.GET['Gearing_Type']) 
    Displacement_cc = float(request.GET['Displacement_cc'])
    Weight_kg = float(request.GET['Weight_kg']) 
    cons_comb = float(request.GET['cons_comb'])

    # make_model = (request.POST['make_model']) 
    # body_type = (request.POST['body_type'])
    # km = float(request.POST['km'])
    # Fuel = (request.POST['Fuel'])
    # age= float(request.POST['age']) 
    # hp_kW = float(request.POST['hp_kW']) 
    # Gearing_Type = (request.POST['Gearing_Type']) 
    # Displacement_cc = float(request.POST['Displacement_cc'])
    # Weight_kg = float(request.POST['Weight_kg']) 
    # cons_comb = float(request.POST['cons_comb'])


    data = {
        'make_model':[make_model],
        'body_type':[body_type], 
        'km':[km],
        'Fuel':[Fuel],
        'age':[age],
        'hp_kW':[hp_kW], 
        'Gearing_Type':[Gearing_Type], 
        'Displacement_cc':[Displacement_cc], 
        'Weight_kg':[Weight_kg], 
        'cons_comb':[cons_comb]     
    }


    df = pd.DataFrame(data)

    
    

    predictions = pipe_model.predict(df)



    
    predictions =round(predictions[0] ,3)

    price = "Prediction "+str(predictions) +"$"
    return render(request, 'pages/services.html', {"result2":price})