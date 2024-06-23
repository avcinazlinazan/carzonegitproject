from django.shortcuts import render
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline


# for call home.html
def home(request):
    return render(request, 'home.html')

# for call predict.html
# def predict(request):
#     return render(request, 'predict.html')
# def predict(request): 
#     df = pd.read_csv(r"otomobil_AutoScout.csv")

#     df.drop_duplicates(keep='first', inplace=True)
#     df.drop(index=[2614], inplace =True)

#     X= df[['make_model','body_type','km','Fuel','age','hp_kW' , 'Gearing_Type','Displacement_cc','Weight_kg','cons_comb']]
    
#     y= df.price
#     X_train, X_test, y_train, y_test = train_test_split(X,
#                                                     y,
#                                                     test_size=0.2,
#                                                     random_state=101)




#     cat_features = X.select_dtypes("object").columns
#     ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
#     #ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

#     column_trans = make_column_transformer((ord_enc, cat_features), remainder='passthrough')

#     #operations = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestRegressor(random_state=101))]
#     operations = [("OrdinalEncoder", column_trans), ("LGBM_model", LGBMRegressor(random_state=101, objective='regression'))]

#     pipe_model = Pipeline(steps=operations)

#     pipe_model.fit(X_train, y_train)

#     operations = [("OrdinalEncoder", column_trans), ("LGBM_model", LGBMRegressor(random_state=101,
#                             objective='regression',
#                             colsample_bytree=0.8,
#                             learning_rate=0.1,
#                             max_depth=5,
#                             subsample=0.8,
#                             num_leaves = 32,
#                             n_estimators=500,
#                             verbose=-1))]

    
#     pipe_model = Pipeline(steps=operations)

#     pipe_model.fit(X_train, y_train)



#     make_model = (request.GET['make_model']) 
#     body_type = (request.GET['body_type'])
#     km = float(request.GET['km'])
#     Fuel = (request.GET['Fuel'])
#     age= float(request.GET['age']) 
#     hp_kW = float(request.GET['hp_kW']) 
#     Gearing_Type = (request.GET['Gearing_Type']) 
#     Displacement_cc = float(request.GET['Displacement_cc'])
#     Weight_kg = float(request.GET['Weight_kg']) 
#     cons_comb = float(request.GET['cons_comb'])


#     data = {
#         'make_model':[make_model],
#         'body_type':[body_type], 
#         'km':[km],
#         'Fuel':[Fuel],
#         'age':[age],
#         'hp_kW':[hp_kW], 
#         'Gearing_Type':[Gearing_Type], 
#         'Displacement_cc':[Displacement_cc], 
#         'Weight_kg':[Weight_kg], 
#         'cons_comb':[cons_comb]     
#     }


#     df = pd.DataFrame(data)

    
    

#     predictions = pipe_model.predict(df)



    
#     predictions =round(predictions[0] ,3)

#     price = "Result Car Prediction "+str(predictions) +"$"
#     return render(request, 'predict.html', {"result2":price})