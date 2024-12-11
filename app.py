import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, flash
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from statsmodels.tsa.stattools import adfuller  # library for finding d
from statsmodels.tsa.arima_model import ARIMA  # library for ARIMA model
from flask import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'C:/weather/Dataset'
app.config['SECRET_KEY'] = 'b0b4fbefdc48be27a6123605f02b6b86'

global df, X_train, X_test, y_train, y_test


def preprocess():
    df = pd.read_csv("C:/weather/weather_history.csv")

    df['Date'] = df["Formatted Date"].str[:10]
    df['Summary'] = df['Summary'].replace({'Partly Cloudy': 'Cloudly',
                                           'Mostly Cloudy': 'Cloudly',
                                           'Overcast': 'Overcast',
                                           'Clear': 'Sunny',
                                           'Foggy': 'Foggy',
                                           'Breezy and Overcast': 'Overcast',
                                           'Breezy and Mostly Cloudy': 'Overcast',
                                           'Breezy and Partly Cloudy': 'Cloudy',
                                           'Dry and Partly Cloudy': 'Cloudy',
                                           'Windy and Partly Cloudy': 'Cloudy',
                                           'Light Rain': 'Rain',
                                           'Breezy': 'Rain',
                                           'Windy and Overcast': 'Overcast',
                                           'Humid and Mostly Cloudy': 'Cloudy',
                                           'Drizzle': 'Rain',
                                           'Windy and Mostly Cloudy': 'Cloudy',
                                           'Breezy and Foggy': 'Foggy',
                                           'Dry': 'Sunny',
                                           'Humid and Partly Cloudy': 'Cloudy',
                                           'Dry and Mostly Cloudy': 'Cloudy',
                                           'Rain': 'Rain',
                                           'Windy': 'Rain',
                                           'Humid and Overcast': 'Overcast',
                                           'Windy and Foggy': 'Foggy',
                                           'Breezy and Dry': 'Rain',
                                           'Windy and Dry': 'Rain',
                                           'Dangerously Windy and Partly Cloudy': 'Cloudy'})

    # There is only one unique value so no need of Loud Cover
    df.drop(['Loud Cover'], axis=1, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])

    df.drop(['Formatted Date'], axis=1, inplace=True)
    df.set_index(['Date'], inplace=True)

    df['Precip Type'].fillna(method='ffill', inplace=True)

    # As the correlation between Temperature (C) and Apparent Temperature (C) is almost 1 so we are dropping Apparent Temperature (C)
    df.drop(['Apparent Temperature (C)'], axis=1, inplace=True)

    df.select_dtypes(exclude='O')
    mm = MinMaxScaler()
    df['Temperature (C)'] = mm.fit_transform(pd.DataFrame(df['Temperature (C)']))
    df['Humidity'] = mm.fit_transform(pd.DataFrame(df['Humidity']))
    df['Wind Speed (km/h)'] = mm.fit_transform(pd.DataFrame(df['Wind Speed (km/h)']))
    df['Wind Bearing (degrees)'] = mm.fit_transform(pd.DataFrame(df['Wind Bearing (degrees)']))
    df['Visibility (km)'] = mm.fit_transform(pd.DataFrame(df['Visibility (km)']))
    df['Pressure (millibars)'] = mm.fit_transform(pd.DataFrame(df['Pressure (millibars)']))

    le = LabelEncoder()
    df['Summary'] = le.fit_transform(pd.DataFrame(df['Summary']))
    df['Precip Type'] = le.fit_transform(pd.DataFrame(df['Precip Type']))
    df['Daily Summary'] = le.fit_transform(pd.DataFrame(df['Daily Summary']))

    cat_names = ['Cloudly', 'Overcast', 'Sunny', 'Foggy', 'Cloudy', 'Rain']
    cat_names = pd.DataFrame(cat_names, columns=['cat_name'])
    cat_names['Summary'] = le.fit_transform(cat_names['cat_name'])

    df['Summary'] = round(df['Summary'])

    df = df.groupby('Date').mean()
    df['Date'] = pd.to_datetime(df.index)
    df.set_index('Date', inplace=True)
    return df


def imputation(data):
    dummy = []
    r = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
    dummy = data.reindex(r).fillna(' ').rename_axis('Date').reset_index()
    dummy = dummy.replace(' ', np.nan)
    dummy = dummy.ffill()
    dummy.set_index('Date', inplace=True)
    return dummy


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/load', methods=["POST", "GET"])
def load():
    if request.method == "POST":
        file = request.files['weather_history']
        ext1 = os.path.splitext(file.filename)[1]
        if ext1.lower() == ".csv":
            try:
                shutil.rmtree(app.config['UPLOAD_FOLDER'])
            except:
                pass
            os.mkdir(app.config['UPLOAD_FOLDER'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'weather_history.csv'))
            flash('The data is loaded successfully', 'success')
            return render_template('load.html')
        else:
            flash('Please upload a csv type documents only', 'warning')
            return render_template('load.html')
    return render_template('load.html')


@app.route('/view', methods=['POST', 'GET'])
def view():
    df = preprocess()
    df = imputation(df)
    df['Summary'] = round(df['Summary'])
    X = df.drop(['Summary'], axis=1)
    y = df['Summary']
    if request.method == 'POST':
        filedata = request.form['df']
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67, random_state=42)
        if filedata == '0':
            flash(r"Please select an option", 'warning')
        elif filedata == '1':
            return render_template('view.html', col=X_train.columns.values, df=list(X_train.values.tolist()))
        else:
            return render_template('view.html', col=X_test.columns.values, df=list(X_test.values.tolist()))

            # return render_template('view.html')
        # temp_df = pd.read_csv('Dataset/weather_history.csv')
        # print(temp_df)
        # temp_df =load(os.path.join(app.config["UPLOAD_FOLDER"]))

    return render_template('view.html')


x_train = None;
y_train = None;
x_test = None;
y_test = None


@app.route('/training', methods=['GET', 'POST'])
def training():
    df = preprocess()
    df = imputation(df)
    df['Summary'] = round(df['Summary'])
    X = df.drop(['Summary'], axis=1)
    y = df['Summary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    if request.method == 'POST':
        model_no = int(request.form['algo'])

        if model_no == 0:
            flash(r"You have not selected any model", "info")

        elif model_no == 1:
            model = SVC()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc= accuracy_score(pred, y_test)
            print("============")
            print("============")
            print(classification_report(pred, y_test))

            msg1 = "THE ACCURACY OF SUPPORT VECTOR MACHINE IS: "+ str(acc)
            return render_template('training.html', mag1=msg1)



        elif model_no == 2:
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(pred, y_test)
            print("============")
            print("============")
            print(classification_report(pred, y_test))

            param_dict = {
                'criterion': ['entropy'],
                'max_depth': [1, 2, 3, 4, 8, 9, 10, 15, 20]
            }
            grid = GridSearchCV(model, param_grid=param_dict, cv=10, n_jobs=-1)

            grid.fit(X_train, y_train)
            print(grid.best_params_)
            print(grid.best_estimator_)
            bs = grid.best_score_

            msg1 = "THE ACCURACY OF DECISION TREE CLASSIFIER IS: " + str(bs)
            return render_template('training.html', mag1=msg1)




        elif model_no == 3:
            model = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=0, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(pred, y_test)
            print("============")
            print("============")
            print(classification_report(pred, y_test))

            param_dict = {'bootstrap': [True, False],
                          'max_depth': [10, 20, 30, 40, 50, 70, 100],
                          'max_features': ['auto', 'sqrt'],
                          'min_samples_leaf': [1, 2, 4, 6],
                          'min_samples_split': [2, 5, 10],
                          'n_estimators': [5, 10, 15, 20, 50, 100,150]}

            grid = RandomizedSearchCV(estimator=model, param_distributions=param_dict, n_iter=50, cv=10, verbose=2,
                                      random_state=42, n_jobs=-1)

            grid.fit(X_train, y_train)
            print(grid.best_params_)
            print(grid.best_estimator_)
            bs = grid.best_score_

            msg1 = "THE ACCURACY OF RANDOM FOREST CLASSIFIER IS: " + str(bs)
            return render_template('training.html', mag1=msg1)



        elif model_no == 4:
            model =LogisticRegression()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(pred, y_test)
            print("============")
            print("============")
            print(classification_report(pred, y_test))

            msg1 = "THE ACCURACY OF LOGISTIC REGRESSION IS: " + str(acc)
            return render_template('training.html', mag1=msg1)
    return render_template('training.html')



@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    df = preprocess()
    df = imputation(df)
    df['Summary'] = round(df['Summary'])
    X = df.drop(['Summary'], axis=1)
    y = df['Summary']
    if request.method == "POST":
        Precip_Type = request.form['Precip Type']
        print(Precip_Type)
        Temperature = request.form['Temperature (C)']
        print(Temperature)
        Humidity = request.form['Humidity']
        print(Humidity)
        Wind_Speed = request.form['Wind Speed (km/h)']
        print(Wind_Speed)
        Wind_Bearing = request.form['Wind Bearing (degrees)']
        print(Wind_Bearing)
        Visibility = request.form['Visibility (km)']
        print(Visibility)
        Pressure = request.form['Pressure (millibars)']
        print(Pressure)
        Daily_Summary = request.form['Daily Summary']
        print(Daily_Summary)

        di = {'Precip Type': [Precip_Type], 'Temperature (C)': [Temperature], 'Humidity': [Humidity],
              'Wind Speed (km/h)': [Wind_Speed],
              'Wind Bearing (degrees)': [Wind_Bearing], 'Visibility (km)': [Visibility],
              'Pressure (millibars)': [Pressure],
              'Daily Summary': [Daily_Summary]}

        test = pd.DataFrame.from_dict(di)
        print(test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        cfr = RandomForestClassifier()
        model = cfr.fit(X_train, y_train)
        output = model.predict(test)
        print(output)

        if output == 1:
            msg = 'The Weather is <span style = color:grey;> Cloudy </span></b>'

        elif output == 2:
            msg = 'The Weather is <span style = color:red;> Foggy </span></b>'

        elif output == 3:
            msg = 'The Weather is <span style = color:blue;> Overcast </span></b>'

        elif output == 4:
            msg = 'The Weather is <span style = color:red;> Rain </span></b>'

        else:
            msg = 'The Weather is <span style = color:dark yellow;>Sunny</span></b>'

        return render_template('prediction.html', mag=msg)
    return render_template('prediction.html')


@app.route('/Graph')
def Graph():

    return render_template('Graph.html')



if __name__ == '__main__':
    app.run(debug=True)
