from flask import Flask, flash , redirect, render_template , request, session, abort , Markup
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from keras import backend as K
from werkzeug import secure_filename
import json
import csv

app = Flask(__name__)
app.secret_key = os.urandom(12)

dropdown_list = []
dropdown_list_2 = []

#saving filename of upload file
def sav_name(n):
    with open("filename.txt", "w") as ff:
        ff.write(n)
#preprocessing data of uploaded file        
def preprocess_data():
    dataset = pd.read_csv('Churn_Modelling.csv')
    ffr = open("filename.txt", "r")
    upl_file = ffr.read()
    upl_file = str(upl_file)
    test = pd.read_csv(upl_file)

    
    X_test = test.iloc[:, 3:13].values
    X = dataset.iloc[:, 3:13].values
    y= dataset.iloc[:, 13].values
    y_test= test.iloc[:, 13].values

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

    onehotencoder = OneHotEncoder(categorical_features = [1])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:, 1:]

    labelencoder_X_3 = LabelEncoder()#creating label encoder object no. 1 to encode region name(index 1 in features)
    X_test[:, 1] = labelencoder_X_3.fit_transform(X_test[:, 1])#encoding region from string to just 3 no.s 0,1,2 respectively
    labelencoder_X_4 = LabelEncoder()
    X_test[:, 2] = labelencoder_X_4.fit_transform(X_test[:, 2])#encoding Gender from string to just 2 no.s 0,1(male,female) respectively

    onehotencoder2 = OneHotEncoder(categorical_features = [1])
    X_test= onehotencoder2.fit_transform(X_test).toarray()
    X_test = X_test[:, 1:]


    from sklearn.model_selection import train_test_split
    X_train=X
    y_train=y


    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_test
#preprocessing data of default file
def preprocess_data_default():
    dataset = pd.read_csv('Churn_Modelling.csv')
    fpath = os.path.join("default", "testtestdefault1.csv")
    test = pd.read_csv(fpath)
    X_test = test.iloc[:, 3:13].values
    X = dataset.iloc[:, 3:13].values
    y= dataset.iloc[:, 13].values
    y_test= test.iloc[:, 13].values

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

    onehotencoder = OneHotEncoder(categorical_features = [1])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:, 1:]

    labelencoder_X_3 = LabelEncoder()#creating label encoder object no. 1 to encode region name(index 1 in features)
    X_test[:, 1] = labelencoder_X_3.fit_transform(X_test[:, 1])#encoding region from string to just 3 no.s 0,1,2 respectively
    labelencoder_X_4 = LabelEncoder()
    X_test[:, 2] = labelencoder_X_4.fit_transform(X_test[:, 2])#encoding Gender from string to just 2 no.s 0,1(male,female) respectively

    onehotencoder2 = OneHotEncoder(categorical_features = [1])
    X_test= onehotencoder2.fit_transform(X_test).toarray()
    X_test = X_test[:, 1:]


    from sklearn.model_selection import train_test_split
    X_train=X
    y_train=y


    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_test
# predicting reason for leaving percentage of specific member of uploaded file   
def model_2(cid1):
    dataset = pd.read_csv('Churn_Modelling.csv')
    data_re=dataset[dataset['Exited']==1]
    data_re.set_index('RowNumber',inplace=True)
    data_re.to_csv('data_re.csv')
    X = dataset.iloc[:, 3:14].values
    test=pd.read_csv('testtest1.csv')
    cid1 = int(cid1)

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

    X_train=X

    X_test=test.loc[test['CustomerId']==cid1].values.copy()
    X_test=X_test[:, 3:14]
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_3 = LabelEncoder()#creating label encoder object no. 1 to encode region name(index 1 in features)
    X_test[:,1] = labelencoder_X_3.fit_transform(X_test[:, 1])#encoding region from string to just 3 no.s 0,1,2 respectively
    labelencoder_X_4 = LabelEncoder()
    X_test[:,2] = labelencoder_X_4.fit_transform(X_test[:, 2])      
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_test
# predicting reason for leaving percentage of specific member of default file   
def model_default_2(cid1):
    dataset = pd.read_csv('Churn_Modelling.csv')
    data_re=dataset[dataset['Exited']==1]
    data_re.set_index('RowNumber',inplace=True)
    data_re.to_csv('data_re.csv')
    X = dataset.iloc[:, 3:14].values
    fpathr = os.path.join("default", "testtestreason1.csv")
    test = pd.read_csv(fpathr)
    cid1 = int(cid1)

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

    X_train=X

    X_test=test.loc[test['CustomerId']==cid1].values.copy()
    X_test=X_test[:, 3:14]
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_3 = LabelEncoder()#creating label encoder object no. 1 to encode region name(index 1 in features)
    X_test[:,1] = labelencoder_X_3.fit_transform(X_test[:, 1])#encoding region from string to just 3 no.s 0,1,2 respectively
    labelencoder_X_4 = LabelEncoder()
    X_test[:,2] = labelencoder_X_4.fit_transform(X_test[:, 2])      
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_test
#search for specific id from uploaded file
def search(cid):
    with open('testtest1.csv') as file:
        allRead = csv.reader(file, delimiter=',')
        for row in allRead:
            if row[1]==cid:
                return row
#search for specific id from default file
def search_default(cid):
    fpathr = os.path.join("default", "testtestreason1.csv")
    with open(fpathr) as file:
        allRead = csv.reader(file, delimiter=',')
        for row in allRead:
            if row[1]==cid:
                return row

#login check page
@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return render_template('upload.html')

#uploading page
@app.route('/upload')
def upload_file():
   dropdown_list.clear()
   dropdown_list_2.clear()
   return render_template('upload.html')

#getting predicted data in dropdown list of uploaded file 
@app.route('/uploader', methods = ['GET', 'POST'])
def uploader_file():
   if request.method == 'POST':
      K.clear_session()
      dropdown_list.clear()
      f = request.files['file']
      f.save(secure_filename(f.filename))
      sav_name(f.filename)
      ffr = open("filename.txt", "r")
      upl_file = ffr.read()
      upl_file = str(upl_file)
      test = preprocess_data()
      model = load_model('my_model.h5')
      model._make_predict_function()
      y_pred = model.predict(test)
      dff = pd.read_csv(upl_file)
      dff['Exited'] = y_pred
      dff.set_index('RowNumber', inplace=True)
      dff.sort_values('Exited', ascending=False, inplace=True)
      dff.to_csv('testtest1.csv')  # output file
      with open(upl_file) as file:
            allRead = csv.reader(file, delimiter=',')
            lineCount = 0
            for row in allRead:
                if lineCount==0:
                    lineCount=lineCount+1
                else:
                    lineCount=lineCount+1
                    dropdown_list.append((row[1]))
              
      return render_template('mytemplate3.html',  dropdown_list=dropdown_list)


 #getting only top % inserted from uploaded file 
@app.route('/input_percent' , methods = ['GET','POST'])
def input_num():
    x = request.form["in"]
    line = pd.read_csv("testtest1.csv").shape[0]
    y = round((float(x)*line)/100)
    ls = []
    lschurn=[]
    with open("testtest1.csv") as file:
        allRead = csv.reader(file, delimiter=',')
        lineCount = 0
        for row in allRead:
            if lineCount == 0:
                lineCount += 1
            elif lineCount <= y and lineCount != 0:
                ls.append(row[1])
                lschurn.append(row[13])
                lineCount += 1
    lss=list(map(lambda x: float(x*100),list(pd.read_csv("testtest1.csv")['Exited'][:y].copy())))  
    return render_template('mytemplate3_percent.html', outList = ls, value_list=lschurn,values_res=lss )


 #getting only top % inserted from default file
@app.route('/input_percent_default' , methods = ['GET','POST'])
def input_num_default():
    x = request.form["in"]
    fpathr = os.path.join("default", "testtestreason1.csv")
    line = pd.read_csv(fpathr).shape[0]
    y = round((float(x)*line)/100)
    ls = []
    lschurn=[]
    with open(fpathr) as file:
        allRead = csv.reader(file, delimiter=',')
        lineCount = 0
        for row in allRead:
            if lineCount == 0:
                lineCount += 1
            elif lineCount <= y and lineCount != 0:
                ls.append(row[1])
                lschurn.append(row[13])
                lineCount += 1
    lss=list(map(lambda x: float(x*100),list(pd.read_csv(fpathr)['Exited'][:y].copy())))              
    return render_template('mytemplate4_percent.html', outList = ls , value_list=lschurn , values_res=lss)

 #getting only top 2 id from uploaded file
@app.route('/uploader/data/2', methods=['GET', 'POST'])
def check_number():
    x = 2
    ls = []
    lschurn=[]

    with open("testtest1.csv") as file:
        allRead = csv.reader(file, delimiter=',')
        lineCount = 0
        for row in allRead:
            if lineCount == 0:
                lineCount += 1
            elif lineCount <= x and lineCount != 0:
                ls.append(row[1])
                lschurn.append(row[13])
                lineCount += 1
    lss=list(map(lambda x: float(x*100),list(pd.read_csv("testtest1.csv")['Exited'][:2].copy())))            
    return render_template('mytemplate3_percent.html', outList = ls, value_list=lschurn , values_res=lss)


 #getting only top 4 id from uploaded file
@app.route('/uploader/data/4', methods=['GET', 'POST'])
def check_number2():
    x = 4
    ls = []
    lschurn=[]

    with open("testtest1.csv") as file:
        allRead = csv.reader(file, delimiter=',')
        lineCount = 0
        for row in allRead:
            if lineCount==0:
                lineCount+=1
            elif lineCount <= x and lineCount != 0:
                ls.append(row[1])
                lschurn.append(row[13])
                lineCount += 1

    lss=list(map(lambda x: float(x*100),list(pd.read_csv("testtest1.csv")['Exited'][:4].copy()))) 
    return render_template('mytemplate3_percent.html', outList = ls, value_list=lschurn , values_res=lss)

 #getting predicted data in dropdown list of default file 
@app.route('/defaultfile', methods = ['GET', 'POST'])
def uploader_default_file():
      K.clear_session()
      dropdown_list_2.clear()
      test = preprocess_data_default()
      model = load_model('my_model.h5')
      model._make_predict_function()
      y_pred = model.predict(test)
      fpath = os.path.join("default", "testtestdefault1.csv")
      dff = pd.read_csv(fpath)
      dff['Exited'] = y_pred
      dff.set_index('RowNumber', inplace=True)
      dff.sort_values('Exited', ascending=False, inplace=True)
      fpathr = os.path.join("default", "testtestreason1.csv")
      dff.to_csv(fpathr)  # output file
      with open(fpath) as file:
            allRead = csv.reader(file, delimiter=',')
            lineCount = 0
            for row in allRead:
                if lineCount==0:
                    lineCount=lineCount+1
                else:
                    lineCount=lineCount+1
                    dropdown_list_2.append((row[1]))
                
      return render_template('mytemplate4.html',  dropdown_list_2=dropdown_list_2)


 #getting only top 2 id from default file
@app.route('/uploader/data_default/2', methods=['GET', 'POST'])
def check_number_default():
    x = 2
    ls = []
    lschurn=[]
    fpathr = os.path.join("default", "testtestreason1.csv")
    with open(fpathr) as file:
        allRead = csv.reader(file, delimiter=',')
        lineCount = 0
        for row in allRead:
            if lineCount == 0:
                lineCount += 1
            elif lineCount <= x and lineCount != 0:
                ls.append(row[1])
                lschurn.append(row[13])
                lineCount += 1
    lss=list(map(lambda x: float(x*100),list(pd.read_csv(fpathr)['Exited'][:2].copy())))               

    return render_template('mytemplate4_percent.html', outList = ls, value_list=lschurn , values_res=lss)


 #getting only top 4 id from default file
@app.route('/uploader/data_default/4', methods=['GET', 'POST'])
def check_number_default2():
    x = 4
    ls = []
    lschurn=[]
    fpathr = os.path.join("default", "testtestreason1.csv")
    with open(fpathr) as file:
        allRead = csv.reader(file, delimiter=',')
        lineCount = 0
        for row in allRead:
            if lineCount==0:
                lineCount+=1
            elif lineCount <= x and lineCount != 0:
                ls.append(row[1])
                lschurn.append(row[13])
                lineCount += 1
    lss=list(map(lambda x: float(x*100),list(pd.read_csv(fpathr)['Exited'][:4].copy())))              
    return render_template('mytemplate4_percent.html', outList = ls, value_list=lschurn , values_res=lss)

#displaying final full data predicted of selected customer from uploaded file
@app.route('/check/<string:dropdown>',methods=['POST','GET'])
def specific(dropdown):
    x = dropdown
    yy,yo = predict(x)
    x = search(x)
    rownum  = x[0]
    ccid = x[1]
    surname = x[2]
    creditscore  = x[3]
    geo = x[4]
    gender  = x[5]
    age  = x[6]
    tenure = x[7]
    balance  = x[8]
    numpro = x[9]
    hascard = x[10]
    activemem = x[11]
    salary = x[12]    
    x = x[13]
    pred= float(x)*100
    labels = ["probability",""]
    values = [pred]
    labels_res = ["Excess Documents Required","High Service Charges/Rate of Interest","Inexperienced Staff / Bad customer service","Long Response Times"]
    values_res = [float(i)*100 for i in yo[0]]
    x = float(x)*100
    x = round(x,2)
    values_res[0] = round(values_res[0],2)
    values_res[1] = round(values_res[1],2)
    values_res[2] = round(values_res[2],2)
    values_res[3] = round(values_res[3],2)
    colors = [ "#F7464A", "#46BFBD", "#FDB45C",  "#ABCDEF"]
    return render_template('chart_meter.html', set=zip(values_res, labels_res, colors),firstname=x, rownum=rownum, ccid=ccid, surname=surname, creditscore=creditscore, geo=geo, gender=gender, age=age, tenure=tenure, balance=balance, numpro=numpro, hascard = hascard, activemem = activemem, salary = salary, secondname = values_res[0] , secondname1 = values_res[1] , secondname2 = values_res[2] , secondname3 = values_res[3] ,labels_res=labels_res,values_res=values_res, values=values, labels=labels)

#displaying final full data predicted of selected customer from default file
@app.route('/check_default/<string:dropdown_2>',methods=['POST','GET'])
def specific_default(dropdown_2):
    x = dropdown_2
    yy,yo = predict_default(x)
    x = search_default(x)
    rownum  = x[0]
    ccid = x[1]
    surname = x[2]
    creditscore  = x[3]
    geo = x[4]
    gender  = x[5]
    age  = x[6]
    tenure = x[7]
    balance  = x[8]
    numpro = x[9]
    hascard = x[10]
    activemem = x[11]
    salary = x[12]
    x = x[13]
    pred= float(x)*100
    labels = ["probability",""]
    values = [pred]
    labels_res = ["Excess Documents Required","High Service Charges/Rate of Interest","Inexperienced Staff / Bad customer service","Long Response Times"]
    values_res = [float(i)*100 for i in yo[0]]
    x = float(x)*100
    x = round(x,2)
    values_res[0] = round(values_res[0],2)
    values_res[1] = round(values_res[1],2)
    values_res[2] = round(values_res[2],2)
    values_res[3] = round(values_res[3],2)
    colors = [ "#F7464A", "#46BFBD", "#FDB45C" , "#ABCDEF"]
    return render_template('chart_meter.html', set=zip(values_res, labels_res, colors),firstname=x, rownum=rownum, ccid=ccid, surname=surname, creditscore=creditscore, geo=geo, gender=gender, age=age, tenure=tenure, balance=balance, numpro=numpro, hascard = hascard, activemem = activemem, salary = salary, secondname = values_res[0] , secondname1 = values_res[1] , secondname2 = values_res[2] , secondname3 = values_res[3] ,labels_res=labels_res,values_res=values_res, values=values, labels=labels)

#login page
@app.route('/login', methods=['GET', 'POST'])
def do_admin_login():
    error = None
    if request.form['username'] != 'dhfl' or request.form['password'] != 'dhfl':
        error = 'Invalid username or password. Please try again!'
    else:
        flash('You were successfully logged in')
        session['logged_in'] = True
        return home()

    return render_template('login.html', error=error)

#logout page
@app.route("/logout")
def logout():
    session['logged_in'] = False
    session.clear()
    ffr = open("filename.txt", "r")
    upl_file = ffr.read()
    upl_file = str(upl_file)
    if os.path.exists(upl_file):
        os.remove(upl_file)

    if os.path.exists('testtest1.csv'):
        os.remove('testtest1.csv')
    return home()

#return to uploading page
@app.route("/backtofile")
def backtofile():
    session['logged_in'] = True
    ffr = open("filename.txt", "r")
    upl_file = ffr.read()
    upl_file = str(upl_file)
    if os.path.exists(upl_file):
        os.remove(upl_file)

    if os.path.exists('testtest1.csv'):
        os.remove('testtest1.csv')
    return home()

#predicting churn risk % for uploaded file
@app.route("/predict", methods=["GET","POST"])
def predict(z):
     K.clear_session()
     test = preprocess_data()
     model = load_model('my_model.h5')
     model._make_predict_function()
     y_pred = model.predict(test)
     ffr = open("filename.txt", "r")
     upl_file = ffr.read()
     upl_file = str(upl_file)
     dff = pd.read_csv(upl_file)
     dff['Exited'] = y_pred
     dff.set_index('RowNumber', inplace=True)
     dff.sort_values('Exited', ascending=False, inplace=True)
     dff.to_csv('testtest1.csv')  # output file
     cid1 = z 
     test3 = model_2(cid1)
     model2 = load_model('my_model2.h5')
     model2._make_predict_function()
     y_pred2 = model2.predict(test3)
     y_pred = y_pred.tolist()
     resons=["Excess Documents Required","High Service Charges/Rate of Interest","Inexperienced Staff / Bad customer service","Long Response Times"]
     dic=dict()
     diff=list()
     for j in range(len(y_pred2)):
        dic.clear()
        for (label, p) in zip(resons, y_pred2[j]):
            dic[label]= p*100
        diff.append(dic.copy())
     j = json.dumps(diff)
     K.clear_session()
     
     return j,y_pred2


#predicting churn risk % for default file
@app.route("/predict_default", methods=["GET","POST"])
def predict_default(z):
     K.clear_session()
     test = preprocess_data_default()
     model = load_model('my_model.h5')
     model._make_predict_function()
     y_pred = model.predict(test)
     fpath = os.path.join("default", "testtestdefault1.csv")
     dff = pd.read_csv(fpath)
     dff['Exited'] = y_pred
     dff.set_index('RowNumber', inplace=True)
     dff.sort_values('Exited', ascending=False, inplace=True)
     fpathr = os.path.join("default", "testtestreason1.csv")
     dff.to_csv(fpathr)  # output file
     cid1 = z 
     test3 = model_default_2(cid1)
     model2 = load_model('my_model2.h5')
     model2._make_predict_function()
     y_pred2 = model2.predict(test3)
     y_pred = y_pred.tolist()
     resons=["Excess Documents Required","High Service Charges/Rate of Interest","Inexperienced Staff / Bad customer service","Long Response Times"]
     dic=dict()
     diff=list()
     for j in range(len(y_pred2)):
        dic.clear()
        for (label, p) in zip(resons, y_pred2[j]):
            dic[label]= p*100
        diff.append(dic.copy())
     j = json.dumps(diff)
     K.clear_session()
     
     return j,y_pred2

    
if __name__ == "__main__":
    app.run()
    
