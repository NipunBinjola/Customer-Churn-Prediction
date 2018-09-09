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


def sav_name(n):
    with open("filename.txt", "w") as ff:
        ff.write(n)
        
def preprocess_data():
    dataset = pd.read_csv('Churn_Modelling.csv')
    #test = pd.read_csv('testtest.csv')
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

def preprocess_data_default():
    dataset = pd.read_csv('Churn_Modelling.csv')
    #test = pd.read_csv('testtest.csv')
    
    test = pd.read_csv('testtestdefault1.csv')

    
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
    
def model_2(cid1):
    dataset = pd.read_csv('Churn_Modelling.csv')
   # test1 = pd.read_csv('testtest.csv')
    #X_test = test1.iloc[:, 3:13].values
    data_re=dataset[dataset['Exited']==1]
    data_re.set_index('RowNumber',inplace=True)
    data_re.to_csv('data_re.csv')
    X = dataset.iloc[:, 3:14].values
    test=pd.read_csv('testtest1.csv')
    '''with open('ttre.csv') as file:
        allRead = csv.reader(file, delimiter=',')
        #lineCount = 0
        for row in allRead:
            if row[1]==cid1:
                X_test = row
    X_test = X_test[3:14]'''
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
    
def model_default_2(cid1):
    dataset = pd.read_csv('Churn_Modelling.csv')
   # test1 = pd.read_csv('testtest.csv')
    #X_test = test1.iloc[:, 3:13].values
    data_re=dataset[dataset['Exited']==1]
    data_re.set_index('RowNumber',inplace=True)
    data_re.to_csv('data_re.csv')
    X = dataset.iloc[:, 3:14].values
    test=pd.read_csv('testtestreason1.csv')
    '''with open('ttre.csv') as file:
        allRead = csv.reader(file, delimiter=',')
        #lineCount = 0
        for row in allRead:
            if row[1]==cid1:
                X_test = row
    X_test = X_test[3:14]'''
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

def search(cid):
    with open('testtest1.csv') as file:
        allRead = csv.reader(file, delimiter=',')
        #lineCount = 0
        for row in allRead:
            if row[1]==cid:
                return row
def search_default(cid):
    with open('testtestreason1.csv') as file:
        allRead = csv.reader(file, delimiter=',')
        #lineCount = 0
        for row in allRead:
            if row[1]==cid:
                return row

@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        #lastname = "<a href='/logout'>Logout</a>"
        #return "<a href='/logout'>Logout</a>"
        #return  '{} {}'.format(firstname, lastname)
        #return predict()
        return render_template('upload.html')
        #return render_template('mytemplate.html')
@app.route('/upload')
def upload_file():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      sav_name(f.filename)
      #return f.filename
      ffr = open("filename.txt", "r")
      upl_file = ffr.read()
      upl_file = str(upl_file)
      dropdown_list=[]
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
  
@app.route('/defaultfile', methods = ['GET', 'POST'])
def uploader_default_file():
      #return f.filename
      dropdown_list_2=[]
      with open('testtestdefault1.csv') as file:
            allRead = csv.reader(file, delimiter=',')
            lineCount = 0
            for row in allRead:
                if lineCount==0:
                    lineCount=lineCount+1
                else:
                    lineCount=lineCount+1
                    dropdown_list_2.append((row[1]))
                
      return render_template('mytemplate4.html',  dropdown_list_2=dropdown_list_2)

@app.route('/check/<string:dropdown>',methods=['POST','GET'])
def specific(dropdown):
    x = dropdown
    #x = request.form['cid']
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
    # labels = ["jan"]
    values = [pred]
    labels_res = ["Excess Documents Required","High Service Charges/Rate of Interest","Inexperienced Staff / Bad customer service","Long Response Times"]
    values_res = [float(i)*100 for i in yo[0]]
    x = float(x)*100
    x = round(x,2)
    values_res[0] = round(values_res[0],2)
    values_res[1] = round(values_res[1],2)
    values_res[2] = round(values_res[2],2)
    values_res[3] = round(values_res[3],2)
    #, "#46BFBD", "#FDB45C", "#FEDCBA","#ABCDEF", "#DDDDDD", "#ABCABC"  
    # colors = [ "#F7464A"  ]
    # return render_template('chart_meter.html', values=values, labels=labels)
    #j = json.dumps(x)
    return render_template('chart_meter.html', firstname=x, rownum=rownum, ccid=ccid, surname=surname, creditscore=creditscore, geo=geo, gender=gender, age=age, tenure=tenure, balance=balance, numpro=numpro, hascard = hascard, activemem = activemem, salary = salary, secondname = values_res[0] , secondname1 = values_res[1] , secondname2 = values_res[2] , secondname3 = values_res[3] ,labels_res=labels_res,values_res=values_res, values=values, labels=labels)



@app.route('/check_default/<string:dropdown_2>',methods=['POST','GET'])
def specific_default(dropdown_2):
    x = dropdown_2
    #x = request.form['cid']
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
    # labels = ["jan"]
    values = [pred]
    labels_res = ["Excess Documents Required","High Service Charges/Rate of Interest","Inexperienced Staff / Bad customer service","Long Response Times"]
    values_res = [float(i)*100 for i in yo[0]]
    x = float(x)*100
    x = round(x,2)
    values_res[0] = round(values_res[0],2)
    values_res[1] = round(values_res[1],2)
    values_res[2] = round(values_res[2],2)
    values_res[3] = round(values_res[3],2)
    #, "#46BFBD", "#FDB45C", "#FEDCBA","#ABCDEF", "#DDDDDD", "#ABCABC"  
    # colors = [ "#F7464A"  ]
    # return render_template('chart_meter.html', values=values, labels=labels)
    #j = json.dumps(x)
    return render_template('chart_meter.html', firstname=x, rownum=rownum, ccid=ccid, surname=surname, creditscore=creditscore, geo=geo, gender=gender, age=age, tenure=tenure, balance=balance, numpro=numpro, hascard = hascard, activemem = activemem, salary = salary, secondname = values_res[0] , secondname1 = values_res[1] , secondname2 = values_res[2] , secondname3 = values_res[3] ,labels_res=labels_res,values_res=values_res, values=values, labels=labels)


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
    #request.form['password'] = '1lakh'
    #request.form['username'] == 'DHF'
    return home()

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
    #request.form['password'] = '1lakh'
    #request.form['username'] == 'DHF'
    return home()

@app.route("/predict", methods=["GET","POST"])
def predict(z):
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
     dff.to_csv('testtest1.csv')  # output file
     cid1 = z 
    # y_pred = y_pred.tolist()
     #j = json.dumps(y_pred)
    

     #test2 = preprocess_data_2()
     #model2 = load_model('my_model.h5')
     #test = test.tolist()
     #model2._make_predict_function()
     #graph = tf.get_default_graph()
     #with graph.as_default():
     #y_pred2 = model.predict(test)
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
     #return j
     K.clear_session()
     
     return j,y_pred2

@app.route("/predict_default", methods=["GET","POST"])
def predict_default(z):
     test = preprocess_data_default()
     model = load_model('my_model.h5')
     model._make_predict_function()
     y_pred = model.predict(test)
     dff = pd.read_csv('testtestdefault1.csv')
     dff['Exited'] = y_pred
     dff.set_index('RowNumber', inplace=True)
     dff.to_csv('testtestreason1.csv')  # output file
     cid1 = z 
    # y_pred = y_pred.tolist()
     #j = json.dumps(y_pred)
    

     #test2 = preprocess_data_2()
     #model2 = load_model('my_model.h5')
     #test = test.tolist()
     #model2._make_predict_function()
     #graph = tf.get_default_graph()
     #with graph.as_default():
     #y_pred2 = model.predict(test)
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
     #return j
     K.clear_session()
     
     return j,y_pred2
'''@app.route("/graph" ,  methods=["GET","POST"])
def chart(pred):
    labels = ["probability",""]
    values = [pred,100-pred]
    #, "#46BFBD", "#FDB45C", "#FEDCBA","#ABCDEF", "#DDDDDD", "#ABCABC"	
    colors = [ "#F7464A","#FDB45C"  ]
    return render_template('chart.html', set=zip(values, labels, colors))'''
    
if __name__ == "__main__":
    app.run()
    
