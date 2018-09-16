
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras

dataset = pd.read_csv('data_re.csv')
#test = pd.read_csv('testtest1.csv')
#data_re=dataset[dataset['Exited']==1]
#print(data_re)
#data_re.to_csv('data_re.csv')
#oth=dict("Reason for exiting company"=[])
#test=test[test['Exited']==1]
#test.set_index('RowNumber',inplace=True)
#test.insert(loc=13,column="Reason for exiting company",value=None,inplace=True)
#test.add(oth,axis="columns", fill_value=None)
# print(len(test.iloc[0]))
#test.to_csv('ttre.csv')
test=pd.read_csv('uploadtest.csv')

X = dataset.iloc[:, 3:14].values
X_test=test.iloc[:, 3:14].values

y= dataset.iloc[:, 14].values
y_test= test.iloc[:, 13].values

print(X)
print(y)
print("test here",X_test)
print(y_test)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#onehotencoder = OneHotEncoder(categorical_features = [1])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]
# labelencoder_X_re = LabelEncoder()#creating label encoder object no. 1 to encode region name(index 1 in features)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse=False)
# print("before",y)
y = y.reshape(len(y), 1)
# print("abefore",y)
y= onehot_encoder.fit_transform(y)
# print(y)
# pp
labelencoder_X_3 = LabelEncoder()#creating label encoder object no. 1 to encode region name(index 1 in features)
X_test[:, 1] = labelencoder_X_3.fit_transform(X_test[:, 1])#encoding region from string to just 3 no.s 0,1,2 respectively
labelencoder_X_4 = LabelEncoder()
X_test[:, 2] = labelencoder_X_4.fit_transform(X_test[:, 2])#encoding Gender from string to just 2 no.s 0,1(male,female) respectively

#onehotencoder2 = OneHotEncoder(categorical_features = [1])
#X_test= onehotencoder2.fit_transform(X_test).toarray()
#X_test = X_test[:, 1:]

#from sklearn.model_selection import train_test_split
X_train=X
y_train=y
print(X_test.shape)
print(X_train.shape)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from keras.models import load_model

from keras.models import Sequential#For building the Neural Network layer by layer
from keras.layers import Dense
#------2)Defining a Graph
classifier = Sequential() #UNCOMMENT if not running from saved model
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))#UNCOMMENT if not running from saved model

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))#UNCOMMENT if not running from saved model

# Adding the output layer
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'sigmoid'))#UNCOMMENT if not running from saved model


# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy')#UNCOMMENT if not running from saved model

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train,batch_size=2,nb_epoch=200)#UNCOMMENT if not running from saved model
classifier.save('my_model2.h5')  # creates a HDF5 file 'my_model.h5'#UNCOMMENT if not running from saved model
#classifier=load_model('my_model2.h5') #UNCOMMENT if running from the saved model
y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.5)
print(y_pred)
# dff=pd.read_csv("testtest.csv")
# dff['Exited']=y_pred
# dff.set_index('RowNumber',inplace=True)
# dff.sort_values('Exited',ascending=False,inplace=True)
# dff.to_csv('testtest1.csv') #output file

for j in range(len(y_pred)):
    print("=================================================================================================")
    for (label, p) in zip(label_encoder.classes_, y_pred[j]):
        print("the",label," %", p*100 )
        
