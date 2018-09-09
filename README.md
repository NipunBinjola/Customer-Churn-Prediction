# Customer Churn Prediction and Reason for leaving Prediction using Machine Learning

We have built a sample prototype to demonstrate how we will develop real industry level solutions. This prototype  helps to identify  about-to-withdraw customers  and act accordingly to ensure that the bank can take the best-possible course of actions. Also the prototype predicts the possible reasons of leaving for a customer which may give a better picture of customer thoughts.

### Using this prototype live !
This prototype to predict customer churn is live at http://predictkaro.herokuapp.com/ !!
We have deployed whole machine-learning model on web for ease-of-access . Simply go to this url and login using the below 
username and password . 

Username - ```dhfl``` <br>
Password - ```dhfl```

## How to use the web-app - a walkthrough

Visit the link mentioned above to launch the fully functional web-application.
On clicking the link, a page like this will appear:

![capture](https://user-images.githubusercontent.com/31181537/45265262-bb304700-b465-11e8-8398-c8e67bb135e9.JPG)

Enter the login credentials provided above, **note** that entering wrong credentials will prevent user-login!

Now , user will have the choice of entering request data.

![home](https://user-images.githubusercontent.com/31181537/45265317-8e306400-b466-11e8-96e5-8f645c54baca.JPG)

Through Default File , we have provided a sample testing file- ```testtestdefault1``` for the user , which is of this format:

![testestdefault1](https://user-images.githubusercontent.com/31181537/45266047-a7d7a880-b472-11e8-8f4d-78682f200f6c.JPG)


As shown , the file requires these entries , alongwith an empty column of *Exited* which represents exiting probability from 0-1.


Now, on clicking **Predict** , a dropdown menu consisting of all employee-ids is available with a go-back option:

![selectdefault](https://user-images.githubusercontent.com/31181537/45265453-a3a68d80-b468-11e8-913f-21d79a2ae2d5.JPG)

Select any customer from drop-down menu and click **Predict** -

![output default](https://user-images.githubusercontent.com/31181537/45265733-8627f280-b46d-11e8-878e-6bb7856a89b6.jpeg)

a data-visualised graph showing churn-risk (low-medium-high) , alongwith top reasons for leaving and details of customers is shown.

Also through the feature of **Upload test data** , one can choose his/her own test requirements for prediction and analysis.
We have provided a sample ```uploadtest.csv``` for the user.

![uploadtest](https://user-images.githubusercontent.com/31181537/45266110-6b587c80-b473-11e8-9538-623c9e054116.JPG)


As usual , you can select any customer for whom you want to calculate the risk.
**Keep in mind**- The user testdata file must be of **same format** as that of the sample shown.

![test1](https://user-images.githubusercontent.com/31181537/45265532-1106ee00-b46a-11e8-8cbd-f59b0a3f2b0d.JPG)

It shows the prediction further:

![output](https://user-images.githubusercontent.com/31181537/45265736-98a22c00-b46d-11e8-876f-d2e1b972635d.jpeg)

When done , customer can **Logout** anytime .


Our web-app uses pre-loaded ```.h5``` models for prediction . These ```.h5``` models are built on our local machines using Python.
The models are based on *Keras and Neural networks* . So we need Keras and other Machine Learning libraries on our web environment to 
load this pre-defined models.
In our live web-app we have already made the environment.If you want to know how our models are trained on local machines and you want to see the working & code of models, check the code of ```howmodel1isbuilt.py``` and ```howmodel2isbuilt.py```


## How to run locally 

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Pre-requisites

* Creating a virtual environment using [Anaconda](https://www.anaconda.com/download/). If you need to create your workflows in Python and keep the dependencies separated out or   share the environment settings, Anaconda distributions are a great option.

The following main Python-based libraries have been used :

* ```Flask```
* ```Numpy```
* ```Pandas```
* ```Tensorflow```
* ```Keras```

which will be installed during setting the environment.

Provided the requirements are already installed in your system , you can simply execute the .py script named ```flask_app.py```
However, for future deployment purposes it is essential to create a virtual environment.

#### Step 1 : Main Directory
Go to command prompt/terminal (for Linux users) and change to the project as the default directory.
Example: If current path is ```C:\Users\name>```
change to ```C:\Users\name>cd Desktop\flasksite```  (if downloaded to Desktop)

#### Step 2 : Create environment
Create a conda environment using ```conda create -n env```
This will create an empty Python environment with name ```env``` . 
Activate using ```activate env```
To exit from the virtual environment , simply execute ```deactivate env ```
For further information , you can refer [here](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)

#### Step 3 : Install required libraries
Anaconda distribution comes with many default downloaded libraries which the user can directly , however to efficiently manage the
environment , manually install libraries.
A simple way would be to use ```pip``` - Python default package manager
```pip install [nameOfPackage]```
Install the libraries specified in pre-requisites
Example:
* ```pip install flask```
* ```pip install numpy```

For more information about ```pip``` you can refer [documentation](https://docs.python.org/3/installing/index.html)

#### Step 4 : Execute script
When all python dependencies are installed , simply execute the .py script named ```flask_app.py```  
```python flask_app.py ``` in the cmd prompt should do the task.
Follow the link generated, and the flask code should be up and running !

## Creators
* [Ashish Bhawnani](https://github.com/ashishgeeky)
* [Nipun Binjola](https://github.com/NipunBinjola)
* [Akash Rana](https://github.com/akash29rana)
* [Rahul Chauhan](https://github.com/RanaRauff)
