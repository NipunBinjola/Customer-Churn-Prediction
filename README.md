# Customer Churn Prediction and reason for leaving prediction using Machine Learning

We have built a sample prototype to demonstrate how we will develop real industry level solutions. This prototype  helps to identify  about-to-withdraw customers  and act accordingly to ensure that the bank can take the best-possible course of actions. Also the prototype predicts the possible reasons of leaving for a customer which may give a better picture of customer thoughts.

### Using this prototype live !
This prototype to predict customer churn is live at http://predictkaro.herokuapp.com/ !!
We have deployed whole machine-learning model on web for ease-of-access . Simply go to this url and login using the below 
username and password . 
Username - ```dhfl``` <br>
Password - ```dhfl```

## How to use the web-app - a walkthrough
Visit the link mentioned above to launch the fully functional web-application.



## Getting Started

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

### How to run locally
Provided the requirements are already installed in your system , you can simply execute the .py script named ```(nameOfMain)```
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
When all python dependencies are installed , simply execute the .py script named ```(nameOfMain)```  
```python (nameofFile) ``` in the cmd prompt should do the task.
Follow the link generated, and the flask code should be up and running !

## How to deploy as a web app:
A number of services are available for web-deployment using a Platform-as-a-Service (PAAS) . One such platorm

