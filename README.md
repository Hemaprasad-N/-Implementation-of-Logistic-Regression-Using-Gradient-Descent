# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression


2.Set variables for assigning dataset values. 


3.Import linear regression from sklearn. 


4.Predict the values of array.


5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn


6.Obtain the graph

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: HEMAPRASAD N
RegisterNumber:  212222040054
*/

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
x=data[:,[0,1]]
y=data[:,2]

x[:5]

y[:5]

plt.figure()
plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted",color="cadetblue")
plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not Admitted",color="plum")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
x_plot=np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot),color="cadetblue")
plt.show()

def costFunction(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

def cost(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j= -(np.dot(y, np.log(h)) + np.dot(1-y, np.log(1-h)))/x.shape[0]
  return j


def gradient(theta,x,y):

  h=sigmoid(np.dot(x,theta))
  grad=np.dot(x.T,h-y)/x.shape[0]
  return grad


x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,x,y):
  x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot=np.c_[xx.ravel(),yy.ravel()]
  x_plot=np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot=np.dot(x_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted",color="mediumpurple")
  plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not admitted",color="pink")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,x,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,x):
  x_train=np.hstack((np.ones((x.shape[0],1)),x))
  prob=sigmoid(np.dot(x_train,theta))
  return(prob>=0.5).astype(int)

np.mean(predict(res.x,x)==y)
```

## Output:
1.Array Value of x


![image](https://github.com/Hemaprasad-N/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135933397/62ddb39d-0b24-4570-ab01-edfda9891016)


2.Array Value of y


![image](https://github.com/Hemaprasad-N/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135933397/d6ea687b-9d1a-4b04-82a8-fb8faeca57ce)


3.Exam 1-Score Graph


![image](https://github.com/Hemaprasad-N/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135933397/df3bbf00-03b0-45d1-8f67-bec3814a9ff8)


4.Sigmoid function graph


![image](https://github.com/Hemaprasad-N/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135933397/05c94fe1-8154-4e6e-aa08-ed02d59063f3)


5.x_train_grad value


![image](https://github.com/Hemaprasad-N/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135933397/beb1fc13-59fb-4c52-b8bd-68eb08a1c9b6)


6.y_train_grad value


![image](https://github.com/Hemaprasad-N/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135933397/8e06f7bd-eb87-48ad-bf08-a47fafb4f4c8)


7.Print res.x


![image](https://github.com/Hemaprasad-N/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135933397/5d90e9a1-08e0-468d-96fd-3df55bb3e2ae)


8.Decision boundary-graph for exam score


![image](https://github.com/Hemaprasad-N/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135933397/09df9419-be95-40d6-bae6-0a63fe9c5366)


9.Probability value


![image](https://github.com/Hemaprasad-N/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135933397/8dfef1f1-68d8-4b36-bfd4-a115f29a9a32)


10.Prediction value of mean


![image](https://github.com/Hemaprasad-N/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135933397/04a6d0a5-84a7-4cd2-9c3b-3fe482157f29)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

