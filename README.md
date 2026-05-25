# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the spam mail dataset and preprocess the data.
2.Convert the email text into numerical features using TF-IDF.
3.Train the Support Vector Machine (SVM) classifier.
4.Test the model and evaluate it using accuracy and confusion matrix.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: B.SASIREKHA
RegisterNumber: 212225040388 
*/


import pandas as pd

# Load only required columns
data = pd.read_csv(
    "spam.csv",
    encoding='Windows-1252',
    usecols=['v1','v2']
)

print("data")
print(data)

x = data['v2']
y = data['v1']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=0
)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
model = SVC()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print("\nconfusion matrix")
print(confusion_matrix(y_test,y_pred))

print("\naccuracy")
print(accuracy_score(y_test,y_pred))

print("\nclassification report")
print(classification_report(y_test,y_pred))
```

## Output:
<img width="756" height="692" alt="image" src="https://github.com/user-attachments/assets/541d151e-711b-49f7-9ba2-9cf48ca657f0" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
