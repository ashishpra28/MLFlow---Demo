import mlflow
import dagshub
dagshub.init(repo_owner='aknexa', repo_name='MLFlow---Demo', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/aknexa/MLFlow---Demo.mlflow")

import mlflow.sklearn
from sklearn.datasets import load_wine 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 

wine = load_wine()
X = wine.data 
y = wine.target 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.10,random_state=42)

max_depth = 5
n_estimators = 5

mlflow.set_experiment("new-experiment")

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators
                                )
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test,y_pred)

    mlflow.log_metric("acc", acc),
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("n_estimators",n_estimators)

    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True)

    plt.savefig("matrix.png")

    mlflow.log_artifact("matrix.png")
    mlflow.log_artifact(__file__)
    
    #log tags 
    mlflow.set_tags({
        "author":"ashish",
        "project":"wine project"
    })

    #log model 
    mlflow.sklearn.log_model(rf, "random_forest_model")

    print(acc)