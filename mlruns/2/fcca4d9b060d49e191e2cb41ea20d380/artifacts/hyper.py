import mlflow 
import mlflow.sklearn
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier

df = load_breast_cancer()
X = pd.DataFrame(df.data, columns=df.feature_names)
y = pd.Series(df.target, name = 'target')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'max_depth' : [None,10,20,30],
    'n_estimators' : [10,20,30]
}

grid = GridSearchCV(
    estimator=rf,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2
)

# grid.fit(X_train,y_train)

# best_param = grid.best_params_
# best_score = grid.best_score_

# print(best_param, best_score)

mlflow.set_experiment("breast_cancer")

with mlflow.start_run():
    grid.fit(X_train,y_train)

    best_param = grid.best_params_
    best_score = grid.best_score_

    mlflow.log_params(best_param)
    mlflow.log_metric("accuracy",best_score)

    #log train data
    train_df = X_train.copy()
    train_df['target'] = y_train

    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df,"training data")

    #log test data 
    test_df = X_test.copy()
    test_df['target'] = y_test 

    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df,"testing data ")

    #log file 
    mlflow.log_artifact(__file__)

    #log model 
    mlflow.sklearn.log_model(grid.best_estimator_,"random Farest")

    #set tags 
    mlflow.set_tag("author","ashish bhai")

    print(best_param, best_score)