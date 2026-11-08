import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from .data_cleaning import FEATURES,NUMERIC_FEATURES,CATEGORICAL_FEATURES,TARGET
def train_model(data):
    X,y=data[FEATURES],(data[TARGET]=="yes").astype(int)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    numeric=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
    categorical=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])
    prep=ColumnTransformer([("numeric",numeric,NUMERIC_FEATURES),("categorical",categorical,CATEGORICAL_FEATURES)])
    model=Pipeline([("preprocessor",prep),("classifier",LogisticRegression(max_iter=1500,class_weight="balanced",random_state=42))])
    model.fit(X_train,y_train); pred=model.predict(X_test); proba=model.predict_proba(X_test)[:,1]
    metrics={"accuracy":accuracy_score(y_test,pred),"precision":precision_score(y_test,pred,zero_division=0),"recall":recall_score(y_test,pred,zero_division=0),"f1":f1_score(y_test,pred,zero_division=0),"roc_auc":roc_auc_score(y_test,proba)}
    return model,X_test,y_test,metrics
def predict_subscription(model,customer):
    probability=float(model.predict_proba(pd.DataFrame([customer]))[0,1])
    return ("yes" if probability>=.5 else "no"),probability
