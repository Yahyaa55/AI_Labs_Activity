# lab21 : prediction de credit de logement
# Realisation oar : yahya Emsi 2023/2024
# REf : code source  https://www.kaggle.com/code/rodsonzepekinio/pr-vision-d-un-cr-dit-logement/notebook

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## Step 1: Dataset
# Feature engineering
dt=pd.read_csv("datasets/train.csv")
print(dt.head(10))
print(dt.info())
print(dt.isna().sum())
# Data transformation
def trans(data):
    for c in data.columns:
        if data[c].dtype=='int64' or data[c].dtype=='float64':
            data[c].fillna(data[c].median(),inplace=True)
        else:
              data[c].fillna(data[c].mode()[0],inplace=True)
trans(dt)
print(dt.isna().sum())
# Target exploration (loan_status)
print(dt["Loan_Status"].value_counts(normalize=True)*100)
# Data vizualisation: px or sns
var_num=["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]
print(dt[var_num].describe())
dt['Credit_History'] = dt['Credit_History'].replace(1.0,'Yes')
dt['Credit_History'] = dt['Credit_History'].replace(0.0,'No')
#Les variables categoriques
var_cat=["Loan_Status","Gender","Married","Dependents","Education","Self_Employed","Property_Area","Credit_History"]
fig, axes = plt.subplots(4, 2, figsize=(12, 15))
dt_num=dt[var_num]
dt_cat=dt[var_cat]
# print(dt_cat)
dt_cat=pd.get_dummies(dt_cat,drop_first=True)
# print(dt_cat)
dt_encoded=pd.concat([dt_cat,dt_num],axis=1)
y=dt_encoded["Loan_Status_Y"]
x=dt_encoded.drop("Loan_Status_Y",axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
dt_encoded.to_csv("datasets/loan.csv")
## Step 2: Model
model=LogisticRegression()
## Step 3: Train
model.fit(x_train,y_train)
## Step 4: Test
print("Votre Intelligence Arti est fiable à")
print(model.score(x_test,y_test)*100),print("%")



