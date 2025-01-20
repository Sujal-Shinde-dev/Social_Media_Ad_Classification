import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data=pd.read_csv("Social_Network_Ads.csv")
print(data.head())
print(data.shape)
print(data.isnull().sum())
print(data.drop_duplicates(inplace=True))
print(data['Gender'].unique())
print(data['Purchased'].unique())
data.drop(['User ID'],axis=1,inplace=True)
print(data.head())
print(data['Gender'].value_counts())

sns.boxplot(x='Purchased',y='Age',data=data)
plt.subplot(1,1,1)
plt.show()

sns.boxplot(x='Gender',y='Age',data=data)
plt.subplot(1,1,1)
plt.show()


sns.boxplot(x='Purchased',y='EstimatedSalary',data=data)
plt.subplot(1,1,1)
plt.show()

sns.boxplot(x='Gender',y='EstimatedSalary',data=data)
plt.subplot(1,1,1)
plt.show()


ohe=OneHotEncoder(sparse_output=False,drop='first')
male=pd.DataFrame(ohe.fit_transform(data[['Gender']]),columns=['Male'])
print(male.head())

data['Gender_male']=male
data.drop(['Gender'],axis=1,inplace=True)
print(data.head())

X=data.drop('Purchased',axis=1)
print(X.head())

Y=data['Purchased']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=10, test_size=0.2, stratify=Y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print(X_train_scaled[0:5])
X_test_scaled = scaler.fit_transform(X_test)