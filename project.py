import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("Social_Network_Ads.csv")
print(data.head())
print(data.shape)
print(data.isnull().sum())
print(data.drop_duplicates(inplace=True))
print(data['Gender'].unique())
print(data['Purchased'].unique())
print(data.drop(['User ID'],axis=1,inplace=True))
print(data.head())
print(data['Gender'].value_counts())