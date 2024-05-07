import libraries
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import folium
from folium import Circle
from geopy import Nominatim

from sklearn.preprocessing import MinMaxScaler
from IPython.display import display
df =pd.read_csv("/kaggle/input/unemployment-in-india/Unemployment in India.csv")
df.head()
df.sample(5)
df.columns
df.describe()
df.isnull().sum()
df[df.duplicated()]
df.drop_duplicates(inplace=True)
df.isnull().sum().sum()
df.isnull().sum()
sns.pairplot(df)
sns.pairplot(data = df , hue = 'Region')
df.Region.unique()
import missingno as msno
msno.matrix(df)
df['Area'].value_counts().plot.pie()
df['Region'].value_counts().plot.pie()
df['Area'].value_counts().plot.bar()