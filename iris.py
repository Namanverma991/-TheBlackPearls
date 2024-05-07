import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/kaggle/input/iriscsv/Iris.csv')
df.head()
df.shape
df.info()
# To check if any null value in dataset.
df.isnull().sum()
# Check the unique values of Species column
df['Species'].unique()
# creating different data frame respect to unique species value..
df_setosa = df.loc[df['Species']=='Iris-setosa']
df_versicolor = df.loc[df['Species']=='Iris-versicolor']
df_virginica = df.loc[df['Species']=='Iris-virginica']
df_setosa.head()
# create plot of sepal lenth..
plt.plot(df_setosa['SepalLengthCm'],np.zeros_like(df_setosa['SepalLengthCm']),'o')
plt.plot(df_versicolor['SepalLengthCm'],np.zeros_like(df_versicolor['SepalLengthCm']),'o')
plt.plot(df_virginica['SepalLengthCm'],np.zeros_like(df_virginica['SepalLengthCm']),'o')
plt.xlabel("SepalLengthCm")
plt.show()
# reation between sepal width and petal width
sns.scatterplot(x='SepalWidthCm',y='PetalWidthCm',data=df,hue='Species')
plt.show()
#relation between Sepal Length and Petal Length
sns.scatterplot(x='SepalLengthCm',y='PetalLengthCm',data=df,hue='Species')
plt.show()
df_no_id_clmn=df.drop('Id',axis=1)
df_no_id_clmn.head()