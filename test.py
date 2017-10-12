import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\Paralizis\Desktop\2017\Machine Learning\machine_learning_utfsm\kc_house_data.csv")
df.drop(['id','date','zipcode',],axis=1,inplace=True)
df.head()

print("1b.-\n")
df.shape
df.info()
df.describe()

print("1c.-\n")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled['price'] = np.log(df['price'])
df_scaled.plot()

import sklearn.linear_model as lm
X = df_scaled.iloc[:,1:] #use .ix instead, in older pandas version //Se consideran todas las filas y columnas menos 'price'
N = X.shape[0] #Cantidad de filas
X.insert(X.shape[1], 'intercept', np.ones(N)) #X.shape[1] = dimensionalidad de datos (18)
y = df_scaled['price']
print(X.head())