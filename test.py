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