
#importing necessary libraries
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn import tree, preprocessing
from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay,f1_score,precision_score,recall_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import sklearn.ensemble as ske

# Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive')

cd '/content/drive/MyDrive/ALDA_Project'

# reading from sqlite database 
from sqlalchemy import create_engine
my_conn=create_engine("sqlite:////content/drive/MyDrive/ALDA_Project/FPA_FOD_20170508.sqlite")

# Reading the sql file 
df = pd.read_sql_query("SELECT FOD_ID, FIRE_YEAR, DISCOVERY_DATE, DISCOVERY_TIME, CONT_DATE, CONT_TIME, FIRE_SIZE, FIRE_SIZE_CLASS, LATITUDE, LONGITUDE, STATE, COUNTY, STAT_CAUSE_DESCR FROM Fires", my_conn)
df_original= df # a copy will be used later

#converting the data to normal datatime format
df['START_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'] - pd.Timestamp(0).to_julian_date(), unit='D')
df['END_DATE'] = pd.to_datetime(df['CONT_DATE'] - pd.Timestamp(0).to_julian_date(), unit='D')
df.drop(['DISCOVERY_DATE', 'CONT_DATE'], axis=1, inplace=True)
df['MONTH'] = pd.DatetimeIndex(df['START_DATE']).month
# Reordering the columns
new_cols=['FOD_ID', 'FIRE_YEAR','MONTH','START_DATE','DISCOVERY_TIME', 'END_DATE','CONT_TIME', 'FIRE_SIZE', 'FIRE_SIZE_CLASS', 'LATITUDE', 'LONGITUDE', 'STATE', 'COUNTY', 'STAT_CAUSE_DESCR']
df=df.reindex(columns=new_cols)
print(df.shape)
# First five rows
print(df.head())

"""## EDA"""

#check mising values and columns
q = df.columns[df.isnull().any()].tolist() 
print(q)

# Number of Fires and different reasons
df['STAT_CAUSE_DESCR'].value_counts().plot(kind='barh')
plt.title('WildFire Reasons and Count')
plt.show()

# Wildfire cause and firecouny distribution
causes = df['STAT_CAUSE_DESCR'].value_counts()
fig = plt.figure(figsize =(10, 7))
plt.pie(causes, autopct='%1.1f%%',labels=causes.index)
plt.show()

"""Debris burning , Arson, Lightning are the major causes of wildfire accounting for more than 50% of total wildfires"""

#Yearwise firecount distribution
df['FIRE_YEAR'].value_counts().plot(kind='barh',color='lightblue')
plt.show()

"""There is no significant difference in year on year wildfire count"""

# monthwise firecount distribution
months = df['MONTH'].value_counts()
fig = plt.figure(figsize =(10, 7))
plt.pie(months, autopct='%1.1f%%',labels=months.index)
plt.show()

"""There is significant difference between wildfire count from October-January and February-September. It might be because of weather, since weather is cold from October-January which is not conducive for wildfire. """

# Plotting statewise firecount
df['STATE'].value_counts().head(n=10).plot(kind='barh')
plt.title('StateWise fire count')
plt.show()

df_CA = df[df['STATE']=='CA']
df_GA = df[df['STATE']=='GA']
df_TX = df[df['STATE']=='TX']

# causes of fire in CA
df_CA['STAT_CAUSE_DESCR'].value_counts().plot(kind='barh',color='lightblue',title='causes of fires for CA')
plt.show()

# causes of fire in GA
df_GA['STAT_CAUSE_DESCR'].value_counts().plot(kind='barh',color='lightblue',title='causes of fires for GA')
plt.show()

# causes of fire in TX
df_TX['STAT_CAUSE_DESCR'].value_counts().plot(kind='barh',color='lightblue',title='causes of fires for TX')
plt.show()

# plotting firesize vs firecause
df1 = df.groupby('STAT_CAUSE_DESCR').mean()
plt.bar(df['STAT_CAUSE_DESCR'].unique(),df1['FIRE_SIZE'])
plt.xticks(rotation=90)
plt.title('Average Fire size (in acres) v/s Fire Cause')
plt.show()

"""Causes of fire vary state by state, so location will be critical attribute in prediction.

# Visualising the Dataset:
"""

#encoding the labels
le = preprocessing.LabelEncoder()
df['STAT_CAUSE_DESCR'] = le.fit_transform(df['STAT_CAUSE_DESCR'])
df['STATE'] = le.fit_transform(df['STATE'])
df.rename(columns = {"STAT_CAUSE_DESCR":"CAUSE"}, inplace="True")

print(df.head())

df['DAY_OF_WEEK'] = pd.DatetimeIndex(df['START_DATE']).weekday

#Corralation
plt.figure(figsize=(12, 8))
df_copy = df.drop('FOD_ID', axis=1)
corr = df_copy.corr()
sb.heatmap(corr, annot=True)
plt.title('Correalation Matrix')
plt.show()

print(df.head())

xdf = df.drop(['FOD_ID','START_DATE','DISCOVERY_TIME','END_DATE','CONT_TIME','FIRE_SIZE_CLASS','COUNTY','CAUSE'], axis=1)
ydf = df['CAUSE']

X = df.drop(['FOD_ID','START_DATE','DISCOVERY_TIME','END_DATE','CONT_TIME','FIRE_SIZE_CLASS','COUNTY','CAUSE'], axis=1).values
y = df['CAUSE'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

# RFC with 13 features in the cause
model1 = ske.RandomForestClassifier(n_estimators=50)
model1 = model1.fit(X_train, y_train)
print(model1.score(X_test,y_test))

# making 4 categories from 13 features
def set_label(cat):
    cause = 0
    natural = ['Lightning']
    accidental = ['Structure','Fireworks','Powerline','Railroad','Smoking','Children','Campfire','Equipment Use','Debris Burning']
    malicious = ['Arson']
    other = ['Missing/Undefined','Miscellaneous']
    if cat in natural:
        cause = 1
    elif cat in accidental:
        cause = 2
    elif cat in malicious:
        cause = 3
    else:
        cause = 4
    return cause
     

df['LABEL'] = df_original['STAT_CAUSE_DESCR'].apply(lambda x: set_label(x)) # I created a copy of the original df earlier in the kernel
df = df.drop('CAUSE',axis=1)
print(df.head())

w = df['LABEL'].value_counts()
print(w)

X = df.drop(['FOD_ID','START_DATE','DISCOVERY_TIME','END_DATE','CONT_TIME','FIRE_SIZE_CLASS','COUNTY','LABEL'], axis=1).values
y = df['LABEL'].values
print(set(y))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

"""# General Methods

## Random Forest Classifier
"""

# Random Forest Classifier
model2 = ske.RandomForestClassifier(n_estimators=50)
y_pred = model2.fit(X_train, y_train).predict(X_test)
#print(confusion_matrix(y_true=y_test,y_pred=y_pred))
print(classification_report(y_test, y_pred))
cm_rfc = confusion_matrix(y_test,y_pred,normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rfc,display_labels=['natural','accidental','malicious','other'])
disp.plot()



"""## Adaboost Classifier"""

#Adaboost classifier

from sklearn.ensemble import AdaBoostClassifier

clfab = AdaBoostClassifier(n_estimators=100, random_state=0)
clfab.fit(X_train, y_train)
y_pred_ab = clfab.predict(X_test)
print(classification_report(y_test, y_pred_ab))

cm_ab = confusion_matrix(y_test,y_pred_ab,normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm_ab,display_labels=['natural','accidental','malicious','other'])
disp.plot()

"""## Decision Tree Classifier"""

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clfdt = DecisionTreeClassifier(random_state=0)
clfdt.fit(X_train, y_train)
y_pred_dt = clfdt.predict(X_test)
print(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test,y_pred_dt,normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm_dt,display_labels=['natural','accidental','malicious','other'])
disp.plot()

"""## K Nearest Neighbor Classifier"""

# K Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print(classification_report(y_test, y_pred_knn))
cm_knn = confusion_matrix(y_test,y_pred_knn,normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm_knn,display_labels=['natural','accidental','malicious','other'])
disp.plot()



"""# Bi-Directional LSTM"""

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout, Conv1D, MaxPooling1D
import keras.utils
from keras.utils.np_utils import to_categorical
from keras.layers import LSTM

# redefining training and testing set
xdf = df.drop(['FOD_ID','START_DATE','DISCOVERY_TIME','END_DATE','CONT_TIME','FIRE_SIZE_CLASS','COUNTY','LABEL'], axis=1)
ydf = df['LABEL']

print("Shape of x dataframe",xdf.shape)
print("Shape of y dataframe",ydf.shape)

x_train, x_test, y_train, y_test = train_test_split(xdf,ydf,test_size=0.3,train_size=0.7)

print("X_train, Y_train shape",x_train.shape, y_train.shape)
print("\nX_test, Y_test shape",x_test.shape, y_test.shape)

# Using Minmax scaler
from sklearn.preprocessing import StandardScaler,MinMaxScaler

scaler = MinMaxScaler()    # robust to outliers

scaler_train = scaler.fit(x_train)
scaler_val = scaler.fit(x_test)

x_train.loc[:] = scaler_train.transform(x_train.to_numpy())
x_test.loc[:] = scaler_train.transform(x_test.to_numpy())

# converting fro 2D data 3D data
from scipy.stats import mode

def create_dataset(x, y, timesteps, step):
  x_set = []
  y_set = []
  for i in range(0,len(x)-timesteps, step):
    data = x.iloc[i: (i + timesteps)].values
    labels = y.iloc[i: i + timesteps]
    x_set.append(data)
    y_set.append(mode(labels)[0][0])
  
  return np.array(x_set), np.array(y_set).reshape(-1,1)

timesteps = 4
steps = 1

X_train, Y_train = create_dataset( x_train, y_train, timesteps, steps)        # making the dataset ready for training
X_test , Y_test = create_dataset( x_test, y_test, timesteps, steps)           # making the dataset ready for validation

from tensorflow.keras.utils import to_categorical

print("Shape of x_train",X_train.shape)          
print("Shape of y_train",Y_train.shape)          

print("Shape of x_test",X_test.shape)           
print("shape of Y_test",Y_test.shape)           
print()

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
print("Shpae of Y_train after categorical",Y_train.shape)          
print("Shape of Y_test after categorical",Y_test.shape)

# parameters model building
batch_size = 128
epochs = 10

# Builfing Bi-LSTM model
model = keras.Sequential()
model.add(keras.layers.Bidirectional(LSTM( units = 128, input_shape = [X_train.shape[1],X_train.shape[2]])))
model.add(Dropout(rate = 0.6))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units = 64,activation='relu'))
model.add(Dense(units=5,activation='softmax'))

model.compile(loss= 'categorical_crossentropy', optimizer = 'Adam', metrics = ['Accuracy'])

# fitting the model
history = model.fit(X_train,Y_train, epochs = 1, batch_size = batch_size, validation_data= (X_test,Y_test))

# model prediction
print(model.summary(line_length= 80))
y_test_pred = model.predict(X_test)
p = np.argmax(y_test_pred,axis=1)
q = np.argmax(Y_test,axis=1)

from sklearn.metrics import precision_score, f1_score, classification_report,recall_score,accuracy_score
prec = precision_score(q,p, average='weighted') # no warning
print("\nPrecision score :",prec)
f1 = f1_score(q,p, average='weighted')
print("\nF1_score \t:", f1)
recall = recall_score(q,p,average= 'micro')
print("\nRecall score    :",recall)
accuracy = accuracy_score(q,p)
print("\nAccuracy \t:",accuracy)
print()

"""# CNN"""

#2D to 3D conversion
def create_dataset(x, y, timesteps, step):
  x_set = []
  y_set = []
  for i in range(0,len(x)-timesteps, step):
    data = x.iloc[i: (i + timesteps)].values
    labels = y.iloc[i: i + timesteps]
    x_set.append(data)
    y_set.append(mode(labels)[0][0])
  
  return np.array(x_set), np.array(y_set).reshape(-1,1)

timesteps = 40
steps = 1
X_train, Y_train = create_dataset( x_train, y_train, timesteps, steps)
X_test , Y_test = create_dataset( x_test, y_test, timesteps, steps)

print("Shape of x_train",X_train.shape)          
print("Shape of y_train",Y_train.shape)          

print("Shape of x_test",X_test.shape)           
print("shape of Y_test",Y_test.shape)           
print()

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
print("Shpae of Y_train after categorical",Y_train.shape)          
print("Shape of Y_test after categorical",Y_test.shape)

# Building the model
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit network
history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=3, batch_size=128, verbose=1)

# model prediction
print(model.summary(line_length= 80))
y_test_pred = model.predict(X_test)
p = np.argmax(y_test_pred,axis=1)
q = np.argmax(Y_test,axis=1)

from sklearn.metrics import precision_score, f1_score, classification_report,recall_score,accuracy_score
prec = precision_score(q,p, average='weighted') # no warning
print("\nPrecision score :",prec)
f1 = f1_score(q,p, average='weighted')
print("\nF1_score \t:", f1)
recall = recall_score(q,p,average= 'micro')
print("\nRecall score    :",recall)
accuracy = accuracy_score(q,p)
print("\nAccuracy \t:",accuracy)
print()