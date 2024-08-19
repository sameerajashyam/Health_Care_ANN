import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import  train_test_split
import pickle
from sklearn.preprocessing import OneHotEncoder
data=pd.read_csv(r'healthcare_dataset.csv')
print(data.head())
print(data.columns)
print(data.isnull().sum())
print('---------------')

columns=data.columns
for i in columns:
    print(i,data[i].unique())
    print('-------------')
print(columns)
'''
['Name', 'Age', 'Gender', 'Blood Type', 'Medical Condition',
       'Date of Admission', 'Doctor', 'Hospital', 'Insurance Provider',
       'Billing Amount', 'Room Number', 'Admission Type', 'Discharge Date',
       'Medication', 'Test Results']
'''
data=data.drop(['Name','Date of Admission','Doctor','Hospital','Insurance Provider','Billing Amount','Room Number','Discharge Date'],axis=1)
print(data.columns)

print('----------------------------')
#label encode for gender
label_encoder_gender=LabelEncoder()
data['Gender']=label_encoder_gender.fit_transform(data['Gender'])

onehot_encoder=OneHotEncoder()

medical_condition_encoder=onehot_encoder.fit_transform(data[['Medical Condition']]).toarray()
medical_condition_encoded_df=pd.DataFrame(medical_condition_encoder,columns=onehot_encoder.get_feature_names_out(['Medical Condition']))
data=pd.concat([data.drop('Medical Condition',axis=1),medical_condition_encoded_df],axis=1)
print(data.head())


admission_type_encoder=onehot_encoder.fit_transform(data[['Admission Type']]).toarray()
print(admission_type_encoder)
admission_type_encoded_df=pd.DataFrame(admission_type_encoder,columns=onehot_encoder.get_feature_names_out(['Admission Type']))
data=pd.concat([data.drop('Admission Type',axis=1),admission_type_encoded_df],axis=1)


blood_type_encoder=onehot_encoder.fit_transform(data[['Blood Type']]).toarray()
blood_type_encoded_df=pd.DataFrame(blood_type_encoder,columns=onehot_encoder.get_feature_names_out(['Blood Type']))
data=pd.concat([data.drop('Blood Type',axis=1),blood_type_encoded_df],axis=1)


medication_encoder=onehot_encoder.fit_transform(data[['Medication']]).toarray()
medication_encoded_df=pd.DataFrame(medication_encoder,columns=onehot_encoder.get_feature_names_out(['Medication']))
data=pd.concat([data.drop('Medication',axis=1),medication_encoded_df],axis=1)
print(data.head())

test_results_encoder=onehot_encoder.fit_transform(data[['Test Results']]).toarray()
test_results_encoded_df=pd.DataFrame(test_results_encoder,columns=onehot_encoder.get_feature_names_out(['Test Results']))
data=pd.concat([data.drop('Test Results',axis=1),test_results_encoded_df],axis=1)
print(data.columns)

X=data.drop(['Test Results_Abnormal','Test Results_Inconclusive', 'Test Results_Normal'],axis=1)
y=data[['Test Results_Abnormal','Test Results_Inconclusive', 'Test Results_Normal']]
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=35)


#scaling the features

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

print(X_train)
with open('label_encoder_gender.pkl','wb') as file:
    pickle.dump(label_encoder_gender,file)

with open('medical_condition_encoder.pkl','wb') as file:
    pickle.dump(medical_condition_encoder,file)

with open('admission_type_encoder.pkl','wb') as file:
    pickle.dump(admission_type_encoder,file)

with open('blood_type_encoder.pkl','wb') as file:
    pickle.dump(blood_type_encoder,file)
with open('medication_encoder.pkl','wb') as file:
    pickle.dump(medication_encoder,file)

with open('test_results_encoder','wb') as file:
    pickle.dump(test_results_encoder,file)


with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)
print(data.head())


#data.to_csv('processed_healthcare_data.csv', index=False)

print(data.columns)


####ANN implementation
import  tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import datetime

print('-------------------------------')
print((X_train.shape[1],))
print('-------------------------------')

#building ANN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),  # Optional additional layer
    Dense(3, activation='softmax')  # 3 output neurons for 3 classes
])
opt = tf.keras.optimizers.Adam(learning_rate=0.01)

# Compile the model with appropriate loss function
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

print('---------------------------------')
import  tensorflow
opt=tensorflow.keras.optimizers.Adam(learning_rate=0.01)
tensorflow.keras.losses.BinaryCrossentropy()

opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)

#complie the model
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])


#setup the tensor board

from tensorflow.keras.callbacks import (EarlyStopping,TensorBoard)


log_dir='logs/fit/'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)

#setup early stopping

early_stopping_callback=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)

### Train the model
history=model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,
    callbacks=[tensorflow_callback,early_stopping_callback]
)

model.save('model.h5')
