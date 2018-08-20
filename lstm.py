import numpy as np
from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(2018)
import pandas as pd

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam,Adadelta,Nadam,Adamax,RMSprop
from sklearn.model_selection import train_test_split



def get_model(timeseries, nfeatures, nclass):
    model = Sequential()
    model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=(timeseries, nfeatures)))
    model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    model.add(Dense(units=nclass, activation='softmax'))

    return model

data = np.load('../data/voice_zaloai/train.npz')
X, gender, region = data['X'], data['gender'], data['region']
X_train, X_test, gender_train, gender_test, region_train, region_test = train_test_split(X, gender, region, test_size=0.2, random_state=2018)

publictest = np.load('../data/voice_zaloai/publictest.npz')
X_publictest, fname = publictest['X'], publictest['name']

print('train test: ', X_train.shape, X_test.shape)
print('public test: ', X_publictest.shape)

opt = RMSprop()
model = get_model(X.shape[1], X.shape[2], 3)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

batch_size = 1024
nb_epochs = 10000

model.fit(X_train, region_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_test, region_test), verbose=2)

predicts = model.predict(X_publictest, batch_size=batch_size)
predicts = np.argmax(predicts, axis=1)

region_dict = {0:'north', 1:'central', 2:'south'}
gender_dict = {0:'female', 1:'male'}
for i in range(32):
    print(fname[i], '-->', region_dict[predicts[i]])

submit = pd.DataFrame.from_dict({'id':fname, 'accent':predicts}) 
submit.to_csv('submit.csv', index=False)
