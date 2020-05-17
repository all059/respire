import tensorflow as tf
import scipy.io as sio
import numpy
import os
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from scipy.io.wavfile import read
from tensorflow.keras.layers.experimental.preprocessing import Normalization


## convert this to a jupyter notebook
## TODO: test model on training data, test model on test data
## get an accuracy we are happy with

## standard convention
batch_size = 32

## Convert training data into arraylist of test arrays
progress = 0

sickTraining = []
for filename in os.listdir("c:/Users/warri/Desktop/respire/audio/audio/test/sick"):
    sickTraining.append("c:/Users/warri/Desktop/respire/audio/audio/test/sick/" + filename)

not_sickTraining = []
for notsickfile in os.listdir("c:/Users/warri/Desktop/respire/audio/audio/test/not_sick"):
    not_sickTraining.append("c:/Users/warri/Desktop/respire/audio/audio/test/not_sick/" + notsickfile)

totalValidationArray = []
totalValidationArrayLabels = []
MAX_SIZE = 300000
## Convert validation data into arraylist of validation arrays
for file in sickTraining:
    a = read(file)
    numberArray = numpy.array(a[1], dtype=float)
    result = numpy.zeros(MAX_SIZE)
    result[:numberArray.shape[0]] = numberArray
    totalValidationArray.append(result)
    totalValidationArrayLabels.append(1)
    progress+=1
    print(progress)

for file in not_sickTraining:
    a = read(file)
    numberArray = numpy.array(a[1], dtype=float)
    result = numpy.zeros(MAX_SIZE)
    result[:numberArray.shape[0]] = numberArray
    totalValidationArray.append(result)    
    totalValidationArrayLabels.append(0)
    progress+=1
    print(progress)

print("converting array")
totalValidationArray = numpy.array(totalValidationArray)
totalValidationArrayLabels = numpy.array(totalValidationArrayLabels)

print(totalValidationArray.shape)
print(totalValidationArrayLabels)

training_data_x = pd.DataFrame(data=totalValidationArray[0:1])
print(training_data_x.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("starting training")
model.fit(totalValidationArray, totalValidationArrayLabels, epochs=10, batch_size=batch_size)

print(model.evaluate(totalValidationArray, totalValidationArrayLabels))

## TEST DATA
sickTrainingTest = []
for filename in os.listdir("c:/Users/warri/Desktop/respire/audio/audio/test/sick"):
    sickTraining.append("c:/Users/warri/Desktop/respire/audio/audio/test/sick/" + filename)

not_sickTrainingTest = []
for notsickfile in os.listdir("c:/Users/warri/Desktop/respire/audio/audio/test/not_sick"):
    not_sickTraining.append("c:/Users/warri/Desktop/respire/audio/audio/test/not_sick/" + notsickfile)

totalValidationArray = []
totalValidationArrayLabels = []
MAX_SIZE = 300000
## Convert validation data into arraylist of validation arrays
for file in sickTraining:
    a = read(file)
    numberArray = numpy.array(a[1], dtype=float)
    result = numpy.zeros(MAX_SIZE)
    result[:numberArray.shape[0]] = numberArray
    totalValidationArray.append(result)
    totalValidationArrayLabels.append(1)
    progress+=1
    print(progress)

for file in not_sickTraining:
    a = read(file)
    numberArray = numpy.array(a[1], dtype=float)
    result = numpy.zeros(MAX_SIZE)
    result[:numberArray.shape[0]] = numberArray
    totalValidationArray.append(result)    
    totalValidationArrayLabels.append(0)
    progress+=1
    print(progress)