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
for filename in os.listdir("c:/Users/warri/Desktop/respire/audio/audio/train/sick"):
    sickTraining.append("c:/Users/warri/Desktop/respire/audio/audio/train/sick/" + filename)

not_sickTraining = []
for notsickfile in os.listdir("c:/Users/warri/Desktop/respire/audio/audio/train/not_sick"):
    not_sickTraining.append("c:/Users/warri/Desktop/respire/audio/audio/train/not_sick/" + notsickfile)

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

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("starting training")
model.fit(totalValidationArray, totalValidationArrayLabels, epochs=5, batch_size=batch_size)
print(model.evaluate(totalValidationArray, totalValidationArrayLabels))

## TEST DATA
sickTrainingTest = []
for filename in os.listdir("c:/Users/warri/Desktop/respire/audio/audio/test/sick"):
    sickTraining.append("c:/Users/warri/Desktop/respire/audio/audio/test/sick/" + filename)

not_sickTrainingTest = []
for notsickfile in os.listdir("c:/Users/warri/Desktop/respire/audio/audio/test/not_sick"):
    not_sickTraining.append("c:/Users/warri/Desktop/respire/audio/audio/test/not_sick/" + notsickfile)

totalTestArray = []

# Convert test data into arraylist of test arrays
for file in sickTraining[0:9]:
    a = read(file)
    print(file)
    numberArray = numpy.array(a[1], dtype=float)
    result = numpy.zeros(MAX_SIZE)
    result[:numberArray.shape[0]] = numberArray
    totalTestArray.append(result)    
    progress+=1
    print(progress)

for file in not_sickTraining[0:9]:
    a = read(file)
    numberArray = numpy.array(a[1], dtype=float)
    print(file)
    result = numpy.zeros(MAX_SIZE)
    result[:numberArray.shape[0]] = numberArray
    totalTestArray.append(result)    
    progress+=1
    print(progress)

model.save('cough_reader.model')
new_model = tf. keras.models.load_model('cough_reader.model')
totalTestArray = numpy.array(totalTestArray)
predictions = new_model.predict(totalTestArray)

##printing predictions
for prediction in predictions:
    print(numpy.argmax(prediction))

