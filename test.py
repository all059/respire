import tensorflow as tf
import scipy.io as sio
import numpy
import os
import pandas as pd
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
# from keras.models import Sequential
from scipy.io.wavfile import read
from tensorflow.keras.layers.experimental.preprocessing import Normalization


## convert this to a jupyter notebook
## TODO: test model on training data, test model on test data
## get an accuracy we are happy with

## Standard convention for batch size; a batch size is the number of training examples used in one epoch
batch_size = 32

## Progress will be an int variable used to confirm that the for loops are running
progress = 0

## Meant to iterate through the training audio folders and append each .wav file into each list respectfully
sickTraining = []
for filename in os.listdir("c:/Users/warri/Desktop/respire/audio/audio/train/sick"):
    sickTraining.append("c:/Users/warri/Desktop/respire/audio/audio/train/sick/" + filename)

not_sickTraining = []
for notsickfile in os.listdir("c:/Users/warri/Desktop/respire/audio/audio/train/not_sick"):
    not_sickTraining.append("c:/Users/warri/Desktop/respire/audio/audio/train/not_sick/" + notsickfile)

## tVA will be the data coming from each wav file filled with 0's at the end
## tVAL is 1 for sick, 0 for not sick
totalValidationArray = []
totalValidationArrayLabels = []
MAX_SIZE = 300000

## Convert training data into a list of training arrays
## The read function from scipy reads wav files and returns an int and numpy array (samplerate, data)
## The data returned will be set as numberArray, however this is a very big array
## numpy.zeros returns an array of (shape, data type, order), our shape is 300,000
## Result is used to find how many rows of data are in a EACH numberArray
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

## list to array
print("converting array")
totalValidationArray = numpy.array(totalValidationArray)
totalValidationArrayLabels = numpy.array(totalValidationArrayLabels)

print(totalValidationArray.shape)
print(totalValidationArrayLabels)

## tf.keras.Sequential(layers=None, name=None)
## Sequential groups a linear stack of layers into a tf.keras.Model.
## Model groups layers into an object with training and inference features.
## https://tinyurl.com/yd9arn68 for info on relu and sigmoid

## model = tf.keras.Sequential()
##model.add(tf.keras.layers.Flatten())
##model.add(tf.keras.layers.Dense(128, activation='relu'))
##model.add(tf.keras.layers.Dense(128, activation='relu'))
##model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

## .fit will train the data
print("starting training")
model.fit(totalValidationArray, totalValidationArrayLabels, epochs=5, batch_size=batch_size)
print(model.evaluate(totalValidationArray, totalValidationArrayLabels))
 
## Meant to iterate through the test audio folders and append each .wav file into each list respectfully
sickTrainingTest = []
for filename in os.listdir("c:/Users/warri/Desktop/respire/audio/audio/test/sick"):
    sickTraining.append("c:/Users/warri/Desktop/respire/audio/audio/test/sick/" + filename)

not_sickTrainingTest = []
for notsickfile in os.listdir("c:/Users/warri/Desktop/respire/audio/audio/test/not_sick"):
    not_sickTraining.append("c:/Users/warri/Desktop/respire/audio/audio/test/not_sick/" + notsickfile)

totalTestArray = []

# Convert testing data into arraylist of test arrays
for file in sickTrainingTest[0:9]:
    a = read(file)
    print(file)
    numberArray = numpy.array(a[1], dtype=float)
    result = numpy.zeros(MAX_SIZE)
    result[:numberArray.shape[0]] = numberArray
    totalTestArray.append(result)    
    progress+=1
    print(progress)

for file in not_sickTrainingTest[0:9]:
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

