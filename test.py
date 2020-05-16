import tensorflow as tf
import scipy.io as sio
import numpy
import os
from tensorflow import keras
from tensorflow.keras import layers
from scipy.io.wavfile import read
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

a = read("audioset__3RvCwwIZ4w_10_15.wav")
testArray = numpy.array(a[1], dtype=float)

## standard convention
batch_size = 32

## Convert training data into arraylist of test arrays
progress = 0

sickTraining = []
for filename in os.listdir("c:/Users/warri/Desktop/respire/audio/audio/test/sick"):
    progress += 1
    print(progress)
    sickTraining.append("c:/Users/warri/Desktop/respire/audio/audio/test/sick/" + filename)

not_sickTraining = []
for notsickfile in os.listdir("c:/Users/warri/Desktop/respire/audio/audio/test/not_sick"):
    progress += 1
    print(progress)
    not_sickTraining.append("c:/Users/warri/Desktop/respire/audio/audio/test/not_sick/" + notsickfile)

## Convert validation data into arraylist of validation arrays
sickTrainingConverted = []
for file in sickTraining:
    a = read(file)
    numberArray = numpy.array(a[1], dtype=float)
    sickTrainingConverted.append(numberArray)
    progress += 1
    print(progress)

notSickTrainingConverted = []
for file in not_sickTraining:
    a = read(file)
    numberArray = numpy.array(a[1], dtype=float)
    notSickTrainingConverted.append(numberArray)
    progress += 1
    print(progress)

# print("Yo im sick.")
# print(sickTrainingConverted)

# print("Yo im healthy.")
# print(notSickTrainingConverted)

## preprocessing/ normalizing our data
training_data = sickTrainingConverted

vectorizer = TextVectorization(output_mode="int")
vectorizer.adapt(training_data[0])

training_data_shit = vectorizer(training_data)
print(training_data)