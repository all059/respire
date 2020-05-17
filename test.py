import tensorflow as tf
import scipy.io as sio
import numpy
import os
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from scipy.io.wavfile import read
from tensorflow.keras.layers.experimental.preprocessing import Normalization

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
## Convert validation data into arraylist of validation arrays
for file in sickTraining:
    a = read(file)
    numberArray = numpy.array(a[1], dtype=float)
    # new_array = numpy.append(numberArray,1)
    totalValidationArray.append(numberArray)
    totalValidationArrayLabels.append(1)

for file in not_sickTraining:
    a = read(file)
    numberArray = numpy.array(a[1], dtype=float)
    # new_array = numpy.append(numberArray,0)
    totalValidationArray.append(numberArray)
    totalValidationArrayLabels.append(0)

# print("Yo im sick.")
# print(sickTrainingConverted)

# print("Yo im healthy.")
# print(notSickTrainingConverted)

## preprocessing/ normalizing our data


# normalizer = Normalization(axis=-1)
# normalizer.adapt(sickTrainingConverted[0])
# normalized_data = normalizer(sickTrainingConverted[0])
# print("var: %.4f" % numpy.var(normalized_data))
# print("mean: %.4f" % numpy.mean(normalized_data))
padded_input = tf.keras.preprocessing.sequence.pad_sequences(totalValidationArray, padding='post')

train_dataset = tf.data.Dataset.from_tensor_slices((padded_input, totalValidationArrayLabels))


BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
# test_dataset = test_dataset.batch(BATCH_SIZE)

model = tf.keras.Sequential([
    # tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_dataset, epochs=10)

# model = keras.Sequential()
# model.add(layers.Dense(12, input_dim = 200, activation='relu'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# sizeOfArray = totalValidationArray[0].size
# X = totalValidationArray[0][0:sizeOfArray-2]
# Y = totalValidationArray[0][sizeOfArray-1]
# model.fit(X, Y, epochs=100, batch_size=batch_size)