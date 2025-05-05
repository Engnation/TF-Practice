import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #1 = INFO messages are not printed, 2 = INFO and WARNING messages are not printed, 3 = INFO, WARNING, and ERROR messages are not printed

import matplotlib.pyplot as plt
from matplotlib import use
import numpy as np
import tensorflow as tf
from tensorflow import keras

#print('imports successful, using tensorflow version: ' + tf.__version__)
#''''
#load train and test data
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()

#normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

#select one sample / label pair
sample = x_train[-1]
label = y_train[-1]

#display pair
use('Agg') #if using headless WSL
plt.imshow(sample,cmap=plt.cm.binary)
plt.show()
print("image label: ",label)
#'''

#Feed forward nerual network
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10),
])

#config
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001) # "adam"
metrics = [keras.metrics.SparseCategoricalAccuracy()] # "accuracy"

#compile
model.compile(loss=loss,optimizer=optim,metrics=metrics)

#fitting/training
#model.fit(x_train,y_train,batch_size=64,epochs=5,shuffle=True,verbose=2) #64_batch
model.fit(x_train,y_train,batch_size=1,epochs=5,shuffle=True,verbose=1) #sgd

#print("Evaluate: ")
#model.evaluate(x_test,y_test,verbose=1)

# 1) Save whole model
#model.save("mnist_nn.h5") #the old way
#model.save("mnist_nn_64_batch.keras") # the new way
model.save("mnist_nn_sgd.keras") # the new way

#load model
print("Evaluate with batch size of 64: ")
new_model = keras.models.load_model("mnist_nn_64_batch.keras")
new_model.evaluate(x_test,y_test,verbose=1)

print("Evaluate with SGD: ")
new_model = keras.models.load_model("mnist_nn_sgd.keras")
new_model.evaluate(x_test,y_test,verbose=1)
# 2) Save only weights

#model.save_weights("nn_weights.h5")

#initialize
#model.load_weights("nn_wights.h5")

# 3) Save only architecture, to_json
'''
json_string = model.to_json()

with open("nn_model", "w") as f:
    f.write(json_string)

with open("nn_model","r") as f:
    loaded_json_string = f.read()

new_model = keras.models.model_from_json(loaded_json_string)
print(new_model.summary())
'''