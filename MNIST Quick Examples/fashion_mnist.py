import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#https://stackoverflow.com/questions/51306862/how-do-i-use-tensorflow-gpu
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
#print(sess) #session information
tf.print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version: ",tf.__version__) 

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

#normalize
train_images, test_images = train_images / 255.0, test_images / 255.0

#plt.imshow(train_images[-1],cmap=plt.cm.binary)
#plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"), # dense = fully connected layer
    keras.layers.Dense(10, activation="softmax")  #all of the values of the network add up to 1
    ])

model.summary()

#specify model parameters (optimizer, loss function, metrics)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=["accuracy"]) 

print("FITTING")
t = time.process_time()
#training for the number of times the training data will be stochastically fed through the network
model.fit(train_images,train_labels, epochs=7) #<-- sometimes more epochs could reduce model reliability

elapsed_time = time.process_time() - t
print("ELAPSED TRAINING TIME: ",elapsed_time)

#model.save("fashion_mnist_default.keras")

#model = keras.models.load_model("fashion_mnist_default.keras")

print("EVALUATING")
#testing on unseen data (accuracy %, loss = sum of errors for each sample in validation set, used for model weights in training)
test_loss, test_acc = model.evaluate(test_images,test_labels)

#epoch: one full cycle through the training data (composed of many iterations (or batches))
#batch size: number of samples being processed in a batch
#steps: total training images / batch size = number of steps

print("Tested accuracy: ", test_acc)

#now let's try to make a prediction based on the trained model. Let's pick a sample to predict:
 
prediction = model.predict(test_images)

show_single_predict = True
show_multiple_predict = False

if show_single_predict == True:
    s = 61
    #the "prediction" for all of the test images is a list of lists with probabilies of a class
    
    #print(prediction)
    #using np.armax to pick the highest probability and index into class names
    print(class_names[np.argmax(prediction[s])])

    plt.imshow(test_images[s],cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[s]])
    plt.title(" Prediction: " + class_names[np.argmax(prediction[s])])
    plt.show()

if show_multiple_predict == True:
    for i in range(5):
        plt.grid(False)
        plt.imshow(test_images[i],cmap=plt.cm.binary)
        plt.xlabel("Actual: " + class_names[test_labels[i]])
        plt.title(" Prediction: " + class_names[np.argmax(prediction[i])])
        plt.show()


