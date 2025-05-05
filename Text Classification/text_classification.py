#text classifier that takes movie reviews and classifies them as good or bad

import tensorflow as tf
from tensorflow import keras
import numpy as np

#vocaublary_size = 10000
vocaublary_size = 88000

data = keras.datasets.imdb

(train_data,train_labels), (test_data,test_labels) = data.load_data(num_words=vocaublary_size)

print("train data: ",train_data[:5],"train labels: ",train_labels[:5])

word_index = data.get_word_index()

#shift values by three and add special cases to index
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

#swap keys and values
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value= word_index["<PAD>"], padding = "post",maxlen= 250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value= word_index["<PAD>"], padding = "post",maxlen= 250)

#print(len(train_data),len(test_data))

def decode_review(text):
    return " ".join([reverse_word_index.get(i,"?") for i in text])

def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded


#print(decode_review(test_data[0]),". Length of review: ", len(test_data[1]))

#model
#we have to create vectors that represent each word so that they can be grouped with other similar context vectors
#we can measure similarites or differences between words by the directional differences between vectors

model = keras.Sequential([])
model.add(keras.layers.Embedding(vocaublary_size,16)) #create 10000 word vectors (every single number that represents a word), in a 16 dimensional space 
model.add(keras.layers.GlobalAveragePooling1D()) #shrinks / averages the vectors into a 1D representation which can be sent to 16 neurons
model.add(keras.layers.Dense(16,activation="relu"))
model.add(keras.layers.Dense(1,activation="sigmoid")) # squishes 16 neuron dense layer output to between 0 and 1

model.summary()

model.compile(optimizer = "adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000] # just taking 10000 examples from training date for validaton
x_train = train_data[10000:] # take last 10000 examples for training

y_val = train_labels[:10000]
y_train = train_labels[10000:]

#Batch size: https://www.geeksforgeeks.org/batch-size-in-neural-network/
#Batch Gradient Descent (BGD), uses lots of memory, may get stuck in poor solutio (less generalization), less noisy, faster convergence
#Mini-Batch Gradient Descent (MBGD), somewhat memory intensive but less (store gradients for mini-batches), helps avoid local minima (more randomness)
#Stochastic Gradient Descent (SGD), can converge faster or slower depending on model complexity, escapes local optima low memory usage 
#https://stackoverflow.com/questions/54413160/training-validation-testing-batch-size-ratio
#Training batch size: followes trade-offs mentioned above
#Validation and test batch sizes: pick the largest that your hardware can support 

fitModel = model.fit(x_train, y_train, epochs= 40, batch_size=512,validation_data = (x_val,y_val),verbose = 1)

results = model.evaluate(test_data,test_labels)

model.save("text_classifation.h5")

with open("test.txt", encoding = "utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"],padding="post",maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])

print("results: ",results)

human_eval = False

if human_eval == True:
    #perform human evaluation of how the model is working:
    test_review = test_data[0]
    nested_test_review = np.array([test_review])
    print("nested test review: ",nested_test_review)
    print("test data shape: ",test_data.shape)
    print("nested test review shape: ",nested_test_review.shape)
    predict = model.predict(nested_test_review)
    print("Review: ")
    print(decode_review(test_review))
    print("Prediction: " + str(predict[0]))
    print("Actual: " + str(test_labels[0]))
    print(results)

