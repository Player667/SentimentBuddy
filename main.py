import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from tensorflow.python.framework import ops


import numpy
import tflearn
import tensorflow
import random
import json
import pickle

nltk.download('punkt')

#saving model so that pre-processing is more efficient through try-except clause
#try:
    #don't run this if you add new intents to the json file
    #with open("data.pickle", "rb") as f:
        #words, labels, training, output = pickle.load(f) #if there are saved values, we load in the lists saved in the file and avoid all the preprocessing stuff
#except:
with open("intents.json") as file:
    data = json.load(file)

    words = []
    labels = []
    #for each pattern stored in docs_x, docs_y should contain what intent/tag it's a part of
    docs_x = []
    docs_y = []
    ignore_case = ["?", "!", ".", "...", "'"]

    #looping through every single intent, the following segment of code is for data-preprocessing
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            #stemming - take each word in our pattern and bring them down to the root words, mmakes the model more accurate
            wrds = nltk.word_tokenize(pattern) #tokenizing is getting all the words in the pattern, returns them in a list
            words.extend(wrds) #add the elements to the list
            docs_x.append(wrds) #appending tokenized words
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])


    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_case] #if statement at the end means punctuation has no value
    #make words into a set to remove duplicates
    words = sorted(list(set(words)))

    labels = sorted(labels)

    #creating training and testing output. neural networks only recognize numbers, so we're gonna represent
    #the input in a way that checks the existence of each word and represent them in a list (one-hot encoding)
    #the output will repeat the same process of one-hot encoding but for the tags
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        
        wrds = [stemmer.stem(w) for w in doc]
        #go through all the diff words in doc and represent them using a 1 or a 0 to represent whether or not they are present
        for w in words:
            if w in wrds: #if word exists in current pattern we're looping through
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    #convert to numpy arrays
    training = numpy.array(training)
    output = numpy.array(output)

    #saving everything in a pickle file
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)




#building the model with tflearn
ops.reset_default_graph() #gets rid of all underlying graphed data, almost like a reset
net = tflearn.input_data(shape=[None, len(training[0])]) #defines the input shape we are expecting for the model, 0 means we should expect array = len[words]
net = tflearn.fully_connected(net, 100) #fully connected layer added to neural network, 8 neurons to the first "hidden layer"
net = tflearn.fully_connected(net, 100) #fully connected layer added to neural network, 8 neurons to the secondd "hidden layer"
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax") #return probabilities of each output
tflearn.regression(net, learning_rate=0.001)

#simple explanation: start with a bunch of input neurons equal to how many words we have (first layer), next layer of 8 neurons
#that each of the first layer neurons connect to, then another layer of 8 neurons that are fully connencted, and then there is
#the output layer with 6 neurons fully connected to the last layer. Softmax activation on the last layer will be run through a func
#that returns a probability to each of the neurons for tags. All the model really does is predict which tag it should
#return to the user. The hidden layers are supposed to examine different words andd manipulate the weights/bias to basically assign words
#to some kind of an output

#training model
model = tflearn.DNN(net) #DNN is a type of neural network

#try:
    #if a model exists, we'll just load it in
    #model.load("model.tflearn")
#except:
    #fitting the model
model.fit(training, output, n_epoch = 1000, batch_size = 8, show_metric = True) #n_epoch is just num times the model will see the data
model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0] * len(words)
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        if se in words:
            bag[words.index(se)] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with SentimentScout! Type \"Quit\" to leave.")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        results = model.predict([bag_of_words(inp.lower(), words)])[0] #returns the probabilities, not a response to return to the user
        results_index = numpy.argmax(results) #gives the index of the greatest prob
        tag = labels[results_index]

        #if results[results_index] > 0.7: greater than 70% confidence
        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]
                print(random.choice(responses))
            
        #else:
            #print("I didn't get that, try again.")

        


chat()
