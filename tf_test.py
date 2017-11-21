###############################################################################
###############################################################################
###############################################################################
##
##
## tf_test.py
##
## @author: Matthew Cline
## @version: 20171120
##
## Description: File to mess around with tensor flow. Might end up using a 
##              deep neural net for the kaggle challenge data in Machine
##              Learning.
##
##
###############################################################################
###############################################################################
###############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import mean_squared_error
import tensorflow as tf

np.set_printoptions(threshold=np.nan)


####### PREPROCESSING FUNCTIONS #######
def splitData(data, trainingSplit=0.6, validationSplit=0.8):
    training, validation, test = np.split(data, [int(trainingSplit*len(data)), int(validationSplit*len(data))])
    return training, validation, test

def shuffleData(data):
    data = data.reindex(np.random.permutation(data.index))
    data = data.reset_index(drop = True)
    return data

def standardizeFeatures(data, exceptions):
    for i in range(0, data.shape[1]):
        if i in exceptions:
            continue
        data.iloc[:,i] = (data.iloc[:,i] - np.mean(data.iloc[:,i]) / np.std(data.iloc[:,i]))
    return data

def scaleFeatures(data, exceptions):
    for i in range(0, data.shape[1]):
        if i in exceptions:
            continue
        data.iloc[:,i] = (data.iloc[:,i] - np.min(data.iloc[:,i])) / (np.max(data.iloc[:,i]) - np.min(data.iloc[:,i]))
    return data





####### READ IN THE DATA SET #######
print("Reading in the data from the csv file...")
data = pd.read_csv('train.csv') # Assumes that the data is in the PWD
data = shuffleData(data)
train, val, test = splitData(data)

trainFeatures = train.iloc[:int(train.shape[0]/2),2:]
trainLabels = train.iloc[:int(train.shape[0]/2),1]
print("Using first %d rows to train..." % int(train.shape[0]/2))

valFeatures = val.iloc[:,2:]
valLabels = val.iloc[:,1]

testFeatures = test.iloc[:,2:]
testLabels = test.iloc[:,1]



######## TENSOR FLOW NETWORK BUILDING #######

### Hyper parameters ###
learningRate = 0.1
itterations = 10000
batchSize = 10000
displayStep = 2

n_nodes_hl1 = 59
n_nodes_hl2 = 59
n_nodes_hl3 = 59

### Inputs ###
x = tf.placeholder("float", [None, 57])
y = tf.placeholder("float", [None, 1])

### Create a model ###

def neural_network_model(data, threshold=0.5):

    ### Initialize Weights ###
    hidden_1_layer = {'weights':tf.Variable(tf.cast(tf.random_normal([57, n_nodes_hl1]), tf.float64)),
                      'biases':tf.Variable(tf.cast(tf.random_normal([n_nodes_hl1]), tf.float64))}

    hidden_2_layer = {'weights':tf.Variable(tf.cast(tf.random_normal([n_nodes_hl1, n_nodes_hl2]), tf.float64)),
                      'biases':tf.Variable(tf.cast(tf.random_normal([n_nodes_hl2]), tf.float64))}

    hidden_3_layer = {'weights':tf.Variable(tf.cast(tf.random_normal([n_nodes_hl2, n_nodes_hl3]), tf.float64)),
                      'biases':tf.Variable(tf.cast(tf.random_normal([n_nodes_hl3]), tf.float64))}

    output_layer = {'weights':tf.Variable(tf.cast(tf.random_normal([n_nodes_hl3, 1]), tf.float64)),
                      'biases':tf.Variable(tf.cast(tf.random_normal([1]), tf.float64))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden_2_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    output = tf.nn.relu(output)

    # for answer in output:
    #     if answer >= threshold:
    #         answer = 1
    #     else:
    #         answer = 0

    return output


testPredictions = neural_network_model(testFeatures)
print("Output Data: ")
tf.Print(testPredictions, [testPredictions])





# ### Set model weights ###
# W = tf.Variable(tf.zeros([59, 10]))
# b = tf.Variable(tf.zeros([10]))

# with tf.name_scope("Wx_b") as scope:
#     ### Construct a linear model ###
#     model = tf.nn.softmax(tf.matmul(x, W) + b)

# ### Add summary ops to collect data ###
# w_h = tf.summary.histogram("weights", W)
# b_h = tf.summary.histogram("biases", b)

# with tf.name_scope("cost_function") as scope:
#     ### Minimize error using cross entropy ###
#     cost_function = -tf.reduce_sum(y*tf.log(model))

#     ### Create a summary to monitor the cost function
#     tf.scalar_summary("cost_function", cost_function)

# with tf.name_scope("train") as scope:
#     ### Gradient Descent ###
#     optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_function)


# ### Initialize the variables ###
# init = tf.initialize_all_tables()

# ### Merge all summaries into a single operator ###
# merged_summary_op = tf.merge_all_summaries()

# ### Launch the graph ###
# with tf.Session() as sess:
#     sess.run(init)

#     ### Set the logs writer to the folder
#     summary_writer = tf.train.SummaryWriter('./', graph_def = sess.graph_def)
    

#     ### Training cycle ###
#     for iteration in range(iterations):
#         avg_cost = 0
#         total_batch = int(trainFeatures.shape[0] / batchSize)

#         ### Loop over all batches
#         for i in range(total_batch):
#             batch_xs = trainFeatures.iloc[i:i+batchSize, :]
#             batch_ys = trainLabels.iloc[i:i+batchSize, :]

#             sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

#             avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})
            

#             summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
#             summary_writer.add_summary(summary_str, iteration=total_batch + i)

#         if iteration % displayStep == 0:
#             print("Iteration: %04d" % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

# print("Tuning COmpleted!")

# ####### TEST THE MODEL #######
# predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))

# accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
# print("Accuracy: ", accuracy.eval({x: testFeatures, y: testLabels}))


