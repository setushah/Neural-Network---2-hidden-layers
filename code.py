import tensorflow as tf
import numpy as np
import urllib
import sys
import os
import json



glove_worddict = {}

with open("glove.txt"
, "r",errors="ignore") as glove:
    for line in glove:
        name, vector = tuple(line.split(" ", 1))
        glove_worddict[name] = np.fromstring(vector, sep=" ")

def s2s(sentence):
   # sentence = remove_stop(sentence)
    tokens = sentence.lower().split(" ")
    row = []
    words = []
    #Greedy search for tokens
    for token in tokens[:15]:
        i = len(token)
        while len(token) > 0 and i > 0:
            word = token[:i]
            if word in glove_worddict:
                row += [item for item in glove_worddict[word]]
                words.append(word)
                token = ""
            else:
                i = i-1
    row += [0]*(750 - len(row))
    return row, words

#Constants setup

display_step = 10

def create_label(row):
    convert_dict = {
      'entailment': 0,
      'neutral': 1,
      'contradiction': 2
    }
    score = np.zeros((3,))
    for x in row['annotator_labels']:
        if x in convert_dict: score[convert_dict[x]] += 1
    return score / (1.0*np.sum(score))


def build_sentences(file_f):
    with open(file_f,"r") as data:
        jsonl_content = data.read()
        data_decoded = [json.loads(jline) for jline in jsonl_content.split('\n') if jline]
        pre_sentences = []
        hyp_sentences = []
        scores = []
        for row in data_decoded:
            hyp_sentences.append(s2s(row['sentence2'].lower())[0])
            pre_sentences.append(s2s(row['sentence1'].lower())[0])
            scores.append(create_label(row))
        hyp_sentences = np.stack(hyp_sentences)
        evi_sentences = np.stack(pre_sentences)
        scores = np.stack(scores)
        return (hyp_sentences, evi_sentences), scores

data_feature_list, correct_scores = build_sentences("train(1).jsonl")

test_data_feature_list, test_correct_scores = build_sentences("test(1).jsonl")
dev_data_feature_list, dev_correct_scores = build_sentences("dev(1).jsonl")

# Parameters
training_epochs = 100
batch_size = 50
display_step = 1
n_neurons_in_h1 = 500
n_neurons_in_g1 = 50
n_neurons_o_f = 300
learning_rate = 0.001
n_features = 15 * 50
n_class = 3

# placeholders
X1 = tf.placeholder(tf.float32, [None, n_features], name='premise')
X2 = tf.placeholder(tf.float32, [None, n_features], name='hypothesis')
Y = tf.placeholder(tf.float32, [None, n_class], name='gold_label')


# Create model
def neural_network():
    # First Hidden layer of F(Premise)
    W1 = tf.Variable(tf.truncated_normal([n_features, n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)),
                     name='weights1')
    b1 = tf.Variable(tf.truncated_normal([n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)), name='biases1')
    y1 = tf.nn.tanh((tf.matmul(X1, W1) + b1), name='activationHiddenLayer1')

    # Output layer of F(Premise)
    Wo = tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_o_f], mean=0, stddev=1 / np.sqrt(n_features)),
                     name='weightsOut')
    bo = tf.Variable(tf.random_normal([n_neurons_o_f], mean=0, stddev=1 / np.sqrt(n_features)), name='biasesOut')
    a_premise = tf.nn.tanh((tf.matmul(y1, Wo) + bo), name='activationOutputLayer')

    # First Hidden layer of F(Hypothesis)
    W1_h = tf.Variable(tf.truncated_normal([n_features, n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)),
                       name='weights1')
    b1_h = tf.Variable(tf.truncated_normal([n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)), name='biases1')
    y2 = tf.nn.tanh((tf.matmul(X2, W1_h) + b1_h), name='activationHiddenLayer1')

    # Output layer of F(Hypothesis)
    Wo_h = tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_o_f], mean=0, stddev=1 / np.sqrt(n_features)),
                       name='weightsOut')
    bo_h = tf.Variable(tf.random_normal([n_neurons_o_f], mean=0, stddev=1 / np.sqrt(n_features)), name='biasesOut')
    a_hypothesis = tf.nn.tanh((tf.matmul(y2, Wo_h) + bo_h), name='activationOutputLayer')

    f_output = tf.concat([a_premise, a_hypothesis], 1)

    # Hidden layer of G
    W1_g = tf.Variable(
        tf.truncated_normal([n_neurons_o_f * 2, n_neurons_in_g1], mean=0, stddev=1 / np.sqrt(n_features)),
        name='weights1')
    b1_g = tf.Variable(tf.truncated_normal([n_neurons_in_g1], mean=0, stddev=1 / np.sqrt(n_features)), name='biases1')
    y1_g = tf.nn.tanh((tf.matmul(f_output, W1_g) + b1_g), name='activationHiddenLayer1')

    # output layer of G
    Wo_g = tf.Variable(tf.random_normal([n_neurons_in_g1, n_class], mean=0, stddev=1 / np.sqrt(n_features)),
                       name='weightsOut')
    bo_g = tf.Variable(tf.random_normal([n_class], mean=0, stddev=1 / np.sqrt(n_features)), name='biasesOut')
    a_g = tf.nn.tanh((tf.matmul(y1_g, Wo_g) + bo_g), name='activationOutputLayer')

    a_g = tf.nn.softmax(a_g)
    return a_g


# Construct model
output = neural_network()

cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(output), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    iterations = 0
    for epoch in range(training_epochs):
        iterations += 1
        avg_cost = 0.
        total_batch = int(data_feature_list[0].shape[0] / batch_size)
        for i in range(total_batch):
            data_x1 = data_feature_list[0][i * batch_size: (i + 1) * batch_size]
            data_x2 = data_feature_list[1][i * batch_size: (i + 1) * batch_size]
            data_y = correct_scores[i * batch_size: (i + 1) * batch_size]
            _, c = sess.run([train_step, cross_entropy], feed_dict={X1: data_x1,
                                                                    X2: data_x2,
                                                                    Y: data_y})
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")


    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
    print("Accuracy:", accuracy.eval({
        X1: data_feature_list[0],
        X2: data_feature_list[1],
        Y: correct_scores}))
