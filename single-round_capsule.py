import os
from random import *
# import input_data_intent
import numpy as np
import tensorflow as tf
import model_1 as model
import tool
import math
from sklearn.metrics import classification_report
from scipy.spatial import distance
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.metrics import accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
a = Random();
a.seed(1)

#-*- coding=utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import json
import jieba
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import os.path
import random
import pandas as pd
import cPickle as pickle
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import *
from keras.optimizers import *
from keras.utils import *
from keras.regularizers import *
from keras.layers import *
from keras.layers import LSTM
from keras.layers.core import *
from keras.layers.recurrent import *
from keras.layers.wrappers import *
from keras.layers.normalization import *
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.backend import stack
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from pprint import pprint



def acc(y_true, y_pred):
    return K.mean(K.equal(y_true > 0.5, y_pred > 0.5))


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def sequence2vec(qustion, word2vec, params):
    tmp = {}
    tmp['matrix'] = []
    question_words = jieba.lcut(qustion)
    for i in question_words:
        # print i
        if i in word2vec:
            tmp['matrix'].append(word2vec[i])
    tmp['matrix'] = np.array(tmp['matrix'])
    if tmp['matrix'].size == 0:
        tmp['result_matrix'] = []
    else:
        if tmp['matrix'].shape[0] < params['seq_max_len']:
            tmp['result_matrix'] = np.zeros((params['seq_max_len'], 200))
            tmp['result_matrix'][:tmp['matrix'].shape[0], :tmp['matrix'].shape[1]] = tmp['matrix']
        else:
            tmp['result_matrix'] = tmp['matrix'][:params['seq_max_len'], :]
    return tmp['result_matrix']


def load_data(word2vec, params):
    X_D = []
    Y_D = []
    X_A = []
    Y_A = []
    rewards = []
    dic = {}
    temp = {}
    dic['count_a_suggestion'] = {}
    dic['result_suggestion_matrix'] = {}
    dic['y'] = {}
    dic_d_a = {}

    # Build suggestion dictionary and count numbers
    with open('userLogs_large.json', 'r') as file:
        for line in file:
            j_line = json.loads(line)
            # If record is not suggest question, skip and save temp
            if j_line['a_type'] != unicode('建议问'):
                temp['q_user'] = j_line['q_user']
                temp['user_id'] = j_line['user_id']
                temp['topic'] = j_line['topic']
            # If record is a suggest question, check and save valid result
            if j_line['a_type'] == unicode('建议问'):
                if temp['q_user'] in j_line['a_suggestion']:
                    temp['matrix'] = []
                    question_words = jieba.lcut(j_line['q_user'])
                    for word in question_words:
                        if word in word2vec:
                            temp['matrix'].append(word2vec[word])
                    temp['matrix'] = np.array(temp['matrix'])
                    if temp['matrix'].size == 0:
                        continue
                    else:
                        for suggestion in j_line['a_suggestion']:
                            if suggestion not in dic['count_a_suggestion']:
                                    dic['count_a_suggestion'][suggestion] = 1
                            else:
                                if suggestion == temp['q_user']:
                                    dic['count_a_suggestion'][suggestion] += 1
                    # count_total += 1
                    # if count_total == 1000:
                    #     break

    count_total = 0
    with open('userLogs_large.json', 'r') as file1:
        y_id = 0
        d_id = 0
        for line in file1:
            j_line = json.loads(line)
            if j_line['a_type'] != unicode('建议问'):
                temp['q_user'] = j_line['q_user']
                temp['user_id'] = j_line['user_id']
                temp['topic'] = j_line['topic']
            if j_line['a_type'] == unicode('建议问'):
                if temp['q_user'] in j_line['a_suggestion']:
                    # if temp_lable not in topic:
                    #     topic.append(temp_lable)
                    # jieba
                    # print j_line['q_user']
                    temp['result_question_matrix'] = sequence2vec(j_line['q_user'], word2vec, params)
                    if len(temp['result_question_matrix']) == 0:
                        continue
                    else:
                        # jianyiwen
                        d_id_array = []
                        a_id_array = []
                        for suggestion in j_line['a_suggestion']:
                            if dic['count_a_suggestion'][suggestion] < params['count']:
                                continue
                            # print a
                            # print temp_q
                            if suggestion not in dic['y']:
                                temp['result_suggestion_matrix'] = sequence2vec(suggestion, word2vec, params)
                                dic['y'][suggestion] = y_id
                                y_id += 1
                                dic['result_suggestion_matrix'][suggestion] = temp['result_suggestion_matrix']
                                X_A.append(temp['result_suggestion_matrix'])
                                Y_A.append(dic['y'][suggestion])
                            else:
                                temp['result_suggestion_matrix'] = dic['result_suggestion_matrix'][suggestion]

                            if suggestion == temp['q_user']:
                                rewards.append(1)
                            else:
                                rewards.append(0)
                            X_D.append(np.concatenate((temp['result_question_matrix'], temp['result_suggestion_matrix']), axis=0))
                            Y_D.append(dic['y'][suggestion])
                            d_id_array.append(d_id)
                            d_id += 1
                            a_id_array.append(dic['y'][suggestion])
                            #print j_line['q_user'], temp_q, a, dic_y[a] #tmp_res_d[0], tmp_res_a[0]
                        # print d_id_array, a_id_array
                        for idx in d_id_array:
                            dic_d_a[idx] = a_id_array

                    count_total += 1
                    # if count_total == 1000:
                    #     break
    # print len(X), len(Y)
    #
    # for i in range(len(X)):
    #     print np.array(X[i]).shape, Y[i]
    print rewards
    print len(X_D), len(Y_D), len(X_A), len(Y_A)
    return X_D, Y_D, X_A, Y_A, rewards, dic_d_a, len(X_A)


# def modelContext(X, Y, model, size):
#     context = np.array(model.predict(np.array(X).reshape((len(X),params['seq_max_len']*size, X[0].shape[1]))))[2]
#     print context.shape
#     return context


def onlineLearning(contextD, contextA, rewards, dic_d_a):
    print rewards
    n_a=len(contextA)  # number of actions
    D= contextD
    n=len(D)  # number of data points
    k=len(D[0])   # number of features
    print k, n_a
    # our data
    th=np.random.random( (n_a,k) ) - 0.5      # our real theta, what we will try to guess/

    P=D.dot(th.T)
    print P[0]
    import matplotlib.pyplot as plt
    optimal=np.array(P.argmax(axis=1), dtype=int)
    plt.title("Distribution of ideal arm choices")
    plt.hist(optimal,bins=range(0,n_a))

    eps=0.3
    choices=np.zeros(n)
    b      =np.zeros_like(th)
    A      =np.zeros( (n_a, k,k)  )
    for a in range (0,n_a):
        A[a]=np.identity(k)
    th_hat =np.zeros_like(th) # our temporary feature vectors, our best current guesses
    p      =np.zeros(n_a)
    alph   =0.2

    j = 0.1
    # LinUCB, using a disjoint model
    # This is all from Algorithm 1, p 664, "A contextual bandit approach..." Li, Langford
    tmp_index = -1
    prev = dic_d_a[0]
    regret = []
    for i in range(0,n):
        x_i = D[i]   # the current context vector
        # print i, dic_d_a[i], rewards[i]
        if np.array_equal(prev, dic_d_a[i]) and tmp_index < len(dic_d_a[i]) - 1:
            tmp_index += 1
        else:
            prev = dic_d_a[i]
            tmp_index = 0
        #for a in range (0,n_a):
        for a in dic_d_a[i]:
            A_inv      = np.linalg.inv(A[a])        # we use it twice so cache it
            th_hat[a]  = A_inv.dot(b[a])            # Line 5
            ta         = x_i.dot(A_inv).dot(x_i)    # how informative is this ?
            a_upper_ci = alph * np.sqrt(ta)         # upper part of variance interval
            a_mean     = th_hat[a].dot(x_i)         # current estimate of mean
            p[a]       = a_mean + a_upper_ci        # top CI

        # Let's not be biased with tiebreaks, but add in some random noise
        p= p + ( np.random.random(len(p)) * 0.000001)
        p_tmp =[]
        for idx in dic_d_a[i]:
            p_tmp.append(p[idx])
        if random.uniform(0, 1) > eps:
            choices[i] = dic_d_a[i][np.array(p_tmp).argmax()]
        else:
            choices[i] = random.choice(dic_d_a[i]) # choose the highest, line 11
        a = int(choices[i])
        # print a, dic_d_a[i][tmp_index], rewards[i], tmp_index
        if rewards[i] == 1:
            if a != dic_d_a[i][tmp_index]:
                regret.append(1)
            else:
                regret.append(0)

        # see what kind of result we get
        #rewards[i] = th[a].dot(x_i)  # using actual theta to figure out reward
        # update the input vector
        A[a]      += np.outer(x_i,x_i)
        b[a]      += rewards[i] * x_i

    # print P.max(axis=1).shape
    # regret=(P.max(axis=1) - rewards)
    print len(regret), regret.count(1), regret.count(0)
    # plt.subplot(122)
    # plt.plot(np.array(regret).cumsum())
    # plt.title("Cumulative Regret")
    # plt.show()


def setting(data):
    vocab_size, word_emb_size = data['embedding'].shape
    sample_num, max_time = data['x_tr'].shape
    test_num = data['x_te'].shape[0]
    s_cnum = np.unique(data['y_tr']).shape[0]
    u_cnum = np.unique(data['y_te']).shape[0]

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_float("keep_prob", 0.8, "dropout keep probability")
    tf.app.flags.DEFINE_integer("hidden_size", 200, "embedding vector size")
    tf.app.flags.DEFINE_integer("batch_size", 64, "vocab size of word vectors")
    tf.app.flags.DEFINE_integer("num_epochs", 20, "num of epochs")
    tf.app.flags.DEFINE_integer("vocab_size", vocab_size, "vocab size of word vectors")
    tf.app.flags.DEFINE_integer("max_time", max_time, "max number of words in one sentence")
    tf.app.flags.DEFINE_integer("sample_num", sample_num, "vocab size of word vectors")
    tf.app.flags.DEFINE_integer("test_num", test_num, "vocab size of word vectors")
    tf.app.flags.DEFINE_integer("s_cnum", s_cnum, "seen class num")
    tf.app.flags.DEFINE_integer("u_cnum", u_cnum, "unseen class num")
    tf.app.flags.DEFINE_integer("word_emb_size", word_emb_size, "embedding size of word vectors")
    tf.app.flags.DEFINE_integer("display_step", 1, "display step")
    tf.app.flags.DEFINE_string("ckpt_dir", './models/' , "check point dir")
    tf.app.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not.")
    tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
    tf.app.flags.DEFINE_float("sim_scale", 1, "sim scale")
    tf.app.flags.DEFINE_float("margin", 2.0, "ranking loss margin")
    tf.app.flags.DEFINE_float("alpha", 0.01, "coefficient for self attention loss")
    tf.app.flags.DEFINE_integer("num_routing", 3, "capsule routing num")
    tf.app.flags.DEFINE_integer("output_atoms", 10, "capsule output atoms")
    tf.app.flags.DEFINE_boolean("save_model", True, "save model to disk")
    tf.app.flags.DEFINE_integer("d_a", 20, "self attention weight hidden units number")
    tf.app.flags.DEFINE_integer("r", 3, "self attention weight hops")
    return FLAGS

def get_sim(data):
    # get unseen and seen categories similarity
    #s = data['sc_vec']
    #u = data['uc_vec']
    s = normalize(data['sc_vec'])
    u = normalize(data['uc_vec'])
    # max = 1
    # other = 0
    #max_ind = np.argmax(sim, 1)
    #sim = np.zeros_like(sim, dtype='float32')
    #for i in range(len(max_ind)):
    #    sim[i, max_ind[i]] = 1
    return sim

def generate_batch(n, batch_size):
    batch_index = a.sample(xrange(n), batch_size)
    return batch_index

def assign_pretrained_word_embedding(sess, data, textRNN):
    print("using pre-trained word emebedding.begin...")
    embedding = data['embedding']

    word_embedding = tf.constant(embedding, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textRNN.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("using pre-trained word emebedding.ended...")

def squash(input_tensor):
    norm = tf.norm(input_tensor, axis=2, keep_dims=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))

def update_unseen_routing(votes, FLAGS, num_routing=3):
    votes_t_shape = [3, 0, 1, 2]
    r_t_shape = [1, 2, 3, 0]
    votes_trans = tf.transpose(votes, votes_t_shape)
    num_dims = 4
    input_dim = FLAGS.r
    output_dim = FLAGS.u_cnum
    input_shape = tf.shape(votes)
    logit_shape = tf.stack([input_shape[0], input_dim, output_dim])

    def _body(i, logits, activations):
        route = tf.nn.softmax(logits, dim=2)
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1)
        activation = squash(preactivate)
        activations = activations.write(i, activation)

        act_3d = tf.expand_dims(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = tf.tile(act_3d, tile_shape)
        distances = tf.reduce_sum(votes * act_replicated, axis=3)
        logits += distances
        return (i + 1, logits, activations)

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)
    i = tf.constant(0, dtype=tf.int32)
    _, logits, activations = tf.while_loop(
        lambda i, logits, activations: i < num_routing,
        _body,
        loop_vars=[i, logits, activations],
        swap_memory=True)

    return activations.read(num_routing - 1), logits

def evaluate_test(data, FLAGS, sess):
    # zero-shot testing state
    # seen votes shape (110, 2, 34, 10)
    x_te = data['x_te']
    y_te_id = data['y_te']
    u_len = data['u_len']

    [te_logits] = sess.run([lstm.logits],
            feed_dict={lstm.input_x: x_te, lstm.s_len: u_len})

    te_pred = np.argmax(te_logits, 1)

    #print "-----------full-batch zero-shot test classification report----------"
    acc = accuracy_score(y_te_id, te_pred)
    #print classification_report(y_te_id, te_pred, digits=4)
    #print "accuracy", acc
    #print confusion_matrix(y_te_id, te_pred)
    return acc

def dump_M(data, FLAGS, sess):
    x_tr = data['x_tr']
    y_tr_id = data['y_tr']
    s_len = data['s_len']

    [M] = sess.run([self.sentence_embedding],
            feed_dict={lstm.input_x: x_tr, lstm.s_len: s_len})
    return M

if __name__ == "__main__":
    params = {'count': 70,
          't': 5,
          'seq_max_len': 20,
          'batch_size': 256,
          'lr': 0.001,
          'dropout': 0.1,
          'n_epochs': 2,
          'n_hidden': 8,
          'n_latent': 8,
          'bias': 1,
          'is_clf': 1,
          'validation_split': 0.2,
          'data_size': 10,
          'loss': 'categorical_crossentropy'}

    # load data
    data = input_data_intent.read_datasets()
    x_tr = data['x_tr']
    y_tr = data['y_tr']
    y_tr_id = data['y_tr']

    y_te_id = data['y_te']
    y_ind = data['s_label']
    s_len = data['s_len']
    embedding = data['embedding']

    x_te = data['x_te']
    u_len = data['u_len']

    # load settings
    FLAGS = setting(data)

    # start
    tf.reset_default_graph()
    config=tf.ConfigProto()
    with tf.Session(config=config) as sess:
        # Instantiate Model
        lstm = model.lstm_model(FLAGS)


        if os.path.exists(FLAGS.ckpt_dir):
            print("Restoring Variables from Checkpoint for rnn model.")
            saver = tf.train.Saver()
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                assign_pretrained_word_embedding(sess, data, lstm)

        best_acc = 0.0
        cur_acc = evaluate_test(data, FLAGS, sess)
        if cur_acc > best_acc:
            best_acc = cur_acc

        var_saver = tf.train.Saver()
        # Training cycle
        batch_num = FLAGS.sample_num / FLAGS.batch_size
        for epoch in range(FLAGS.num_epochs):
            print("----------------epoch : ", epoch, "---------------")
            for batch in range(batch_num):
                batch_index = generate_batch(FLAGS.sample_num, FLAGS.batch_size)
                batch_x = x_tr[batch_index]
                batch_y_id = y_tr_id[batch_index]
                batch_len = s_len[batch_index]
                batch_ind = y_ind[batch_index]

                [sen, logits, _, loss] = sess.run([lstm.sentence_embedding, lstm.logits, lstm.train_op, lstm.loss_val],
                        feed_dict={lstm.input_x: batch_x, lstm.s_len: batch_len, lstm.IND: batch_ind})

            if (epoch >= FLAGS.num_epochs - 30):
                print(classification_report(batch_y_id, np.argmax(logits, 1)))
                cur_acc = evaluate_test(data, FLAGS, sess)
                if cur_acc > best_acc:
                    best_acc = cur_acc
                    var_saver.save(sess, os.path.join(FLAGS.ckpt_dir, "model.ckpt"), 1)
        M = dump_M(data, FLAGS, sess)
